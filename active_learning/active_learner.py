"""
Active Learning Main Loop
- 整合所有组件
- 实现完整的主动学习流程
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict
from pathlib import Path
import json
from tqdm import tqdm

# 兼容直接运行和作为包导入两种方式
try:
    from .feature_extractor import MultiLayerDINOv3
    from .classifier import SimpleClassifier, train_epoch, evaluate
    from .memory_bank import MemoryBank
    from .uncertainty import UncertaintyEstimator
    from .cascading_selector_v1 import CascadingSelectorV1
    from .dataset import ActiveLearningDataset
    from .prediction_uncertainty import PredictionUncertaintyEstimator, HybridUncertaintyFusion
except ImportError:
    from feature_extractor import MultiLayerDINOv3
    from classifier import SimpleClassifier, train_epoch, evaluate
    from memory_bank import MemoryBank
    from uncertainty import UncertaintyEstimator
    from cascading_selector_v1 import CascadingSelectorV1
    from dataset import ActiveLearningDataset
    from prediction_uncertainty import PredictionUncertaintyEstimator, HybridUncertaintyFusion


def _get_config():
    """获取 Config 对象"""
    try:
        from .config import Config
    except ImportError:
        from config import Config
    return Config


class ActiveLearner:
    """
    主动学习器
    
    整合所有组件，实现完整的主动学习流程
    """
    
    def __init__(
        self,
        feature_extractor: MultiLayerDINOv3,
        classifier: SimpleClassifier,
        memory_bank: MemoryBank,
        uncertainty_estimator: UncertaintyEstimator,
        selector: CascadingSelectorV1,
        device: str = None,
        num_classes: int = None,
        use_prediction_uncertainty: bool = None,
        fusion_strategy: str = None,
        random_sampling: bool = None
    ):
        """
        Args:
            feature_extractor: DINOv3特征提取器
            classifier: 分类头
            memory_bank: Memory Bank
            uncertainty_estimator: 不确定性估计器
            selector: 样本选择器
            device: 设备
            num_classes: 类别数量
            use_prediction_uncertainty: 是否使用预测不确定性（闭环）
                                        如果为 None，从 Config 读取
            fusion_strategy: 融合策略 ('attention', 'multiply', 'add')
                            如果为 None，从 Config 读取
            random_sampling: 是否使用随机采样（baseline）
                            如果为 None，从 Config 读取
        """
        Config = _get_config()
        
        # 如果参数为 None，从 Config 读取
        if device is None:
            device = getattr(Config, 'device', 'cuda')
        if num_classes is None:
            num_classes = getattr(Config, 'num_classes', 100)
        if use_prediction_uncertainty is None:
            use_prediction_uncertainty = getattr(Config, 'use_prediction_uncertainty', False)
        if fusion_strategy is None:
            fusion_strategy = getattr(Config, 'fusion_strategy', 'attention')
        if random_sampling is None:
            random_sampling = getattr(Config, 'random_sampling', False)
        
        self.feature_extractor = feature_extractor.to(device)
        self.classifier = classifier.to(device)
        self.memory_bank = memory_bank
        self.uncertainty_estimator = uncertainty_estimator
        self.selector = selector
        self.device = device
        self.use_prediction_uncertainty = use_prediction_uncertainty
        self.random_sampling = random_sampling
        
        # 确保特征提取器冻结
        self.feature_extractor.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # 打印当前模式
        if random_sampling:
            print("  Random Sampling Mode (Baseline)")
            print("  No uncertainty calculation, purely random selection")
        elif use_prediction_uncertainty:
            # 预测不确定性组件（闭环模式）
            self.prediction_uncertainty_estimator = PredictionUncertaintyEstimator(
                num_classes=num_classes
            )
            # HybridUncertaintyFusion 会自动从 Config 读取配置
            self.hybrid_fusion = HybridUncertaintyFusion()
            
            # 打印当前配置
            pred_mode = getattr(Config, 'prediction_uncertainty_mode', 'entropy')
            print("  Prediction Uncertainty Enabled (Closed-Loop Mode)")
            print(f"  Prediction Uncertainty Mode: {pred_mode}")
            print(f"  Fusion Strategy: {fusion_strategy}")
        else:
            print("  Only Using Feature Uncertainty (Open-Loop Mode)")
        
        # 历史记录
        self.history = {
            'rounds': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'labeled_samples': [],
            'selected_priorities': [],
            'class_distribution': [],  # 方向1：记录每轮选中样本的类别分布
            'per_class_val_accuracy': []  # 记录每轮验证时各类别的准确率
        }
    
    def train_classifier(
        self,
        train_loader: DataLoader,
        num_epochs: int = None,
        learning_rate: float = None,
        verbose: bool = True
    ) -> float:
        """
        训练分类头
        
        Args:
            train_loader: 训练数据加载器
            num_epochs: 训练轮数
            learning_rate: 学习率
            verbose: 是否打印训练信息
        
        Returns:
            final_loss: 最后一个epoch的损失
        """
        Config = _get_config()
        
        # 从 Config 读取默认值
        if num_epochs is None:
            num_epochs = getattr(Config, 'classifier_epochs', 10)
        if learning_rate is None:
            learning_rate = getattr(Config, 'classifier_lr', 0.001)
        
        # 从 Config 读取日志间隔
        log_interval = getattr(Config, 'log_interval', 5)
        
        optimizer = optim.Adam(self.classifier.parameters(), lr=learning_rate)
        
        if verbose:
            print(f"  Start Training Classifier ({num_epochs} epochs)...")
        
        for epoch in range(num_epochs):
            avg_loss = train_epoch(
                self.classifier,
                self.feature_extractor,
                train_loader,
                optimizer,
                self.device
            )
            
            if verbose and (epoch + 1) % log_interval == 0:
                print(f"    Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    @torch.no_grad()
    def extract_features_from_loader(
        self,
        dataloader: DataLoader,
        return_labels: bool = True,
        desc: str = "提取特征"
    ) -> Dict[str, torch.Tensor]:
        """
        从DataLoader中提取所有样本的特征
        
        Args:
            dataloader: 数据加载器
            return_labels: 是否返回标签
            desc: 进度条描述
        
        Returns:
            features_dict: {
                'cls': [N, 768],
                'layer_3': [N, num_patches, 768],
                ...,
                'labels': [N] (可选)
            }
        """
        # 设置为评估模式（关闭 Dropout 等训练专用行为）
        self.feature_extractor.eval()
        
        # ===== 准备收集容器 =====
        # 数据集太大无法一次性送入GPU，所以分批(batch)处理，
        # 先用列表收集每批的结果，最后再拼接成完整的矩阵
        
        all_cls_features = []       # 收集每个batch的CLS特征，最终拼成 [N, 768]
        # 为每一层创建一个空列表，用于收集该层的特征
        # 例如 layers=[3,6,9,11] → {'layer_3': [], 'layer_6': [], 'layer_9': [], 'layer_11': []}
        all_multi_layer_features = {f'layer_{i}': [] for i in self.memory_bank.layers}
        all_labels = []             # 收集标签（如果需要的话）
        
        # ===== 统计信息（用于进度条显示） =====
        total_batches = len(dataloader)                     # 总共有多少个batch
        # 总样本数：优先从 dataset 获取精确值，否则用 batch数 × batch_size 估算
        total_samples = len(dataloader.dataset) if hasattr(dataloader, 'dataset') else total_batches * dataloader.batch_size
        
        # ===== 逐batch提取特征 =====
        # tqdm 创建进度条，leave=False 表示完成后进度条消失不占屏幕
        pbar = tqdm(dataloader, desc=desc, total=total_batches, leave=False)
        for batch_idx, batch in enumerate(pbar):
            # 从batch中取出数据
            # DataLoader 返回 (images, labels) 或 (images,)，取决于数据集
            if return_labels:
                images, labels = batch      # 已标注数据：需要图像和标签
                all_labels.append(labels)
            else:
                images = batch[0]           # 未标注数据：只需要图像
            
            # 将图像搬到GPU上（DINOv3在GPU上运行）
            images = images.to(self.device)
            
            # 调用DINOv3提取特征
            # return_cls_only=False → 返回CLS + 各层特征（不只是CLS）
            # pool_patches=True → 对patch特征做平均池化，节省内存
            #   池化前: layer_3 形状 [B, 196, 768]（196个patch各自768维）
            #   池化后: layer_3 形状 [B, 768]（196个patch平均成1个向量）
            features = self.feature_extractor(images, return_cls_only=False, pool_patches=True)
            
            # 把这个batch的特征搬回CPU，存入收集列表
            # （GPU显存有限，特征提取完就搬回CPU释放显存给下一个batch用）
            all_cls_features.append(features['cls'].cpu())          # CLS特征 [B, 768]
            
            for layer_key in all_multi_layer_features.keys():       # 各层特征
                all_multi_layer_features[layer_key].append(
                    features[layer_key].cpu()                        # 每层 [B, 768]
                )
            
            # 更新进度条右侧的附加信息
            processed = (batch_idx + 1) * images.size(0)   # images.size(0) = 这个batch的样本数
            pbar.set_postfix({
                'Processed': f'{processed}/{total_samples}',
                'batch': f'{batch_idx+1}/{total_batches}'
            })
            
            # 每处理50个batch，清理一次GPU缓存，防止显存碎片积累
            if (batch_idx + 1) % 50 == 0:
                torch.cuda.empty_cache()
        
        pbar.close()
        
        # ===== 合并所有batch的结果 =====
        # 上面循环结束后，列表里存的是每个batch的小矩阵，例如：
        #   all_cls_features = [ [4,768], [4,768], [4,768], ... ]  ← 每个元素是一个batch
        # torch.cat 沿第0维(样本维)拼接，把它们合并成一个完整的大矩阵：
        #   [4,768] + [4,768] + ... → [N, 768]
        print(f"  Combining Features...")
        result = {
            'cls': torch.cat(all_cls_features, dim=0)
        }
        
        # 同理，拼接各层特征：layer_3, layer_6, layer_9, layer_11 各自拼成 [N, 768]
        for layer_key, features_list in all_multi_layer_features.items():
            result[layer_key] = torch.cat(features_list, dim=0)
        
        # 如果需要标签，也拼接起来：[N] 每个样本对应的类别编号
        if return_labels:
            result['labels'] = torch.cat(all_labels, dim=0)
        
        print(f"  Feature Extraction Completed: {result['cls'].shape[0]} samples")
        
        # 最终返回的 result 字典结构：
        # {
        #     'cls':     [N, 768],    ← CLS全局特征（给分类器和Memory Bank用）
        #     'layer_3': [N, 768],    ← 第3层特征（给多尺度不确定性用）
        #     'layer_6': [N, 768],    ← 第6层特征
        #     'layer_9': [N, 768],    ← 第9层特征
        #     'layer_11':[N, 768],    ← 第11层特征
        #     'labels':  [N]          ← 标签（仅 return_labels=True 时有）
        # }
        return result
    
    def select_samples(
        self,
        unlabeled_loader: DataLoader,
        budget: int,
        use_prediction_uncertainty: bool = False,
        current_round: int = None,
        total_rounds: int = None
    ) -> tuple:
        """
        从unlabeled pool中选择样本
        
        Args:
            unlabeled_loader: unlabeled数据加载器
            budget: 选择的样本数量
            use_prediction_uncertainty: 是否使用预测不确定性（闭环）
            current_round: 当前轮次（用于动态权重调整，可选）
            total_rounds: 总轮次（用于动态权重调整，可选）
        
        Returns:
            selected_indices: 选中的样本索引（在unlabeled pool中的相对索引）
            priorities: 对应的优先级
        """
        # ========== 随机采样模式 (Baseline) ==========
        if self.random_sampling:
            print(f"  Random Sampling Mode: selecting {budget} samples randomly...")
            num_unlabeled = len(unlabeled_loader.dataset)
            
            # 随机打乱索引，取前 budget 个
            selected_indices = torch.randperm(num_unlabeled)[:budget]
            
            # 优先级全部设为 0（无意义，仅占位）
            priorities = torch.zeros(len(selected_indices), dtype=torch.long)
            
            print(f"  Randomly selected {len(selected_indices)} samples")
            return selected_indices, priorities
        
        # ========== 主动学习模式（开环/闭环） ==========
        # 提取unlabeled样本的特征
        print(f"  Getting Unlabeled Samples Features...")
        unlabeled_features = self.extract_features_from_loader(
            unlabeled_loader,
            return_labels=False,
            desc="  Getting Unlabeled Samples Features"
        )
        
        # 计算特征不确定性
        print(f"  Computing Feature Uncertainty...")
        feature_uncertainties = self.uncertainty_estimator.compute_all_uncertainties(
            cls_features=unlabeled_features['cls'].to(self.device),
            multi_layer_features={
                k: v.to(self.device) 
                for k, v in unlabeled_features.items() 
                if k != 'cls'
            },
            normalize=True
        )
        
        # 如果启用预测不确定性，计算并融合
        if use_prediction_uncertainty and hasattr(self, 'prediction_uncertainty_estimator'):
            print(f"  Computing Prediction Uncertainty...")
            # 用分类头预测
            with torch.no_grad():
                logits = self.classifier(unlabeled_features['cls'].to(self.device))
            
            # 计算预测不确定性
            prediction_uncertainties = self.prediction_uncertainty_estimator.compute_all(
                logits, normalize=True
            )
            
            # 融合特征不确定性和预测不确定性
            print(f"  Fusing Uncertainties (Dynamic Attention Modulation)...")
            final_uncertainty = self.hybrid_fusion.fuse(
                feature_uncertainties,
                prediction_uncertainties,
                current_round=current_round,
                total_rounds=total_rounds
            )
            
            # 方案A：使用融合后的不确定性 + Density过滤
            print(f"  Applying Density Filter (Noise Protection)...")
            u_density = feature_uncertainties['density']
            
            # 过滤噪声：只保留density不确定性低于0.7的样本（可能是噪声）
            Config = _get_config()
            noise_threshold = getattr(Config, 'noise_threshold', 0.7)
            
            # 过滤噪声：只保留density不确定性低于阈值的样本
            noise_mask = u_density < noise_threshold  # True = 非噪声
            num_valid = noise_mask.sum().item()
            num_total = len(noise_mask)
            
            print(f"    Before Filtering: {num_total} samples")
            print(f"    After Filtering: {num_valid} samples (filtered out {num_total - num_valid} noise samples)")
            
            if num_valid >= budget:
                # 有足够的非噪声样本
                valid_indices = torch.where(noise_mask)[0]
                valid_uncertainties = final_uncertainty[noise_mask]
                
                # 从非噪声样本中选择top-k
                topk_values, topk_local_indices = torch.topk(
                    valid_uncertainties, 
                    k=min(budget, len(valid_uncertainties))
                )
                selected_indices = valid_indices[topk_local_indices]
                
                print(f"    Selected {len(selected_indices)} samples from {num_valid} non-noise samples")
            else:
                # 非噪声样本不够，放宽条件
                print(f"    Not enough non-noise samples ({num_valid} < {budget}), relaxing filtering conditions")
                selected_indices = torch.topk(
                    final_uncertainty, 
                    k=min(budget, len(final_uncertainty))
                )[1]
            
            # 保证闭环模式里面select_samples函数的返回值结构一致
            priorities = torch.full((len(selected_indices),), 4, dtype=torch.long)  # VERY_HIGH
            
        else:
            # 只使用特征不确定性（开环模式）
            # 使用级联选择器
            selected_indices, priorities = self.selector.select_samples(
                uncertainties=feature_uncertainties,
                budget=budget,
                allow_noise=False
            )
        
        return selected_indices, priorities
    
    def update_memory_bank(
        self,
        labeled_loader: DataLoader
    ):
        """
        更新Memory Bank（添加新标注的样本）
        
        Args:
            labeled_loader: labeled数据加载器
        """
        # 提取labeled样本的特征
        print(f"  Extracting Labeled Samples Features...")
        labeled_features = self.extract_features_from_loader(
            labeled_loader,
            return_labels=True,
            desc="  Extracting Labeled Samples Features"
        )
        
        # 更新Memory Bank
        self.memory_bank.add_samples(
            cls_features=labeled_features['cls'],
            multi_layer_features={
                k: v 
                for k, v in labeled_features.items() 
                if k not in ['cls', 'labels']
            },
            labels=labeled_features['labels']
        )
    
    def run_one_round(
        self,
        round_num: int,
        al_dataset: ActiveLearningDataset,
        val_loader: DataLoader,
        budget: int,
        train_epochs: int = 10,
        learning_rate: float = 0.001
    ) -> Dict:
        """
        运行一轮主动学习
        
        Args:
            round_num: 轮次编号
            al_dataset: 主动学习数据集
            val_loader: 验证数据加载器
            budget: 每轮选择的样本数量
            train_epochs: 训练轮数
            learning_rate: 学习率
        
        Returns:
            metrics: 本轮的指标
        """
        print(f"\n{'='*60}")
        print(f"Round {round_num}")
        print(f"{'='*60}")
        
        # 1. 训练分类头
        print(f"1. Training Classifier...")
        labeled_loader = al_dataset.get_labeled_loader(shuffle=True)
        
        if labeled_loader is not None:
            train_loss = self.train_classifier(
                labeled_loader,
                num_epochs=train_epochs,
                learning_rate=learning_rate,
                verbose=True
            )
        else:
            print("  No labeled samples, skipping training")
            train_loss = 0.0
        
        # 2. 评估当前性能
        print(f"\n2. Evaluating Current Performance...")
        train_metrics = evaluate(
            self.classifier,
            self.feature_extractor,
            labeled_loader,
            self.device
        ) if labeled_loader is not None else {'accuracy': 0.0, 'loss': 0.0}
        
        # 验证集评估，根据配置决定是否获取每个类别的准确率
        Config = _get_config()
        record_per_class = getattr(Config, 'record_per_class_accuracy', False)
        val_metrics = evaluate(
            self.classifier,
            self.feature_extractor,
            val_loader,
            self.device,
            return_per_class=record_per_class,
            num_classes=Config.num_classes
        )
        
        print(f"  Training Accuracy: {train_metrics['accuracy']:.2f}%")
        print(f"  Validation Accuracy: {val_metrics['accuracy']:.2f}%")
        
        # 打印类别准确率统计（简要）
        if record_per_class:
            per_class_acc = val_metrics.get('per_class_accuracy', [])
            if per_class_acc:
                low_acc_classes = [i for i, acc in enumerate(per_class_acc) if acc < 50]
                print(f"  Classes with <50% accuracy: {len(low_acc_classes)} classes")
                if len(low_acc_classes) <= 10:
                    print(f"    Low accuracy classes: {low_acc_classes}")
        
        # 3. 选择新样本
        print(f"\n3. Selecting New Samples (budget={budget})...")
        unlabeled_loader = al_dataset.get_unlabeled_loader(shuffle=False)
        
        # 获取total_rounds用于动态权重调整
        Config = _get_config()
        total_rounds = getattr(Config, 'total_rounds', 20)
        
        if unlabeled_loader is not None and len(al_dataset.unlabeled_indices) > 0:
            selected_indices, priorities = self.select_samples(
                unlabeled_loader,
                budget=min(budget, len(al_dataset.unlabeled_indices)),
                use_prediction_uncertainty=self.use_prediction_uncertainty,
                current_round=round_num,
                total_rounds=total_rounds
            )
            
            print(f"  Selected {len(selected_indices)} samples")
            
            # 优先级分布（仅开环模式有意义，闭环模式的优先级是占位符）
            if not self.use_prediction_uncertainty:
                from collections import Counter
                priority_dist = Counter(priorities.cpu().numpy())
                print(f"  Priority Distribution:")
                for priority_val, count in sorted(priority_dist.items(), reverse=True):
                    priority_name = ['NOISE', 'LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH'][priority_val]
                    print(f"    {priority_name}: {count}")
            
            # 【方向1】统计选中样本的类别分布
            from collections import Counter
            train_dataset = al_dataset.dataset  # 注意：是 dataset 不是 train_dataset
            unlabeled_indices = al_dataset.unlabeled_indices
            
            # 获取选中样本在原始数据集中的真实索引
            selected_global_indices = [unlabeled_indices[i] for i in selected_indices.cpu().numpy()]
            # 获取这些样本的标签
            selected_labels = [train_dataset.targets[idx] for idx in selected_global_indices]
            
            # 统计类别分布
            class_dist = Counter(selected_labels)
            num_classes_covered = len(class_dist)
            min_count = min(class_dist.values())
            max_count = max(class_dist.values())
            
            print(f"  Class Distribution of Selected Samples:")
            print(f"    Classes Covered: {num_classes_covered}/{getattr(_get_config(), 'num_classes', 100)}")
            print(f"    Min/Max per Class: {min_count}/{max_count}")
            
            # 如果覆盖率低，打印警告
            Config = _get_config()
            if num_classes_covered < Config.num_classes * 0.5:
                print(f"    Warning: Low class coverage! Only {num_classes_covered} classes selected.")
            
            # 4. 添加到labeled pool
            print(f"\n4. Adding Samples to Labeled Pool...")
            al_dataset.add_labeled_samples(selected_indices.cpu().numpy().tolist())
            
            # 5. 更新Memory Bank
            print(f"\n5. Updating Memory Bank...")
            labeled_loader = al_dataset.get_labeled_loader(shuffle=False)
            # 只更新新添加的样本（这里简化处理，更新所有）
            self.memory_bank.clear()  # 清空后重新添加
            self.update_memory_bank(labeled_loader)
        else:
            print("  No unlabeled samples, skipping updating memory bank")
            selected_indices = []
            priorities = []
            class_dist = {}  # 空的类别分布
        
        # 6. 记录历史
        self.history['rounds'].append(round_num)
        self.history['train_accuracy'].append(train_metrics['accuracy'])
        self.history['val_accuracy'].append(val_metrics['accuracy'])
        self.history['labeled_samples'].append(len(al_dataset.labeled_indices))
        
        # 仅开环模式记录优先级（闭环都是4，随机都是0，都没意义）
        # 开环模式 = 非随机采样 且 不使用预测不确定性
        is_open_loop = (not self.random_sampling) and (not self.use_prediction_uncertainty)
        if is_open_loop:
            if hasattr(priorities, 'cpu') and len(priorities) > 0:
                self.history['selected_priorities'].append(
                    priorities.cpu().numpy().tolist()
                )
            elif isinstance(priorities, list) and len(priorities) > 0:
                self.history['selected_priorities'].append(priorities)
            else:
                self.history['selected_priorities'].append([])
        else:
            self.history['selected_priorities'].append([])
        
        # 【方向1】记录类别分布
        # 转换为可JSON序列化的格式：{class_id: count}
        class_dist_record = {
            'distribution': {str(k): v for k, v in class_dist.items()},
            'num_classes_covered': len(class_dist),
            'min_count': min(class_dist.values()) if class_dist else 0,
            'max_count': max(class_dist.values()) if class_dist else 0
        }
        self.history['class_distribution'].append(class_dist_record)
        
        # 记录每个类别的验证准确率（根据配置）
        if record_per_class:
            per_class_acc = val_metrics.get('per_class_accuracy', [])
            self.history['per_class_val_accuracy'].append(per_class_acc)
        
        # 返回指标
        metrics = {
            'round': round_num,
            'train_accuracy': train_metrics['accuracy'],
            'val_accuracy': val_metrics['accuracy'],
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'labeled_samples': len(al_dataset.labeled_indices),
            'unlabeled_samples': len(al_dataset.unlabeled_indices),
            'selected_samples': len(selected_indices)
        }
        
        return metrics
    
    def save_history(self, save_path: str):
        """
        保存历史记录
        
        特殊处理：per_class_val_accuracy 和 selected_priorities 的内层数组不换行，
        每轮数据占一行，方便对比查看
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 自定义格式化：内层列表不换行
        def format_history(history):
            lines = ['{']
            keys = list(history.keys())
            for i, key in enumerate(keys):
                value = history[key]
                comma = ',' if i < len(keys) - 1 else ''
                
                # 对于嵌套列表（如 per_class_val_accuracy, selected_priorities, class_distribution）
                # 内层元素放在一行
                if key == 'per_class_val_accuracy':
                    # per_class_val_accuracy: 保留2位小数，格式对齐
                    lines.append(f'  "{key}": [')
                    for j, item in enumerate(value):
                        item_comma = ',' if j < len(value) - 1 else ''
                        # 每个数值保留2位小数
                        formatted_item = [f'{v:.2f}' for v in item]
                        item_str = '[' + ','.join(formatted_item) + ']'
                        lines.append(f'    {item_str}{item_comma}')
                    lines.append(f'  ]{comma}')
                elif key == 'selected_priorities':
                    lines.append(f'  "{key}": [')
                    for j, item in enumerate(value):
                        item_comma = ',' if j < len(value) - 1 else ''
                        # 内层列表转为紧凑的一行
                        item_str = json.dumps(item, separators=(',', ':'))
                        lines.append(f'    {item_str}{item_comma}')
                    lines.append(f'  ]{comma}')
                elif key == 'class_distribution':
                    # class_distribution: distribution放一行，其他三个字段各自换行
                    lines.append(f'  "{key}": [')
                    for j, item in enumerate(value):
                        item_comma = ',' if j < len(value) - 1 else ''
                        # distribution 放一行
                        dist_str = json.dumps(item.get('distribution', {}), separators=(',', ':'))
                        num_covered = item.get('num_classes_covered', 0)
                        min_c = item.get('min_count', 0)
                        max_c = item.get('max_count', 0)
                        lines.append(f'    {{')
                        lines.append(f'      "distribution": {dist_str},')
                        lines.append(f'      "num_classes_covered": {num_covered},')
                        lines.append(f'      "min_count": {min_c},')
                        lines.append(f'      "max_count": {max_c}')
                        lines.append(f'    }}{item_comma}')
                    lines.append(f'  ]{comma}')
                else:
                    # 普通字段正常格式化
                    value_str = json.dumps(value, indent=2)
                    # 添加缩进
                    value_lines = value_str.split('\n')
                    if len(value_lines) == 1:
                        lines.append(f'  "{key}": {value_str}{comma}')
                    else:
                        lines.append(f'  "{key}": {value_lines[0]}')
                        for vl in value_lines[1:-1]:
                            lines.append(f'  {vl}')
                        lines.append(f'  {value_lines[-1]}{comma}')
            lines.append('}')
            return '\n'.join(lines)
        
        with open(save_path, 'w') as f:
            f.write(format_history(self.history))
        
        print(f"  History saved to: {save_path}")
    
    def save_checkpoint(self, save_path: str):
        """保存检查点"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'classifier_state_dict': self.classifier.state_dict(),
            'history': self.history
        }
        
        torch.save(checkpoint, save_path)
        print(f"  Checkpoint saved to: {save_path}")


def test_active_learner():
    """测试主动学习器（不需要真实数据）"""
    Config = _get_config()
    
    print("=" * 60)
    print("测试主动学习器")
    print("=" * 60)
    print()
    
    print(f"配置来源: config.py")
    print(f"  device: {Config.device}")
    print(f"  num_classes: {Config.num_classes}")
    print(f"  use_prediction_uncertainty: {Config.use_prediction_uncertainty}")
    print(f"  fusion_strategy: {Config.fusion_strategy}")
    print(f"  random_sampling: {Config.random_sampling}")
    print(f"  classifier_lr: {Config.classifier_lr}")
    print(f"  classifier_epochs: {Config.classifier_epochs}")
    print()
    
    print("ℹ️  这个测试需要真实数据集")
    print("   请在有ImageNet-100数据集的情况下运行完整实验")
    print()
    
    # 创建组件（参数从 Config 读取）
    print("1. 创建组件...")
    feature_extractor = MultiLayerDINOv3(pretrained=False)
    classifier = SimpleClassifier()
    memory_bank = MemoryBank()
    uncertainty_estimator = UncertaintyEstimator(memory_bank)
    selector = CascadingSelectorV1()
    
    # ActiveLearner 参数从 Config 读取
    active_learner = ActiveLearner(
        feature_extractor=feature_extractor,
        classifier=classifier,
        memory_bank=memory_bank,
        uncertainty_estimator=uncertainty_estimator,
        selector=selector
    )
    
    print("✓ ActiveLearner已创建")
    print(f"  - Device: {active_learner.device}")
    print(f"  - Random Sampling: {active_learner.random_sampling}")
    print(f"  - Use Prediction Uncertainty: {active_learner.use_prediction_uncertainty}")
    print()
    
    print("=" * 60)
    print("✓ 测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    test_active_learner()

