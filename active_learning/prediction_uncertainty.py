"""
Prediction Uncertainty (预测不确定性)
- 基于分类头的预测置信度计算不确定性
- 与特征不确定性融合，形成完整的主动学习闭环

配置参数来源: config.py
- Config.num_classes: 类别数
- Config.prediction_uncertainty_mode: 预测不确定性模式
- Config.PREDICTION_UNCERTAINTY_WEIGHTS: 各模式权重
- Config.fusion_strategy: 融合策略
- Config.feature_uncertainty_weight: 特征不确定性权重
- Config.prediction_uncertainty_weight: 预测不确定性权重
- Config.attention_weights: attention模式的动态权重配置
"""

import torch
import torch.nn.functional as F
from typing import Dict


def _get_config():
    """获取 Config 对象"""
    try:
        from .config import Config
    except ImportError:
        from config import Config
    return Config


class PredictionUncertaintyEstimator:
    """
    预测不确定性估计器
    
    基于分类头的输出（logits/probs）计算多种预测不确定性：
    1. Entropy（熵）：概率分布的混乱程度
    2. Margin（间隔）：Top-2类别的概率差
    3. Confidence（置信度）：最大概率的补
    4. Variance（方差）：概率分布的离散度
    """
    
    def __init__(self, num_classes: int = None):
        """
        Args:
            num_classes: 类别数量，如果为 None 则从 Config 读取
        """
        Config = _get_config()
        
        # 从 Config 读取默认值
        if num_classes is None:
            num_classes = getattr(Config, 'num_classes', 100)
        
        self.num_classes = num_classes
        self.max_entropy = torch.log(torch.tensor(num_classes, dtype=torch.float32))
    
    def compute_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """
        计算熵不确定性
        
        原理：概率分布越均匀，熵越大，不确定性越高
        
        Args:
            probs: [B, num_classes] 预测概率分布
        
        Returns:
            entropy: [B] 熵不确定性，归一化到[0, 1]
        
        Example:
            probs = [[0.9, 0.05, 0.05, ...]]  # 确定 → 低熵
            entropy = 0.15
            
            probs = [[0.33, 0.33, 0.34, ...]]  # 混乱 → 高熵
            entropy = 0.95
        """
        # H = -Σ p_i * log(p_i)
        entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=1)  # [B]
        
        # 归一化到[0, 1]
        entropy = entropy / self.max_entropy.to(entropy.device)
        
        return entropy
    
    def compute_margin(self, probs: torch.Tensor) -> torch.Tensor:
        """
        计算间隔不确定性
        
        原理：Top-2类别概率差越小，模型越犹豫，不确定性越高
        
        Args:
            probs: [B, num_classes]
        
        Returns:
            margin_uncertainty: [B] 间隔不确定性，归一化到[0, 1]
        
        Example:
            probs = [[0.9, 0.05, ...]]  # Top1=0.9, Top2=0.05
            margin = 1 - (0.9 - 0.05) = 0.15  # 低不确定性
            
            probs = [[0.51, 0.49, ...]]  # Top1=0.51, Top2=0.49
            margin = 1 - (0.51 - 0.49) = 0.98  # 高不确定性
        """
        # 找Top-2概率
        top2_probs, _ = torch.topk(probs, k=2, dim=1)  # [B, 2]
        
        # 计算间隔
        margin = top2_probs[:, 0] - top2_probs[:, 1]  # [B]
        
        # 转换为不确定性（间隔越小，不确定性越大）
        margin_uncertainty = 1.0 - margin
        
        return margin_uncertainty
    
    def compute_confidence(self, probs: torch.Tensor) -> torch.Tensor:
        """
        计算置信度不确定性
        
        原理：最大概率越低，模型越不确定
        
        Args:
            probs: [B, num_classes]
        
        Returns:
            confidence_uncertainty: [B] 归一化到[0, 1]
        
        Example:
            probs = [[0.95, ...]]  # max_prob = 0.95
            confidence_uncertainty = 1 - 0.95 = 0.05  # 低不确定性
            
            probs = [[0.15, ...]]  # max_prob = 0.15
            confidence_uncertainty = 1 - 0.15 = 0.85  # 高不确定性
        """
        max_probs, _ = probs.max(dim=1)  # [B]
        confidence_uncertainty = 1.0 - max_probs
        return confidence_uncertainty
    
    def compute_variance(self, probs: torch.Tensor) -> torch.Tensor:
        """
        计算方差不确定性
        
        原理：概率分布越集中（低方差），模型越确定
        
        Args:
            probs: [B, num_classes]
        
        Returns:
            variance_uncertainty: [B] 归一化到[0, 1]
        """
        variance = probs.var(dim=1)  # [B]
        
        # 归一化
        max_variance = (1.0 / self.num_classes) * (1.0 - 1.0 / self.num_classes)
        variance_uncertainty = variance / max_variance
        
        return variance_uncertainty
    
    def compute_all(
        self, 
        logits: torch.Tensor,
        normalize: bool = True,
        mode: str = None,
        weights: Dict[str, float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算所有预测不确定性
        
        Args:
            logits: [B, num_classes] 分类头的原始输出
            normalize: 是否归一化
            mode: 预测不确定性模式 ('entropy', 'margin', 'combined')
                  如果为 None，从 Config 读取
            weights: 自定义权重，如果为 None，从 Config 读取
        
        Returns:
            uncertainties: {
                'entropy': [B],
                'margin': [B],
                'confidence': [B],
                'variance': [B],
                'combined': [B]
            }
        """
        # 从 Config 读取模式和权重
        try:
            from .config import Config
        except ImportError:
            from config import Config
        
        if mode is None:
            mode = getattr(Config, 'prediction_uncertainty_mode', 'entropy')
        
        if weights is None:
            weights_dict = getattr(Config, 'PREDICTION_UNCERTAINTY_WEIGHTS', {})
            weights = weights_dict.get(mode, {
                'entropy': 1.0, 'margin': 0.0, 'confidence': 0.0, 'variance': 0.0
            })
        
        # 转换为概率
        probs = F.softmax(logits, dim=1)  # [B, num_classes]
        
        # 计算各种不确定性
        uncertainties = {
            'entropy': self.compute_entropy(probs),
            'margin': self.compute_margin(probs),
            'confidence': self.compute_confidence(probs),
            'variance': self.compute_variance(probs)
        }
        
        # 根据模式计算组合不确定性
        # 使用配置的权重进行加权组合
        uncertainties['combined'] = (
            weights.get('entropy', 0.0) * uncertainties['entropy'] +
            weights.get('margin', 0.0) * uncertainties['margin'] +
            weights.get('confidence', 0.0) * uncertainties['confidence'] +
            weights.get('variance', 0.0) * uncertainties['variance']
        )
        
        if normalize:
            # 归一化到[0, 1]
            for key in uncertainties:
                u = uncertainties[key]
                min_val = u.min()
                max_val = u.max()
                if max_val > min_val:
                    uncertainties[key] = (u - min_val) / (max_val - min_val)
        
        return uncertainties


class HybridUncertaintyFusion:
    """
    混合不确定性融合器（方案4：动态注意力调制）
    
    核心思想：
    - 特征不确定性：告诉我们样本在特征空间的"位置"（探索/边界/密度/多尺度）
    - 预测不确定性：告诉我们分类头对样本的"困惑程度"
    - 融合策略：用预测不确定性动态调整特征不确定性的权重
    """
    
    def __init__(
        self,
        strategy: str = None,  # 'attention', 'multiply', 'add'
        feature_weight: float = None,
        prediction_weight: float = None
    ):
        """
        Args:
            strategy: 融合策略（如果为 None，从 Config 读取）
                - 'attention': 动态注意力调制（推荐）
                - 'multiply': 直接相乘
                - 'add': 加权相加
            feature_weight: 特征不确定性权重（如果为 None，从 Config 读取）
            prediction_weight: 预测不确定性权重（如果为 None，从 Config 读取）
        """
        # 从 Config 读取默认值
        try:
            from .config import Config
        except ImportError:
            from config import Config
        
        self.strategy = strategy if strategy is not None else getattr(Config, 'fusion_strategy', 'attention')
        self.feature_weight = feature_weight if feature_weight is not None else getattr(Config, 'feature_uncertainty_weight', 0.7)
        self.prediction_weight = prediction_weight if prediction_weight is not None else getattr(Config, 'prediction_uncertainty_weight', 0.3)
    
    def _get_round_based_weights(self, current_round: int, total_rounds: int, Config) -> dict:
        """
        根据当前轮次计算动态权重
        
        早期阶段：强调exploration，促进类覆盖
        后期阶段：强调boundary，精细化边界
        过渡阶段：线性插值
        
        Args:
            current_round: 当前轮次（从1开始）
            total_rounds: 总轮次
            Config: 配置对象
        
        Returns:
            attention_weights: 动态调整后的权重字典
        """
        dynamic_config = getattr(Config, 'dynamic_weights_config', {
            'early_phase_ratio': 0.4,
            'early': {
                'exploration': 0.50, 'boundary': 0.20, 
                'multiscale': 0.15, 'density': 0.15
            },
            'late': {
                'exploration': 0.25, 'boundary': 0.35,
                'multiscale': 0.20, 'density': 0.20
            }
        })
        
        early_ratio = dynamic_config.get('early_phase_ratio', 0.4)
        early_weights = dynamic_config.get('early', {})
        late_weights = dynamic_config.get('late', {})
        
        # 计算当前进度
        progress = current_round / total_rounds
        
        # 计算插值因子
        if progress <= early_ratio:
            # 早期阶段：使用early权重
            alpha = 0.0
        elif progress >= 1.0:
            # 后期阶段：使用late权重
            alpha = 1.0
        else:
            # 过渡阶段：线性插值
            alpha = (progress - early_ratio) / (1.0 - early_ratio)
        
        # 从原始attention_weights获取boost值
        original_weights = getattr(Config, 'attention_weights', {
            'exploration': {'base': 0.40, 'boost': 0.10},
            'boundary':    {'base': 0.25, 'boost': 0.10},
            'multiscale':  {'base': 0.20, 'boost': -0.05},
            'density':     {'base': 0.15, 'boost': -0.05}
        })
        
        # 构建动态attention_weights（base线性插值，boost保持不变）
        attention_weights = {}
        for key in ['exploration', 'boundary', 'multiscale', 'density']:
            early_base = early_weights.get(key, original_weights.get(key, {}).get('base', 0.25))
            late_base = late_weights.get(key, original_weights.get(key, {}).get('base', 0.25))
            interpolated_base = early_base * (1 - alpha) + late_base * alpha
            
            attention_weights[key] = {
                'base': interpolated_base,
                'boost': original_weights.get(key, {}).get('boost', 0.0)
            }
        
        # 打印当前轮次的动态权重（仅首次调用时打印）
        if current_round == 1 or current_round == total_rounds // 2 or current_round == total_rounds:
            print(f"    [Dynamic Weights] Round {current_round}/{total_rounds} (progress={progress:.2f}, alpha={alpha:.2f})")
            for key in ['exploration', 'boundary', 'multiscale', 'density']:
                print(f"      {key}: base={attention_weights[key]['base']:.3f}")
        
        return attention_weights
    
    def attention_modulation(
        self,
        feature_uncertainties: Dict[str, torch.Tensor],
        prediction_uncertainty: torch.Tensor,
        current_round: int = None,
        total_rounds: int = None
    ) -> torch.Tensor:
        """
        方案4：动态注意力调制
        
        策略：
        1. 预测不确定性低（分类头确定）→ 信任特征不确定性
        2. 预测不确定性高（分类头困惑）→ 放大特征不确定性
        3. 【可选】根据轮次动态调整：早期强调exploration，后期强调boundary
        
        Args:
            feature_uncertainties: {
                'density': [B],
                'exploration': [B],
                'boundary': [B],
                'multiscale': [B]
            }
            prediction_uncertainty: [B] 预测不确定性（熵）
            current_round: 当前轮次（用于动态权重调整）
            total_rounds: 总轮次（用于动态权重调整）
        
        Returns:
            final_uncertainty: [B] 融合后的不确定性
        """
        # 从 Config 读取动态权重配置
        Config = _get_config()
        
        # 检查是否启用动态权重调整（根据轮次）
        use_dynamic_weights = getattr(Config, 'use_dynamic_weights', False)
        
        if use_dynamic_weights and current_round is not None and total_rounds is not None:
            # 动态权重模式：根据轮次调整base权重
            attention_weights = self._get_round_based_weights(current_round, total_rounds, Config)
        else:
            # 静态权重模式：使用配置的attention_weights
            attention_weights = getattr(Config, 'attention_weights', {
                'exploration': {'base': 0.25, 'boost': 0.15},
                'boundary':    {'base': 0.25, 'boost': 0.15},
                'multiscale':  {'base': 0.25, 'boost': -0.10},
                'density':     {'base': 0.25, 'boost': -0.20}
            })
        
        modulation_factor_range = getattr(Config, 'modulation_factor_range', [1.0, 1.5])
        
        # 获取参与融合的不确定性列表（用于消融实验）
        active_uncertainties = getattr(Config, 'active_feature_uncertainties', 
                                       ['exploration', 'boundary', 'density', 'multiscale'])
        
        # 1. 提取4种特征不确定性
        u_density = feature_uncertainties['density']
        u_exploration = feature_uncertainties['exploration']
        u_boundary = feature_uncertainties['boundary']
        u_multiscale = feature_uncertainties['multiscale']
        
        # 2. 基于预测不确定性生成动态权重
        # 预测不确定性高（分类器困惑）→ 更依赖探索和边界信号
        # 预测不确定性低（分类器确定）→ 均等权重，不特别偏重某一方面
        
        # 归一化预测不确定性
        u_pred_norm = prediction_uncertainty  # 已归一化到[0, 1]
        
        # 动态权重生成（根据 active_uncertainties 决定哪些参与）
        # 不参与的不确定性权重设为 0
        if 'exploration' in active_uncertainties:
            w_exploration = attention_weights['exploration']['base'] + attention_weights['exploration']['boost'] * u_pred_norm
        else:
            w_exploration = u_pred_norm * 0  # 保持 tensor 形状，值为 0
        
        if 'boundary' in active_uncertainties:
            w_boundary = attention_weights['boundary']['base'] + attention_weights['boundary']['boost'] * u_pred_norm
        else:
            w_boundary = u_pred_norm * 0
        
        if 'multiscale' in active_uncertainties:
            w_multiscale = attention_weights['multiscale']['base'] + attention_weights['multiscale']['boost'] * u_pred_norm
        else:
            w_multiscale = u_pred_norm * 0
        
        if 'density' in active_uncertainties:
            w_density = attention_weights['density']['base'] + attention_weights['density']['boost'] * u_pred_norm
        else:
            w_density = u_pred_norm * 0
        
        # 归一化权重（只对参与的不确定性归一化）
        total_weight = w_exploration + w_boundary + w_multiscale + w_density
        # 避免除以0（当所有权重都为0时，使用均等权重作为fallback）
        total_weight = torch.where(total_weight > 0, total_weight, 
                                   torch.ones_like(total_weight))
        w_exploration = w_exploration / total_weight
        w_boundary = w_boundary / total_weight
        w_multiscale = w_multiscale / total_weight
        w_density = w_density / total_weight
        
        # 3. 加权组合特征不确定性
        feature_combined = (
            w_exploration * u_exploration +
            w_boundary * u_boundary +
            w_multiscale * u_multiscale +
            w_density * u_density
        )
        
        # 4. 用预测不确定性调制
        # 预测不确定性高 → 放大特征不确定性
        mod_min, mod_max = modulation_factor_range
        modulation_factor = mod_min + (mod_max - mod_min) * u_pred_norm
        final_uncertainty = feature_combined * modulation_factor
        
        return final_uncertainty
    
    def multiply_fusion(
        self,
        feature_uncertainty: torch.Tensor,
        prediction_uncertainty: torch.Tensor
    ) -> torch.Tensor:
        """
        简单相乘融合
        
        原理：两种不确定性都高才真正高
        """
        return feature_uncertainty * prediction_uncertainty
    
    def add_fusion(
        self,
        feature_uncertainty: torch.Tensor,
        prediction_uncertainty: torch.Tensor
    ) -> torch.Tensor:
        """
        加权相加融合
        
        原理：线性组合两种不确定性
        """
        return (
            self.feature_weight * feature_uncertainty +
            self.prediction_weight * prediction_uncertainty
        )
    
    def fuse(
        self,
        feature_uncertainties: Dict[str, torch.Tensor],
        prediction_uncertainties: Dict[str, torch.Tensor],
        current_round: int = None,
        total_rounds: int = None
    ) -> torch.Tensor:
        """
        融合特征不确定性和预测不确定性
        
        Args:
            feature_uncertainties: 4种特征不确定性
            prediction_uncertainties: 预测不确定性（使用entropy）
            current_round: 当前轮次（用于动态权重调整，可选）
            total_rounds: 总轮次（用于动态权重调整，可选）
        
        Returns:
            final_uncertainty: [B] 最终不确定性
        """
        # 使用熵作为主要的预测不确定性
        pred_uncertainty = prediction_uncertainties['entropy']
        
        if self.strategy == 'attention':
            # 动态注意力调制（推荐）
            return self.attention_modulation(
                feature_uncertainties, 
                pred_uncertainty,
                current_round=current_round,
                total_rounds=total_rounds
            )
        elif self.strategy == 'multiply':
            # 相乘
            feat_combined = feature_uncertainties.get(
                'combined',
                sum(feature_uncertainties.values()) / len(feature_uncertainties)
            )
            return self.multiply_fusion(feat_combined, pred_uncertainty)
        elif self.strategy == 'add':
            # 相加
            feat_combined = feature_uncertainties.get(
                'combined',
                sum(feature_uncertainties.values()) / len(feature_uncertainties)
            )
            return self.add_fusion(feat_combined, pred_uncertainty)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")


def test_prediction_uncertainty():
    """测试预测不确定性计算"""
    print("=" * 60)
    print("测试预测不确定性估计器")
    print("=" * 60)
    print()
    
    # 创建估计器
    estimator = PredictionUncertaintyEstimator(num_classes=100)
    
    # 测试场景1：模型非常确定
    print("场景1：模型非常确定")
    logits_confident = torch.tensor([
        [10.0, 1.0, 0.5, 0.3] + [0.0] * 96  # 第1类概率远高于其他
    ])
    pred_uncertainties = estimator.compute_all(logits_confident)
    print(f"  Logits: {logits_confident[0, :4].tolist()}")
    print(f"  Probs: {F.softmax(logits_confident, dim=1)[0, :4].tolist()}")
    print(f"  Entropy: {pred_uncertainties['entropy'].item():.4f} (应该很低)")
    print(f"  Margin: {pred_uncertainties['margin'].item():.4f} (应该很低)")
    print(f"  Confidence: {pred_uncertainties['confidence'].item():.4f} (应该很低)")
    print()
    
    # 测试场景2：模型很困惑
    print("场景2：模型很困惑（Top-2接近）")
    logits_confused = torch.tensor([
        [2.0, 1.9, 0.5, 0.3] + [0.0] * 96  # Top-2非常接近
    ])
    pred_uncertainties = estimator.compute_all(logits_confused)
    print(f"  Logits: {logits_confused[0, :4].tolist()}")
    print(f"  Probs: {F.softmax(logits_confused, dim=1)[0, :4].tolist()}")
    print(f"  Entropy: {pred_uncertainties['entropy'].item():.4f} (应该较高)")
    print(f"  Margin: {pred_uncertainties['margin'].item():.4f} (应该很高)")
    print(f"  Confidence: {pred_uncertainties['confidence'].item():.4f} (应该较高)")
    print()
    
    # 测试场景3：完全随机
    print("场景3：完全随机（均匀分布）")
    logits_random = torch.zeros(1, 100)  # 所有类别概率相同
    pred_uncertainties = estimator.compute_all(logits_random)
    print(f"  Logits: 全部为0（均匀分布）")
    print(f"  Entropy: {pred_uncertainties['entropy'].item():.4f} (应该最高)")
    print(f"  Confidence: {pred_uncertainties['confidence'].item():.4f} (应该最高)")
    print()
    
    print("=" * 60)
    print("✓ 测试完成!")
    print("=" * 60)


def test_hybrid_fusion():
    """测试混合融合"""
    print("=" * 60)
    print("测试混合不确定性融合")
    print("=" * 60)
    print()
    
    # 创建融合器
    fusion = HybridUncertaintyFusion(strategy='attention')
    
    # 模拟特征不确定性
    B = 5
    feature_uncertainties = {
        'density': torch.rand(B),
        'exploration': torch.rand(B),
        'boundary': torch.rand(B),
        'multiscale': torch.rand(B)
    }
    
    # 模拟预测不确定性
    prediction_estimator = PredictionUncertaintyEstimator()
    logits = torch.randn(B, 100)
    prediction_uncertainties = prediction_estimator.compute_all(logits)
    
    # 融合
    final_uncertainty = fusion.fuse(feature_uncertainties, prediction_uncertainties)
    
    print("特征不确定性:")
    for key, value in feature_uncertainties.items():
        print(f"  {key}: {value.tolist()}")
    print()
    
    print("预测不确定性:")
    print(f"  entropy: {prediction_uncertainties['entropy'].tolist()}")
    print()
    
    print("融合后的不确定性:")
    print(f"  final: {final_uncertainty.tolist()}")
    print()
    
    print("=" * 60)
    print("✓ 测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    test_prediction_uncertainty()
    print()
    test_hybrid_fusion()

