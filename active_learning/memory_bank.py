"""
Memory Bank for Active Learning
- 存储已标注样本的特征和标签
- 支持KNN查询（用于计算不确定性）
- 支持增量更新
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


def _get_config():
    """获取 Config 对象"""
    try:
        from .config import Config
    except ImportError:
        from config import Config
    return Config


class MemoryBank:
    """
    Memory Bank: 存储已标注样本的特征和标签
    
    用于计算以下不确定性:
    1. Exploration: 距离最近已标注样本的距离
    2. Boundary: KNN标签一致性
    3. Density: 局部密度
    4. Multi-Scale: 多层特征一致性
    """
    
    def __init__(
        self,
        feature_dim: int = None,
        num_classes: int = None,
        device: str = None,
        layers: List[int] = None
    ):
        """
        Args:
            feature_dim: 特征维度（ViT-B: 768, ViT-L: 1024）
            num_classes: 类别数
            device: 存储设备
            layers: 要存储的层索引
        """
        Config = _get_config()
        
        # 从 Config 读取默认值
        if feature_dim is None:
            feature_dim = getattr(Config, 'feature_dim', 768)
        if num_classes is None:
            num_classes = getattr(Config, 'num_classes', 100)
        if device is None:
            device = getattr(Config, 'device', 'cuda')
        if layers is None:
            layers = getattr(Config, 'feature_layers', [3, 6, 9, 11])
        
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.device = device
        self.layers = layers
        
        # 存储特征和标签
        # - cls_features: [N, feature_dim] 用于分类和Exploration
        # - multi_layer_features: {layer_idx: [N, num_patches, feature_dim]} 用于Multi-Scale
        # - labels: [N] 标签
        self.cls_features = None
        self.multi_layer_features = {f'layer_{i}': None for i in layers}
        self.labels = None
        
        # 样本数量
        self.num_samples = 0
        
        print(f"  Memory Bank initialized")
        print(f"  - Feature dimension: {feature_dim}")
        print(f"  - Number of classes: {num_classes}")
        print(f"  - Device: {device}")
        print(f"  - Stored layers: {layers}")
    
    def add_samples(
        self,
        cls_features: torch.Tensor,
        multi_layer_features: Dict[str, torch.Tensor],
        labels: torch.Tensor
    ):
        """
        添加新的已标注样本
        
        Args:
            cls_features: [B, feature_dim] CLS token特征
            multi_layer_features: {
                'layer_3': [B, num_patches, feature_dim],
                'layer_6': [B, num_patches, feature_dim],
                ...
            }
            labels: [B] 标签
        """
        # 移动到指定设备
        cls_features = cls_features.to(self.device)
        labels = labels.to(self.device)
        
        # L2归一化（用于余弦相似度计算）
        cls_features = F.normalize(cls_features, p=2, dim=1)
        
        # 初始化或追加CLS特征
        if self.cls_features is None:
            # 第一次添加（Memory Bank 为空）
            self.cls_features = cls_features
            self.labels = labels
        else:
            # 后续追加（torch.cat: 沿着指定维度拼接张量）
            self.cls_features = torch.cat([self.cls_features, cls_features], dim=0)
            self.labels = torch.cat([self.labels, labels], dim=0)
        
        # 处理多层特征
        # layer_key: 层索引(layer_3), layer_feat: 层特征[B, num_patches, feature_dim]
        for layer_key, layer_feat in multi_layer_features.items():
            if layer_key == 'cls':
                continue  # 跳过CLS token
            
            # 移动到设备
            layer_feat = layer_feat.to(self.device)
            
            # 自适应处理：检查特征是否已经被池化
            if layer_feat.dim() == 3:
                # [B, num_patches, D] → [B, D]
                layer_feat_pooled = layer_feat.mean(dim=1)
            elif layer_feat.dim() == 2:
                # [B, D] 已经被池化
                layer_feat_pooled = layer_feat
            else:
                raise ValueError(f"Unexpected feature shape for {layer_key}: {layer_feat.shape}")
            
            # L2归一化
            layer_feat_pooled = F.normalize(layer_feat_pooled, p=2, dim=1)
            
            # 初始化或追加
            if self.multi_layer_features[layer_key] is None:
                self.multi_layer_features[layer_key] = layer_feat_pooled
            else:
                self.multi_layer_features[layer_key] = torch.cat(
                    [self.multi_layer_features[layer_key], layer_feat_pooled],
                    dim=0
                )
        
        self.num_samples = len(self.labels)
        print(f"  Added {len(labels)} samples to Memory Bank (Total: {self.num_samples})")
    
    def compute_knn(
        self,
        query_features: torch.Tensor,
        k: int = 10,
        use_cls: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算KNN（K近邻）
        
        Args:
            query_features: [B, feature_dim] 查询特征
            k: 近邻数量
            use_cls: 是否使用CLS特征（False则使用最后一层的pooled patch特征）
        
        Returns:
            distances: [B, k] K个最近邻的距离
            indices: [B, k] K个最近邻的索引
        """
        if self.num_samples == 0:
            raise ValueError("Memory Bank is empty, cannot compute KNN")
        
        # 归一化查询特征
        query_features = F.normalize(query_features, p=2, dim=1).to(self.device)
        
        # 选择存储的特征
        if use_cls:
            stored_features = self.cls_features
        else:
            # 使用最后一层的pooled特征
            last_layer_key = f'layer_{self.layers[-1]}'
            stored_features = self.multi_layer_features[last_layer_key]
        
        # 计算余弦相似度 → 距离
        # similarity: [B, N]
        similarity = torch.mm(query_features, stored_features.t())
        distances = 1 - similarity  # 余弦距离
        
        # 获取top-k最近邻
        k = min(k, self.num_samples)
        topk_distances, topk_indices = torch.topk(distances, k, largest=False, dim=1)
        
        return topk_distances, topk_indices
    
    def get_knn_labels(self, indices: torch.Tensor) -> torch.Tensor:
        """
        获取KNN的标签
        
        Args:
            indices: [B, k] KNN索引
        
        Returns:
            knn_labels: [B, k] KNN标签
        """
        return self.labels[indices]
    
    def compute_min_distance(self, query_features: torch.Tensor) -> torch.Tensor:
        """
        计算到最近已标注样本的距离（用于Exploration不确定性）
        
        Args:
            query_features: [B, feature_dim] 查询特征
        
        Returns:
            min_distances: [B] 最近距离
        """
        if self.num_samples == 0:
            # 如果Memory Bank为空，返回最大距离
            return torch.ones(len(query_features), device=self.device)
        
        distances, _ = self.compute_knn(query_features, k=1, use_cls=True)
        return distances.squeeze(1)  # 压缩维度：[B, 1] → [B]
    
    def compute_knn_similarity_std(
        self,
        query_features: torch.Tensor,
        k: int = 10
    ) -> torch.Tensor:
        """
        计算KNN相似度的标准差（用于Density不确定性）
        
        原理: 
        - 标准差大 → 邻居有远有近 → 样本孤立/稀疏区域
        - 标准差小 → 邻居距离一致 → 样本在密集区域
        
        Args:
            query_features: [B, feature_dim] 查询特征
            k: 近邻数量
        
        Returns:
            similarity_std: [B] KNN相似度的标准差
        """
        if self.num_samples == 0:
            # 如果Memory Bank为空，返回最大标准差
            return torch.ones(len(query_features), device=self.device)
        
        # 归一化查询特征
        query_features = F.normalize(query_features, p=2, dim=1).to(self.device)
        
        # 计算余弦相似度
        stored_features = self.cls_features
        similarity = torch.mm(query_features, stored_features.t())  # [B, N]
        
        # 获取top-k最近邻的相似度
        k = min(k, self.num_samples)
        topk_similarities, _ = torch.topk(similarity, k, largest=True, dim=1)  # [B, k]
        
        # 计算标准差
        similarity_std = topk_similarities.std(dim=1)  # [B]
        
        return similarity_std
    
    def get_multi_layer_features(self,
        layer_key: str
    ) -> Optional[torch.Tensor]:
        """
        获取特定层的特征
        
        Args:
            layer_key: 层名称，如 'layer_3'
        
        Returns:
            features: [N, feature_dim] 或 None
        """
        return self.multi_layer_features.get(layer_key, None)
    
    def compute_multi_layer_distances(
        self,
        query_multi_layer_features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        计算多层特征的最近邻距离（用于Multi-Scale不确定性）
        
        Args:
            query_multi_layer_features: {
                'layer_3': [B, num_patches, feature_dim],
                'layer_6': [B, num_patches, feature_dim],
                ...
            }
        
        Returns:
            layer_distances: {
                'layer_3': [B] 最近邻距离,
                'layer_6': [B],
                ...
            }
        """
        if self.num_samples == 0:
            # 如果Memory Bank为空，返回最大距离
            layer_distances = {}
            for layer_key in query_multi_layer_features.keys():
                if layer_key == 'cls':
                    continue
                batch_size = len(query_multi_layer_features[layer_key])
                layer_distances[layer_key] = torch.ones(batch_size, device=self.device)
            return layer_distances
        
        layer_distances = {}
        
        # layer_key: 层索引(layer_3), query_feat: 查询特征[B, num_patches, feature_dim]
        for layer_key, query_feat in query_multi_layer_features.items():
            if layer_key == 'cls':
                continue
            
            # 自适应处理：检查特征是否已经被池化
            if query_feat.dim() == 3:
                # [B, num_patches, D] → [B, D]
                query_feat_pooled = query_feat.mean(dim=1)
            elif query_feat.dim() == 2:
                # [B, D] 已经被池化
                query_feat_pooled = query_feat
            else:
                raise ValueError(f"Unexpected feature shape: {query_feat.shape}")
            
            # 归一化
            query_feat_pooled = F.normalize(query_feat_pooled, p=2, dim=1).to(self.device)
            
            # 获取存储的特征
            stored_feat = self.multi_layer_features[layer_key]
            
            # 计算余弦相似度 → 距离（.t()转置存储特征）
            similarity = torch.mm(query_feat_pooled, stored_feat.t())
            distances = 1 - similarity  # [B, N]
            
            # 获取最近邻距离（torch.min: 沿着指定维度找到最小值）
            min_distances, _ = torch.min(distances, dim=1)
            layer_distances[layer_key] = min_distances
        
        return layer_distances
    
    def clear(self):
        """清空Memory Bank"""
        self.cls_features = None
        self.multi_layer_features = {f'layer_{i}': None for i in self.layers}
        self.labels = None
        self.num_samples = 0
        print("  Memory Bank cleared")
    
    def __len__(self):
        return self.num_samples
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        if self.num_samples == 0:
            return {
                'num_samples': 0,
                'num_classes': 0,
                'samples_per_class': {}
            }
        
        # 统计每个类别的样本数
        unique_labels, counts = torch.unique(self.labels, return_counts=True)
        samples_per_class = {
            int(label): int(count)
            for label, count in zip(unique_labels.cpu(), counts.cpu())
        }
        
        return {
            'num_samples': self.num_samples,
            'num_classes': len(unique_labels),
            'samples_per_class': samples_per_class,
            'memory_usage_mb': {
                'cls_features': self.cls_features.element_size() * self.cls_features.nelement() / 1024 / 1024,
                'total': sum([
                    feat.element_size() * feat.nelement() / 1024 / 1024
                    for feat in self.multi_layer_features.values()
                    if feat is not None
                ]) + self.cls_features.element_size() * self.cls_features.nelement() / 1024 / 1024
            }
        }


def test_memory_bank():
    """测试Memory Bank"""
    Config = _get_config()
    
    print("=" * 60)
    print("测试Memory Bank")
    print("=" * 60)
    print()
    
    print(f"配置来源: config.py")
    print(f"  feature_dim: {Config.feature_dim}")
    print(f"  num_classes: {Config.num_classes}")
    print(f"  device: {Config.device}")
    print(f"  feature_layers: {Config.feature_layers}")
    print()
    
    # 创建Memory Bank（参数从 Config 读取）
    memory_bank = MemoryBank()
    print()
    
    # 模拟一些特征数据
    batch_size = 8
    num_patches = 195
    feature_dim = Config.feature_dim
    
    # 生成随机特征
    cls_features = torch.randn(batch_size, feature_dim)
    multi_layer_features = {
        'layer_3': torch.randn(batch_size, num_patches, feature_dim),
        'layer_6': torch.randn(batch_size, num_patches, feature_dim),
        'layer_9': torch.randn(batch_size, num_patches, feature_dim),
        'layer_11': torch.randn(batch_size, num_patches, feature_dim),
        'cls': cls_features
    }
    labels = torch.randint(0, 100, (batch_size,))
    
    print("1. 添加样本到Memory Bank:")
    memory_bank.add_samples(cls_features, multi_layer_features, labels)
    print()
    
    # 再添加一批
    cls_features2 = torch.randn(batch_size, feature_dim)
    multi_layer_features2 = {
        'layer_3': torch.randn(batch_size, num_patches, feature_dim),
        'layer_6': torch.randn(batch_size, num_patches, feature_dim),
        'layer_9': torch.randn(batch_size, num_patches, feature_dim),
        'layer_11': torch.randn(batch_size, num_patches, feature_dim),
        'cls': cls_features2
    }
    labels2 = torch.randint(0, 100, (batch_size,))
    memory_bank.add_samples(cls_features2, multi_layer_features2, labels2)
    print()
    
    # 测试KNN查询
    print("2. 测试KNN查询:")
    query_features = torch.randn(4, feature_dim)
    distances, indices = memory_bank.compute_knn(query_features, k=5)
    print(f"  查询特征: {query_features.shape}")
    print(f"  KNN距离: {distances.shape}")
    print(f"  KNN索引: {indices.shape}")
    print(f"  KNN距离范围: [{distances.min():.4f}, {distances.max():.4f}]")
    
    # 获取KNN标签
    knn_labels = memory_bank.get_knn_labels(indices)
    print(f"  KNN标签: {knn_labels.shape}")
    print()
    
    # 测试最近邻距离
    print("3. 测试最近邻距离 (Exploration):")
    min_distances = memory_bank.compute_min_distance(query_features)
    print(f"  最近邻距离: {min_distances.shape}")
    print(f"  距离值: {min_distances}")
    print()
    
    # 测试KNN相似度标准差（用于Density不确定性）
    print("4. 测试KNN相似度标准差 (Density):")
    similarity_std = memory_bank.compute_knn_similarity_std(query_features, k=5)
    print(f"  相似度标准差: {similarity_std.shape}")
    print(f"  标准差值: {similarity_std}")
    print()
    
    # 测试多层距离计算
    print("5. 测试多层距离计算 (Multi-Scale):")
    query_multi_layer = {
        'layer_3': torch.randn(4, num_patches, feature_dim),
        'layer_6': torch.randn(4, num_patches, feature_dim),
        'layer_9': torch.randn(4, num_patches, feature_dim),
        'layer_11': torch.randn(4, num_patches, feature_dim),
    }
    layer_distances = memory_bank.compute_multi_layer_distances(query_multi_layer)
    for layer_key, dists in layer_distances.items():
        print(f"  {layer_key}: {dists.shape}, 范围 [{dists.min():.4f}, {dists.max():.4f}]")
    print()
    
    # 获取统计信息
    print("6. Memory Bank统计:")
    stats = memory_bank.get_statistics()
    print(f"  样本数: {stats['num_samples']}")
    print(f"  类别数: {stats['num_classes']}")
    print(f"  显存占用: {stats['memory_usage_mb']['total']:.2f} MB")
    print()
    
    print("=" * 60)
    print("✓ 测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    test_memory_bank()

