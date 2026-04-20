"""
Uncertainty Estimation for Active Learning
- Exploration Uncertainty: 距离已标注样本的距离
- Boundary Uncertainty: KNN标签一致性
- Density Uncertainty: 局部密度
- Multi-Scale Feature Uncertainty: 多层特征一致性
"""

import torch
from typing import Dict, Optional

try:
    from .memory_bank import MemoryBank
except ImportError:
    # 如果作为脚本运行，使用绝对导入
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from active_learning.memory_bank import MemoryBank


def _get_config():
    """获取 Config 对象"""
    try:
        from .config import Config
    except ImportError:
        from config import Config
    return Config


class UncertaintyEstimator:
    """
    不确定性估计器
    
    实现4种基于特征的不确定性:
    1. Exploration: 远离已标注样本 → 需要探索
    2. Boundary: KNN标签不一致 → 决策边界
    3. Density: 低密度区域 → 稀疏区域
    4. Multi-Scale: 多层特征不一致 → 语义复杂度
    """
    
    def __init__(
        self,
        memory_bank: MemoryBank,
        k_neighbors: int = None
    ):
        """
        Args:
            memory_bank: Memory Bank实例
            k_neighbors: KNN的K值
        """
        Config = _get_config()
        
        # 从 Config 读取默认值
        if k_neighbors is None:
            k_neighbors = getattr(Config, 'k_neighbors', 10)
        
        self.memory_bank = memory_bank
        self.k_neighbors = k_neighbors
    
    def compute_exploration_uncertainty(
        self,
        cls_features: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        1. Exploration Uncertainty: 距离最近已标注样本的距离
        
        原理: 远离已标注样本的样本更有价值（探索未知区域），
        虽然是计算min，但是值越大，不确定性越大，所以需要探索。
        
        Args:
            cls_features: [B, feature_dim] CLS token特征
            normalize: 是否归一化到[0, 1]
        
        Returns:
            exploration_uncertainty: [B] 探索不确定性
        """
        # 计算到最近已标注样本的距离
        min_distances = self.memory_bank.compute_min_distance(cls_features)
        
        if normalize:
            # 归一化到[0, 1]
            min_dist = min_distances.min()
            max_dist = min_distances.max()
            if max_dist > min_dist:
                min_distances = (min_distances - min_dist) / (max_dist - min_dist)
        
        return min_distances
    
    def compute_boundary_uncertainty(
        self,
        cls_features: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        2. Boundary Uncertainty: KNN标签一致性
        
        原理: KNN标签不一致的样本位于决策边界，更有价值
        
        计算方式:
        - 获取K个最近邻的标签
        - 计算标签的熵或方差
        - 熵越大，不确定性越高
        
        Args:
            cls_features: [B, feature_dim] CLS token特征
            normalize: 是否归一化
        
        Returns:
            boundary_uncertainty: [B] 边界不确定性
        """
        if self.memory_bank.num_samples == 0:
            return torch.zeros(len(cls_features), device=cls_features.device)
        
        # 获取KNN
        k = min(self.k_neighbors, self.memory_bank.num_samples)
        _, knn_indices = self.memory_bank.compute_knn(cls_features, k=k)
        
        # 获取KNN标签
        knn_labels = self.memory_bank.get_knn_labels(knn_indices)  # [B, k]
        
        # 计算标签的熵
        # 方法1: 使用标签多样性（1 - 最多标签的比例）
        batch_size = len(knn_labels)
        uncertainties = []
        
        for i in range(batch_size):
            labels = knn_labels[i]  # 第i个样本的K个邻居标签, 形状 [k]
            
            # torch.unique: 统计唯一值和出现次数
            # 例: labels=[3,3,5,3,7,5,3,3,5,7] → unique_labels=[3,5,7], counts=[5,3,2]
            unique_labels, counts = torch.unique(labels, return_counts=True)
            
            # 计算概率分布: 每个标签的出现比例
            # 例: counts=[5,3,2], k=10 → probs=[0.5, 0.3, 0.2]
            # .float() 将整数张量转为浮点数，否则整数除法会丢失小数
            probs = counts.float() / k
            
            # 计算信息熵: H = -Σ p*log(p)
            # - torch.log(): 计算自然对数 (底数e)
            # - + 1e-10: 防止 log(0) 产生 -inf
            # - .sum(): 对所有概率项求和
            # 例: probs=[0.5,0.3,0.2] → 熵 ≈ 1.03
            # 熵越大 → 标签越分散 → 越接近决策边界
            entropy = -(probs * torch.log(probs + 1e-10)).sum()
            
            # 归一化熵到 [0, 1] 范围
            # 最大熵发生在均匀分布时: 每个标签出现次数相同
            # 均匀分布熵 = -k*(1/k)*log(1/k) = log(k)
            # torch.tensor(): 创建标量张量, dtype指定数据类型
            max_entropy = torch.log(torch.tensor(k, dtype=torch.float))
            normalized_entropy = entropy / max_entropy
            
            uncertainties.append(normalized_entropy)
        
        # torch.stack(): 将张量列表堆叠成一个张量
        # 例: [tensor(0.3), tensor(0.8), ...] → tensor([0.3, 0.8, ...])
        boundary_uncertainty = torch.stack(uncertainties).to(cls_features.device)
        
        # 归一化到 [0, 1]（与其他三种不确定性保持一致）
        # 虽然熵已经除以 max_entropy，但 batch 内的实际分布可能不是 [0, 1]
        # 使用 min-max 归一化确保与其他不确定性可公平比较和融合
        if normalize:
            min_val = boundary_uncertainty.min()
            max_val = boundary_uncertainty.max()
            if max_val > min_val:
                boundary_uncertainty = (boundary_uncertainty - min_val) / (max_val - min_val)
        
        return boundary_uncertainty
    
    def compute_density_uncertainty(
        self,
        cls_features: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        3. Density Uncertainty: KNN相似度的分散程度
        
        原理: KNN相似度标准差大 → 邻居有远有近 → 样本孤立/稀疏区域
        
        公式: u_density = Std({sim(f(x), f(x_i)) : x_i ∈ kNN(x)})
        
        物理意义:
        - 标准差大 → 有的邻居很近，有的很远 → 孤立样本/噪声
        - 标准差小 → 邻居距离一致 → 密集区域
        
        Args:
            cls_features: [B, feature_dim] CLS token特征
            normalize: 是否归一化
        
        Returns:
            density_uncertainty: [B] 密度不确定性
        """
        if self.memory_bank.num_samples == 0:
            # 如果Memory Bank为空，返回零不确定性
            return torch.zeros(len(cls_features), device=cls_features.device)
        
        # 计算KNN相似度的标准差
        k = min(self.k_neighbors, self.memory_bank.num_samples)
        density_uncertainty = self.memory_bank.compute_knn_similarity_std(
            cls_features,
            k=k
        )
        
        if normalize:
            # 归一化到[0, 1]
            min_val = density_uncertainty.min()
            max_val = density_uncertainty.max()
            if max_val > min_val:
                density_uncertainty = (density_uncertainty - min_val) / (max_val - min_val)
        
        return density_uncertainty
    
    def compute_multiscale_uncertainty(
        self,
        multi_layer_features: Dict[str, torch.Tensor],
        normalize: bool = True
    ) -> torch.Tensor:
        """
        4. Multi-Scale Feature Uncertainty: 多层特征一致性
        
        原理: 不同层特征的不确定性差异大 → 语义复杂，需要标注
        
        公式: u_multiscale = ū · (1 - Var(u))
        其中:
        - u = [u_layer3, u_layer6, u_layer9, u_layer11] 每层的exploration uncertainty
        - ū = mean(u) 平均不确定性
        - Var(u) = variance(u) 方差
        
        Args:
            multi_layer_features: {
                'layer_3': [B, num_patches, feature_dim],
                'layer_6': [B, num_patches, feature_dim],
                ...
            }
            normalize: 是否归一化
        
        Returns:
            multiscale_uncertainty: [B] 多尺度不确定性
        """
        if self.memory_bank.num_samples == 0:
            batch_size = len(multi_layer_features['layer_3'])
            return torch.ones(batch_size, device=multi_layer_features['layer_3'].device)
        
        # 计算每层的exploration uncertainty（距离）
        layer_distances = self.memory_bank.compute_multi_layer_distances(multi_layer_features)
        
        # 将距离堆叠成 [B, num_layers]
        layer_keys = sorted([k for k in layer_distances.keys()])
        uncertainties_per_layer = torch.stack([layer_distances[k] for k in layer_keys], dim=1)
        
        # 计算平均不确定性 ū
        mean_uncertainty = uncertainties_per_layer.mean(dim=1)  # [B]
        
        # 计算方差 Var(u)
        variance = uncertainties_per_layer.var(dim=1)  # [B]
        
        # 公式: ū · (1 - Var(u))
        # - 平均不确定性高 → 整体距离远
        # - 方差小（一致性高）→ (1 - Var) 大
        # - 两者相乘 → 高不确定性 + 高一致性 = 需要标注
        multiscale_uncertainty = mean_uncertainty * (1 - variance)
        
        if normalize:
            # 归一化到[0, 1]
            min_val = multiscale_uncertainty.min()
            max_val = multiscale_uncertainty.max()
            if max_val > min_val:
                multiscale_uncertainty = (multiscale_uncertainty - min_val) / (max_val - min_val)
        
        return multiscale_uncertainty
    
    def compute_all_uncertainties(
        self,
        cls_features: torch.Tensor,
        multi_layer_features: Dict[str, torch.Tensor],
        weights: Optional[Dict[str, float]] = None,
        normalize: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        计算所有4种不确定性
        
        Args:
            cls_features: [B, feature_dim] CLS token特征
            multi_layer_features: {
                'layer_3': [B, num_patches, feature_dim],
                ...
            }
            weights: 各不确定性的权重，如 {'exploration': 0.25, ...}
            normalize: 是否归一化
        
        Returns:
            uncertainties: {
                'exploration': [B],
                'boundary': [B],
                'density': [B],
                'multiscale': [B],
                'combined': [B]  # 加权组合
            }
        """
        # 从 Config 读取默认权重
        if weights is None:
            Config = _get_config()
            weights = getattr(Config, 'feature_uncertainty_weights', {
                'exploration': 0.25,
                'boundary': 0.25,
                'density': 0.25,
                'multiscale': 0.25
            })
        
        # 计算各不确定性
        u_exploration = self.compute_exploration_uncertainty(cls_features, normalize=normalize)
        u_boundary = self.compute_boundary_uncertainty(cls_features, normalize=normalize)
        u_density = self.compute_density_uncertainty(cls_features, normalize=normalize)
        u_multiscale = self.compute_multiscale_uncertainty(multi_layer_features, normalize=normalize)
        
        # 加权组合
        u_combined = (
            weights['exploration'] * u_exploration +
            weights['boundary'] * u_boundary +
            weights['density'] * u_density +
            weights['multiscale'] * u_multiscale
        )
        
        return {
            'exploration': u_exploration,
            'boundary': u_boundary,
            'density': u_density,
            'multiscale': u_multiscale,
            'combined': u_combined
        }


def test_uncertainty_estimator():
    """测试不确定性估计器"""
    Config = _get_config()
    
    print("=" * 60)
    print("测试不确定性估计器")
    print("=" * 60)
    print()
    
    print(f"配置来源: config.py")
    print(f"  k_neighbors: {Config.k_neighbors}")
    print(f"  feature_uncertainty_weights: {Config.feature_uncertainty_weights}")
    print()
    
    # 创建Memory Bank（参数从 Config 读取）
    memory_bank = MemoryBank()
    print()
    
    # 添加一些已标注样本
    batch_size = 20
    num_patches = 195
    feature_dim = Config.feature_dim
    
    cls_features_labeled = torch.randn(batch_size, feature_dim)
    multi_layer_labeled = {
        'layer_3': torch.randn(batch_size, num_patches, feature_dim),
        'layer_6': torch.randn(batch_size, num_patches, feature_dim),
        'layer_9': torch.randn(batch_size, num_patches, feature_dim),
        'layer_11': torch.randn(batch_size, num_patches, feature_dim),
    }
    labels = torch.randint(0, Config.num_classes, (batch_size,))
    
    print("1. 添加已标注样本:")
    memory_bank.add_samples(cls_features_labeled, multi_layer_labeled, labels)
    print()
    
    # 创建不确定性估计器（参数从 Config 读取）
    print("2. 创建不确定性估计器:")
    estimator = UncertaintyEstimator(memory_bank=memory_bank)
    print(f"✓ 不确定性估计器已创建 (k_neighbors={estimator.k_neighbors})")
    print()
    
    # 生成未标注样本（查询样本）
    query_batch = 8
    cls_features_query = torch.randn(query_batch, feature_dim).cuda()
    multi_layer_query = {
        'layer_3': torch.randn(query_batch, num_patches, feature_dim),
        'layer_6': torch.randn(query_batch, num_patches, feature_dim),
        'layer_9': torch.randn(query_batch, num_patches, feature_dim),
        'layer_11': torch.randn(query_batch, num_patches, feature_dim),
    }
    
    print(f"3. 查询样本: {query_batch}个")
    print()
    
    # 测试各种不确定性
    print("4. 计算Exploration Uncertainty:")
    u_exploration = estimator.compute_exploration_uncertainty(cls_features_query)
    print(f"  Shape: {u_exploration.shape}")
    print(f"  范围: [{u_exploration.min():.4f}, {u_exploration.max():.4f}]")
    print(f"  值: {u_exploration.cpu().numpy()}")
    print()
    
    print("5. 计算Boundary Uncertainty:")
    u_boundary = estimator.compute_boundary_uncertainty(cls_features_query)
    print(f"  Shape: {u_boundary.shape}")
    print(f"  范围: [{u_boundary.min():.4f}, {u_boundary.max():.4f}]")
    print(f"  值: {u_boundary.cpu().numpy()}")
    print()
    
    print("6. 计算Density Uncertainty:")
    u_density = estimator.compute_density_uncertainty(cls_features_query)
    print(f"  Shape: {u_density.shape}")
    print(f"  范围: [{u_density.min():.4f}, {u_density.max():.4f}]")
    print(f"  值: {u_density.cpu().numpy()}")
    print()
    
    print("7. 计算Multi-Scale Uncertainty:")
    u_multiscale = estimator.compute_multiscale_uncertainty(multi_layer_query)
    print(f"  Shape: {u_multiscale.shape}")
    print(f"  范围: [{u_multiscale.min():.4f}, {u_multiscale.max():.4f}]")
    print(f"  值: {u_multiscale.cpu().numpy()}")
    print()
    
    print("8. 计算所有不确定性（组合）:")
    all_uncertainties = estimator.compute_all_uncertainties(
        cls_features_query,
        multi_layer_query,
        weights={
            'exploration': 0.3,
            'boundary': 0.3,
            'density': 0.2,
            'multiscale': 0.2
        }
    )
    
    for name, values in all_uncertainties.items():
        print(f"  {name:12s}: 范围 [{values.min():.4f}, {values.max():.4f}], 均值 {values.mean():.4f}")
    print()
    
    # 展示top-3最不确定的样本
    print("9. Top-3 最不确定的样本:")
    combined_uncertainty = all_uncertainties['combined']
    top3_values, top3_indices = torch.topk(combined_uncertainty, k=3)
    
    for i, (idx, val) in enumerate(zip(top3_indices, top3_values)):
        print(f"  #{i+1}: 样本{idx.item()}, 不确定性={val.item():.4f}")
        print(f"       - Exploration: {all_uncertainties['exploration'][idx]:.4f}")
        print(f"       - Boundary: {all_uncertainties['boundary'][idx]:.4f}")
        print(f"       - Density: {all_uncertainties['density'][idx]:.4f}")
        print(f"       - Multi-Scale: {all_uncertainties['multiscale'][idx]:.4f}")
    
    print()
    print("=" * 60)
    print("✓ 测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    test_uncertainty_estimator()

