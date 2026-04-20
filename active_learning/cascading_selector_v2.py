"""
Cascading Uncertainty Selector V2 - 加权分数版
- 所有不确定性参与计算
- 密度作为置信度惩罚项（不直接过滤）
- 更灵活的分数机制
"""

import torch
from typing import Dict, Tuple
from enum import Enum


class Priority(Enum):
    """样本优先级"""
    NOISE_CANDIDATE = 0      # 可能是噪声，降低优先级
    LOW = 1                  # 不需要标注
    MEDIUM = 2               # 中等优先级
    HIGH = 3                 # 高优先级
    VERY_HIGH = 4            # 超高优先级


class CascadingSelectorV2:
    """
    加权分数版选择器
    
    核心思想:
    1. 每个不确定性贡献一个价值分数
    2. 密度不确定性作为置信度惩罚（不直接过滤）
    3. 最终分数 = 价值分数 × 置信度
    4. 根据最终分数分档决定优先级
    """
    
    def __init__(
        self,
        exploration_weight: float = 0.35,
        boundary_weight: float = 0.35,
        multiscale_weight: float = 0.30,
        density_penalty: float = 1.0
    ):
        """
        Args:
            exploration_weight: 探索性不确定性权重
            boundary_weight: 边界不确定性权重
            multiscale_weight: 多尺度不确定性权重
            density_penalty: 密度惩罚系数（越大，密度影响越大）
        """
        self.weights = {
            'exploration': exploration_weight,
            'boundary': boundary_weight,
            'multiscale': multiscale_weight
        }
        self.density_penalty = density_penalty
        
        # 分数阈值（用于分档）
        self.score_thresholds = {
            'very_high': 0.65,  # >= 0.65 → VERY_HIGH
            'high': 0.45,       # >= 0.45 → HIGH
            'medium': 0.25,     # >= 0.25 → MEDIUM
            'noise': 0.15       # < 0.15 → NOISE_CANDIDATE
        }
    
    def compute_value_score(
        self,
        u_exploration: float,
        u_boundary: float,
        u_multiscale: float
    ) -> float:
        """
        计算价值分数
        
        价值分数 = Σ (weight_i × uncertainty_i)
        
        Args:
            u_exploration: 探索性不确定性
            u_boundary: 边界不确定性
            u_multiscale: 多尺度不确定性
        
        Returns:
            value_score: 价值分数 [0, 1]
        """
        value_score = (
            self.weights['exploration'] * u_exploration +
            self.weights['boundary'] * u_boundary +
            self.weights['multiscale'] * u_multiscale
        )
        return value_score
    
    def compute_confidence(
        self,
        u_density: float
    ) -> float:
        """
        计算置信度（密度作为惩罚项）
        
        置信度 = exp(-penalty × density)
        
        物理意义:
        - 密度低 (u_density≈0) → 置信度高 (≈1.0)
        - 密度高 (u_density≈1) → 置信度低 (≈0.37 when penalty=1)
        
        Args:
            u_density: 密度不确定性
        
        Returns:
            confidence: 置信度 [0, 1]
        """
        confidence = torch.exp(torch.tensor(-self.density_penalty * u_density)).item()
        return confidence
    
    def evaluate_sample(
        self,
        u_density: float,
        u_exploration: float,
        u_boundary: float,
        u_multiscale: float
    ) -> Tuple[Priority, float]:
        """
        评估单个样本的优先级和分数
        
        计算流程:
        1. 价值分数 = 加权组合(探索性, 边界, 多尺度)
        2. 置信度 = exp(-density_penalty × density)
        3. 最终分数 = 价值分数 × 置信度
        4. 根据最终分数分档
        
        Args:
            u_density: 密度不确定性
            u_exploration: 探索性不确定性
            u_boundary: 边界不确定性
            u_multiscale: 多尺度不确定性
        
        Returns:
            priority: 样本优先级
            final_score: 最终分数
        """
        # 计算价值分数
        value_score = self.compute_value_score(
            u_exploration, u_boundary, u_multiscale
        )
        
        # 计算置信度（密度惩罚）
        confidence = self.compute_confidence(u_density)
        
        # 最终分数
        final_score = value_score * confidence
        
        # 根据分数分档
        if final_score >= self.score_thresholds['very_high']:
            priority = Priority.VERY_HIGH
        elif final_score >= self.score_thresholds['high']:
            priority = Priority.HIGH
        elif final_score >= self.score_thresholds['medium']:
            priority = Priority.MEDIUM
        elif final_score >= self.score_thresholds['noise']:
            priority = Priority.LOW
        else:
            priority = Priority.NOISE_CANDIDATE
        
        return priority, final_score
    
    def select_samples(
        self,
        uncertainties: Dict[str, torch.Tensor],
        budget: int,
        allow_noise: bool = False,
        return_scores: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用加权分数选择要标注的样本
        
        Args:
            uncertainties: 不确定性字典
            budget: 标注预算
            allow_noise: 是否允许选择噪声候选样本
            return_scores: 是否返回分数而不是优先级
        
        Returns:
            selected_indices: 选中的样本索引
            selected_values: 优先级或分数
        """
        num_samples = len(uncertainties['exploration'])
        
        # 评估每个样本
        priorities = []
        scores = []
        
        for i in range(num_samples):
            priority, score = self.evaluate_sample(
                u_density=uncertainties['density'][i].item(),
                u_exploration=uncertainties['exploration'][i].item(),
                u_boundary=uncertainties['boundary'][i].item(),
                u_multiscale=uncertainties['multiscale'][i].item()
            )
            priorities.append(priority.value)
            scores.append(score)
        
        priorities = torch.tensor(priorities)
        scores = torch.tensor(scores)
        
        # 按分数排序（分数更精细）
        sorted_indices = torch.argsort(scores, descending=True)
        
        if not allow_noise:
            # 过滤掉噪声候选样本
            non_noise_mask = priorities[sorted_indices] > Priority.NOISE_CANDIDATE.value
            sorted_indices = sorted_indices[non_noise_mask]
        
        # 选择top-budget样本
        if len(sorted_indices) < budget:
            selected_indices = sorted_indices
        else:
            selected_indices = sorted_indices[:budget]
        
        if return_scores:
            selected_values = scores[selected_indices]
        else:
            selected_values = priorities[selected_indices]
        
        return selected_indices, selected_values
    
    def get_priority_distribution(
        self,
        uncertainties: Dict[str, torch.Tensor]
    ) -> Dict[str, int]:
        """
        获取优先级分布统计
        """
        num_samples = len(uncertainties['exploration'])
        
        distribution = {
            'NOISE_CANDIDATE': 0,
            'LOW': 0,
            'MEDIUM': 0,
            'HIGH': 0,
            'VERY_HIGH': 0
        }
        
        for i in range(num_samples):
            priority, _ = self.evaluate_sample(
                u_density=uncertainties['density'][i].item(),
                u_exploration=uncertainties['exploration'][i].item(),
                u_boundary=uncertainties['boundary'][i].item(),
                u_multiscale=uncertainties['multiscale'][i].item()
            )
            distribution[priority.name] += 1
        
        return distribution
    
    def analyze_score_distribution(
        self,
        uncertainties: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        分析分数分布
        
        Returns:
            statistics: {
                'mean': 平均分数,
                'std': 标准差,
                'min': 最小值,
                'max': 最大值
            }
        """
        num_samples = len(uncertainties['exploration'])
        
        scores = []
        for i in range(num_samples):
            _, score = self.evaluate_sample(
                u_density=uncertainties['density'][i].item(),
                u_exploration=uncertainties['exploration'][i].item(),
                u_boundary=uncertainties['boundary'][i].item(),
                u_multiscale=uncertainties['multiscale'][i].item()
            )
            scores.append(score)
        
        scores = torch.tensor(scores)
        
        return {
            'mean': scores.mean().item(),
            'std': scores.std().item(),
            'min': scores.min().item(),
            'max': scores.max().item()
        }


def test_cascading_selector_v2():
    """测试加权分数版选择器"""
    print("=" * 70)
    print("测试Cascading Selector V2 - 加权分数版")
    print("=" * 70)
    print()
    
    # 创建选择器
    selector = CascadingSelectorV2(
        exploration_weight=0.35,
        boundary_weight=0.35,
        multiscale_weight=0.30,
        density_penalty=1.0
    )
    print("✓ Cascading Selector V2已创建")
    print(f"  权重: E={selector.weights['exploration']:.2f}, "
          f"B={selector.weights['boundary']:.2f}, "
          f"M={selector.weights['multiscale']:.2f}")
    print(f"  密度惩罚系数: {selector.density_penalty:.2f}")
    print()
    
    # 测试不同类型样本
    print("1. 测试不同类型样本的分数和优先级:")
    print()
    
    test_cases = [
        {
            'name': '🔥 完美样本（都高+密度低）',
            'density': 0.1,
            'exploration': 0.9,
            'boundary': 0.9,
            'multiscale': 0.9
        },
        {
            'name': '⭐ 高价值样本（探索+边界高）',
            'density': 0.3,
            'exploration': 0.9,
            'boundary': 0.8,
            'multiscale': 0.4
        },
        {
            'name': '🦊 狐狸（边界高但密度也高）',
            'density': 0.7,
            'exploration': 0.3,
            'boundary': 0.8,
            'multiscale': 0.4
        },
        {
            'name': '🐺 哈士奇（多尺度高+密度高）',
            'density': 0.6,
            'exploration': 0.3,
            'boundary': 0.4,
            'multiscale': 0.7
        },
        {
            'name': '❌ 疑似噪声（只有密度高）',
            'density': 0.8,
            'exploration': 0.3,
            'boundary': 0.4,
            'multiscale': 0.3
        },
        {
            'name': '💤 普通样本（都不高）',
            'density': 0.2,
            'exploration': 0.3,
            'boundary': 0.4,
            'multiscale': 0.3
        }
    ]
    
    for test_case in test_cases:
        priority, score = selector.evaluate_sample(
            u_density=test_case['density'],
            u_exploration=test_case['exploration'],
            u_boundary=test_case['boundary'],
            u_multiscale=test_case['multiscale']
        )
        
        # 计算价值分数和置信度
        value = selector.compute_value_score(
            test_case['exploration'],
            test_case['boundary'],
            test_case['multiscale']
        )
        confidence = selector.compute_confidence(test_case['density'])
        
        print(f"{test_case['name']}")
        print(f"    不确定性: E={test_case['exploration']:.1f}, "
              f"B={test_case['boundary']:.1f}, "
              f"D={test_case['density']:.1f}, "
              f"M={test_case['multiscale']:.1f}")
        print(f"    价值分数: {value:.3f}, 置信度: {confidence:.3f}")
        print(f"    最终分数: {score:.3f} → 优先级: {priority.name}")
        print()
    
    # 测试批量样本
    print("2. 测试批量样本选择:")
    print()
    
    num_samples = 30
    uncertainties = {
        'exploration': torch.rand(num_samples),
        'boundary': torch.rand(num_samples),
        'density': torch.rand(num_samples),
        'multiscale': torch.rand(num_samples)
    }
    
    print(f"总样本数: {num_samples}")
    
    # 优先级分布
    distribution = selector.get_priority_distribution(uncertainties)
    print(f"\n优先级分布:")
    for priority_name, count in distribution.items():
        print(f"  {priority_name:20s}: {count:2d} ({count/num_samples*100:5.1f}%)")
    
    # 分数统计
    score_stats = selector.analyze_score_distribution(uncertainties)
    print(f"\n分数统计:")
    print(f"  平均值: {score_stats['mean']:.3f}")
    print(f"  标准差: {score_stats['std']:.3f}")
    print(f"  范围: [{score_stats['min']:.3f}, {score_stats['max']:.3f}]")
    
    # 选择top-8样本
    budget = 8
    selected_indices, selected_scores = selector.select_samples(
        uncertainties,
        budget=budget,
        allow_noise=False,
        return_scores=True
    )
    
    print(f"\n选择 top-{budget} 样本 (按分数排序):")
    for i, (idx, score) in enumerate(zip(selected_indices, selected_scores)):
        priority, _ = selector.evaluate_sample(
            u_density=uncertainties['density'][idx].item(),
            u_exploration=uncertainties['exploration'][idx].item(),
            u_boundary=uncertainties['boundary'][idx].item(),
            u_multiscale=uncertainties['multiscale'][idx].item()
        )
        
        print(f"  #{i+1}: 样本{idx:2d}, 分数={score:.3f}, 优先级={priority.name}")
        print(f"       E={uncertainties['exploration'][idx]:.3f}, "
              f"B={uncertainties['boundary'][idx]:.3f}, "
              f"D={uncertainties['density'][idx]:.3f}, "
              f"M={uncertainties['multiscale'][idx]:.3f}")
    
    print()
    print("=" * 70)
    print("✓ 测试完成!")
    print("=" * 70)


if __name__ == "__main__":
    test_cascading_selector_v2()

