"""
Cascading Uncertainty Selector
- 逐次决策选择样本
- 基于4种不确定性的逻辑顺序
"""

import torch
from typing import Dict, Tuple
from enum import Enum


class Priority(Enum):
    """样本优先级"""
    NOISE_CANDIDATE = 0      # 可能是噪声，降低优先级
    LOW = 1                  # 不需要标注
    MEDIUM = 2               # 中等优先级
    HIGH = 3                 # 高优先级（新区域/决策边界）


class CascadingSelector:
    """
    逐次不确定性选择器
    
    决策逻辑（按顺序）:
    1. 密度不确定性 → 过滤噪声
    2. 探索性不确定性 → 新区域优先
    3. 边界不确定性 → 决策边界优先
    4. 多尺度不确定性 → 深度一致性筛选
    """
    
    def __init__(
        self,
        density_threshold: float = 0.5,
        exploration_threshold: float = 0.5,
        boundary_threshold: float = 0.6,
        multiscale_threshold: float = 0.5
    ):
        """
        Args:
            density_threshold: 密度不确定性阈值（超过此值认为可能是噪声）
            exploration_threshold: 探索性不确定性阈值
            boundary_threshold: 边界不确定性阈值
            multiscale_threshold: 多尺度不确定性阈值
        """
        self.thresholds = {
            'density': density_threshold,
            'exploration': exploration_threshold,
            'boundary': boundary_threshold,
            'multiscale': multiscale_threshold
        }
    
    def evaluate_sample(
        self,
        u_density: float,
        u_exploration: float,
        u_boundary: float,
        u_multiscale: float
    ) -> Priority:
        """
        评估单个样本的优先级（逐次决策）
        
        决策树:
        ```
        开始
         ↓
        密度高? → Yes → 可能噪声 (NOISE_CANDIDATE)
         ↓ No
        探索性高? → Yes → 新区域 (HIGH)
         ↓ No
        边界高? → Yes → 决策边界 (HIGH)
         ↓ No
        多尺度高? → Yes → 语义复杂 (MEDIUM)
         ↓ No
        不标注 (LOW)
        ```
        
        Args:
            u_density: 密度不确定性
            u_exploration: 探索性不确定性
            u_boundary: 边界不确定性
            u_multiscale: 多尺度不确定性
        
        Returns:
            priority: 样本优先级
        """
        # Step 1: 密度检查（噪声过滤）
        if u_density > self.thresholds['density']:
            # 密度不确定性高 → 邻居距离分散 → 可能是孤立样本/噪声
            return Priority.NOISE_CANDIDATE
        
        # Step 2: 探索性检查（新区域优先）
        if u_exploration > self.thresholds['exploration']:
            # 远离已标注样本 → 新区域 → 高优先级
            return Priority.HIGH
        
        # Step 3: 边界检查（决策边界优先）
        if u_boundary > self.thresholds['boundary']:
            # KNN标签混乱 → 决策边界 → 高优先级
            return Priority.HIGH
        
        # Step 4: 多尺度检查（深度一致性）
        if u_multiscale > self.thresholds['multiscale']:
            # 多层一致困难 → 语义复杂 → 中等优先级
            return Priority.MEDIUM
        
        # 都不满足 → 不需要标注
        return Priority.LOW
    
    def select_samples(
        self,
        uncertainties: Dict[str, torch.Tensor],
        budget: int,
        allow_noise: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用逐次决策选择要标注的样本
        
        Args:
            uncertainties: {
                'exploration': [N],
                'boundary': [N],
                'density': [N],
                'multiscale': [N]
            }
            budget: 标注预算（要选择多少个样本）
            allow_noise: 是否允许选择噪声候选样本
        
        Returns:
            selected_indices: [budget] 选中的样本索引
            priorities: [budget] 对应的优先级
        """
        num_samples = len(uncertainties['exploration'])
        
        # 评估每个样本的优先级
        priorities = []
        for i in range(num_samples):
            priority = self.evaluate_sample(
                u_density=uncertainties['density'][i].item(),
                u_exploration=uncertainties['exploration'][i].item(),
                u_boundary=uncertainties['boundary'][i].item(),
                u_multiscale=uncertainties['multiscale'][i].item()
            )
            priorities.append(priority.value)
        
        priorities = torch.tensor(priorities)
        
        # 按优先级排序
        sorted_indices = torch.argsort(priorities, descending=True)
        
        if not allow_noise:
            # 过滤掉噪声候选样本
            non_noise_mask = priorities[sorted_indices] > Priority.NOISE_CANDIDATE.value
            sorted_indices = sorted_indices[non_noise_mask]
        
        # 选择top-budget样本
        if len(sorted_indices) < budget:
            # 如果可选样本不足，全部选择
            selected_indices = sorted_indices
        else:
            selected_indices = sorted_indices[:budget]
        
        selected_priorities = priorities[selected_indices]
        
        return selected_indices, selected_priorities
    
    def get_priority_distribution(
        self,
        uncertainties: Dict[str, torch.Tensor]
    ) -> Dict[str, int]:
        """
        获取优先级分布统计
        
        Args:
            uncertainties: 不确定性字典
        
        Returns:
            distribution: {
                'NOISE_CANDIDATE': count,
                'LOW': count,
                'MEDIUM': count,
                'HIGH': count
            }
        """
        num_samples = len(uncertainties['exploration'])
        
        distribution = {
            'NOISE_CANDIDATE': 0,
            'LOW': 0,
            'MEDIUM': 0,
            'HIGH': 0
        }
        
        for i in range(num_samples):
            priority = self.evaluate_sample(
                u_density=uncertainties['density'][i].item(),
                u_exploration=uncertainties['exploration'][i].item(),
                u_boundary=uncertainties['boundary'][i].item(),
                u_multiscale=uncertainties['multiscale'][i].item()
            )
            distribution[priority.name] += 1
        
        return distribution


def test_cascading_selector():
    """测试逐次选择器"""
    print("=" * 60)
    print("测试Cascading Uncertainty Selector")
    print("=" * 60)
    print()
    
    # 创建选择器
    selector = CascadingSelector(
        density_threshold=0.5,
        exploration_threshold=0.5,
        boundary_threshold=0.6,
        multiscale_threshold=0.5
    )
    print("✓ Cascading Selector已创建")
    print(f"  阈值设置:")
    print(f"    - 密度: {selector.thresholds['density']}")
    print(f"    - 探索性: {selector.thresholds['exploration']}")
    print(f"    - 边界: {selector.thresholds['boundary']}")
    print(f"    - 多尺度: {selector.thresholds['multiscale']}")
    print()
    
    # 模拟一些样本的不确定性
    print("1. 测试不同类型样本的优先级:")
    print()
    
    test_cases = [
        {
            'name': '噪声样本（密度高）',
            'density': 0.8,
            'exploration': 0.3,
            'boundary': 0.4,
            'multiscale': 0.3,
            'expected': 'NOISE_CANDIDATE'
        },
        {
            'name': '新区域样本（探索性高）',
            'density': 0.2,
            'exploration': 0.9,
            'boundary': 0.3,
            'multiscale': 0.4,
            'expected': 'HIGH'
        },
        {
            'name': '决策边界样本（边界高）',
            'density': 0.3,
            'exploration': 0.4,
            'boundary': 0.8,
            'multiscale': 0.5,
            'expected': 'HIGH'
        },
        {
            'name': '语义复杂样本（多尺度高）',
            'density': 0.2,
            'exploration': 0.3,
            'boundary': 0.4,
            'multiscale': 0.7,
            'expected': 'MEDIUM'
        },
        {
            'name': '普通样本（都不高）',
            'density': 0.2,
            'exploration': 0.3,
            'boundary': 0.4,
            'multiscale': 0.3,
            'expected': 'LOW'
        },
        {
            'name': '高价值样本（探索+边界都高）',
            'density': 0.3,
            'exploration': 0.9,
            'boundary': 0.8,
            'multiscale': 0.6,
            'expected': 'HIGH'  # 探索性先判断
        }
    ]
    
    for test_case in test_cases:
        priority = selector.evaluate_sample(
            u_density=test_case['density'],
            u_exploration=test_case['exploration'],
            u_boundary=test_case['boundary'],
            u_multiscale=test_case['multiscale']
        )
        
        status = "✅" if priority.name == test_case['expected'] else "❌"
        print(f"{status} {test_case['name']}")
        print(f"    不确定性: D={test_case['density']:.1f}, E={test_case['exploration']:.1f}, "
              f"B={test_case['boundary']:.1f}, M={test_case['multiscale']:.1f}")
        print(f"    优先级: {priority.name} (期望: {test_case['expected']})")
        print()
    
    # 测试批量选择
    print("2. 测试批量样本选择:")
    print()
    
    # 生成随机不确定性
    num_samples = 20
    uncertainties = {
        'exploration': torch.rand(num_samples),
        'boundary': torch.rand(num_samples),
        'density': torch.rand(num_samples),
        'multiscale': torch.rand(num_samples)
    }
    
    print(f"总样本数: {num_samples}")
    
    # 获取优先级分布
    distribution = selector.get_priority_distribution(uncertainties)
    print(f"\n优先级分布:")
    for priority_name, count in distribution.items():
        print(f"  {priority_name:20s}: {count:2d} ({count/num_samples*100:5.1f}%)")
    
    # 选择top-5样本
    budget = 5
    selected_indices, selected_priorities = selector.select_samples(
        uncertainties,
        budget=budget,
        allow_noise=False
    )
    
    print(f"\n选择 top-{budget} 样本 (不包括噪声):")
    for i, (idx, priority) in enumerate(zip(selected_indices, selected_priorities)):
        priority_name = Priority(priority.item()).name
        print(f"  #{i+1}: 样本{idx:2d}, 优先级={priority_name}")
        print(f"       不确定性: E={uncertainties['exploration'][idx]:.3f}, "
              f"B={uncertainties['boundary'][idx]:.3f}, "
              f"D={uncertainties['density'][idx]:.3f}, "
              f"M={uncertainties['multiscale'][idx]:.3f}")
    
    print()
    print("=" * 60)
    print("✓ 测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    test_cascading_selector()

