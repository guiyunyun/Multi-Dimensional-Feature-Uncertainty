"""
Cascading Uncertainty Selector V1 - 综合判断版
- 考虑所有不确定性后再决策
- 高价值信号可以"拯救"密度异常的样本
- 多个不确定性同时high时，优先级更高
"""

import torch
from typing import Dict, Tuple
from enum import Enum


def _get_config():
    """获取 Config 对象"""
    try:
        from .config import Config
    except ImportError:
        from config import Config
    return Config


class Priority(Enum):
    """样本优先级"""
    NOISE_CANDIDATE = 0      # 可能是噪声，降低优先级
    LOW = 1                  # 不需要标注
    MEDIUM = 2               # 中等优先级
    HIGH = 3                 # 高优先级（新区域/决策边界）
    VERY_HIGH = 4            # 超高优先级（多个不确定性叠加）


class CascadingSelectorV1:
    """
    综合判断版选择器
    
    核心思想:
    1. 先检查是否有高价值信号（探索性/边界/多尺度）
    2. 如果有，即使密度高也保留（可能是类内/类间边界）
    3. 多个不确定性同时high时，优先级更高（信号叠加）
    4. 只有在没有任何高价值信号时，密度高才判为噪声
    """
    
    def __init__(
        self,
        density_threshold: float = None,
        exploration_threshold: float = None,
        boundary_threshold: float = None,
        multiscale_threshold: float = None
    ):
        """
        Args:
            density_threshold: 密度不确定性阈值
            exploration_threshold: 探索性不确定性阈值
            boundary_threshold: 边界不确定性阈值
            multiscale_threshold: 多尺度不确定性阈值
        """
        Config = _get_config()
        
        # 从 Config 读取默认值
        if density_threshold is None:
            density_threshold = getattr(Config, 'density_threshold', 0.5)
        if exploration_threshold is None:
            exploration_threshold = getattr(Config, 'exploration_threshold', 0.5)
        if boundary_threshold is None:
            boundary_threshold = getattr(Config, 'boundary_threshold', 0.6)
        if multiscale_threshold is None:
            multiscale_threshold = getattr(Config, 'multiscale_threshold', 0.5)
        
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
        综合评估单个样本的优先级
        
        决策逻辑:
        ```
        1. 计算高价值信号数量
        2. 如果有高价值信号:
           - 3个high → VERY_HIGH（最优先）
           - 2个high → HIGH（高优先）
           - 1个high → HIGH/MEDIUM（根据类型）
        3. 如果没有高价值信号:
           - 密度高 → NOISE_CANDIDATE
           - 密度低 → LOW
        ```
        
        Args:
            u_density: 密度不确定性
            u_exploration: 探索性不确定性
            u_boundary: 边界不确定性
            u_multiscale: 多尺度不确定性
        
        Returns:
            priority: 样本优先级
        """
        # Step 1: 检查哪些不确定性是high
        is_exploration_high = u_exploration > self.thresholds['exploration']
        is_boundary_high = u_boundary > self.thresholds['boundary']
        is_multiscale_high = u_multiscale > self.thresholds['multiscale']
        
        # 计算高价值信号数量
        high_signals = sum([is_exploration_high, is_boundary_high, is_multiscale_high])
        
        # Step 2: 如果有高价值信号，根据叠加程度决定优先级
        if high_signals > 0:
            # 多个不确定性叠加 → 超高优先级
            if high_signals >= 3:
                # 探索性+边界+多尺度都高 → 非常有价值！
                return Priority.VERY_HIGH
            
            elif high_signals == 2:
                # 两个不确定性高 → 高优先级
                # 特别组合分析
                if is_exploration_high and is_boundary_high:
                    # 新区域 + 决策边界 → 最有价值的组合
                    return Priority.VERY_HIGH
                else:
                    # 其他两两组合也很好
                    return Priority.HIGH
            
            else:  # high_signals == 1
                # 只有一个不确定性高
                if is_exploration_high:
                    # 新区域 → 高优先级
                    return Priority.HIGH
                elif is_boundary_high:
                    # 决策边界 → 高优先级
                    return Priority.HIGH
                elif is_multiscale_high:
                    # 多尺度 → 中等优先级
                    return Priority.MEDIUM
        
        # Step 3: 没有高价值信号，检查密度
        if u_density > self.thresholds['density']:
            # 密度高 + 没有其他信号 → 可能是噪声
            return Priority.NOISE_CANDIDATE
        
        # Step 4: 都不满足
        return Priority.LOW
    
    def select_samples(
        self,
        uncertainties: Dict[str, torch.Tensor],
        budget: int,
        allow_noise: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用综合判断选择要标注的样本
        
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
            distribution: 各优先级的样本数量
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
            priority = self.evaluate_sample(
                u_density=uncertainties['density'][i].item(),
                u_exploration=uncertainties['exploration'][i].item(),
                u_boundary=uncertainties['boundary'][i].item(),
                u_multiscale=uncertainties['multiscale'][i].item()
            )
            distribution[priority.name] += 1
        
        return distribution
    
    def get_signal_analysis(
        self,
        uncertainties: Dict[str, torch.Tensor]
    ) -> Dict[str, int]:
        """
        分析高价值信号的叠加情况
        
        Returns:
            analysis: {
                '3_signals': count,  # 3个不确定性都high
                '2_signals': count,  # 2个不确定性high
                '1_signal': count,   # 1个不确定性high
                '0_signal': count    # 没有high
            }
        """
        num_samples = len(uncertainties['exploration'])
        
        signal_counts = {
            '3_signals': 0,
            '2_signals': 0,
            '1_signal': 0,
            '0_signal': 0
        }
        
        for i in range(num_samples):
            is_exploration_high = uncertainties['exploration'][i] > self.thresholds['exploration']
            is_boundary_high = uncertainties['boundary'][i] > self.thresholds['boundary']
            is_multiscale_high = uncertainties['multiscale'][i] > self.thresholds['multiscale']
            
            high_count = sum([is_exploration_high, is_boundary_high, is_multiscale_high])
            
            if high_count >= 3:
                signal_counts['3_signals'] += 1
            elif high_count == 2:
                signal_counts['2_signals'] += 1
            elif high_count == 1:
                signal_counts['1_signal'] += 1
            else:
                signal_counts['0_signal'] += 1
        
        return signal_counts


def test_cascading_selector_v1():
    """测试综合判断版选择器"""
    Config = _get_config()
    
    print("=" * 70)
    print("测试Cascading Selector V1 - 综合判断版")
    print("=" * 70)
    print()
    
    print(f"配置来源: config.py")
    print(f"  density_threshold: {Config.density_threshold}")
    print(f"  exploration_threshold: {Config.exploration_threshold}")
    print(f"  boundary_threshold: {Config.boundary_threshold}")
    print(f"  multiscale_threshold: {Config.multiscale_threshold}")
    print()
    
    # 创建选择器（参数从 Config 读取）
    selector = CascadingSelectorV1()
    print(f"✓ Cascading Selector V1已创建")
    print(f"  阈值: {selector.thresholds}")
    print()
    
    # 测试不同类型样本
    print("1. 测试不同类型样本的优先级:")
    print()
    
    test_cases = [
        {
            'name': '🔥 超级样本（3个high）',
            'density': 0.3,
            'exploration': 0.9,
            'boundary': 0.8,
            'multiscale': 0.7,
            'expected': 'VERY_HIGH'
        },
        {
            'name': '🌟 高价值样本（探索+边界high）',
            'density': 0.3,
            'exploration': 0.9,
            'boundary': 0.8,
            'multiscale': 0.3,
            'expected': 'VERY_HIGH'
        },
        {
            'name': '⭐ 好样本（边界+多尺度high）',
            'density': 0.3,
            'exploration': 0.3,
            'boundary': 0.8,
            'multiscale': 0.7,
            'expected': 'HIGH'
        },
        {
            'name': '🎯 新区域样本（只有探索性high）',
            'density': 0.2,
            'exploration': 0.9,
            'boundary': 0.3,
            'multiscale': 0.4,
            'expected': 'HIGH'
        },
        {
            'name': '🦊 狐狸（边界high + 密度也高）',
            'density': 0.7,  # 密度高
            'exploration': 0.3,
            'boundary': 0.8,  # 但边界也高
            'multiscale': 0.4,
            'expected': 'HIGH'  # 应该保留！
        },
        {
            'name': '🐺 哈士奇（类内边界，密度高）',
            'density': 0.6,  # 密度高（大小型犬差异）
            'exploration': 0.3,
            'boundary': 0.4,
            'multiscale': 0.6,  # 多尺度high
            'expected': 'MEDIUM'  # 应该保留！
        },
        {
            'name': '❌ 真噪声（只有密度high）',
            'density': 0.8,
            'exploration': 0.3,
            'boundary': 0.4,
            'multiscale': 0.3,
            'expected': 'NOISE_CANDIDATE'
        },
        {
            'name': '💤 普通样本（都不high）',
            'density': 0.2,
            'exploration': 0.3,
            'boundary': 0.4,
            'multiscale': 0.3,
            'expected': 'LOW'
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
        print(f"    不确定性: E={test_case['exploration']:.1f}, "
              f"B={test_case['boundary']:.1f}, "
              f"D={test_case['density']:.1f}, "
              f"M={test_case['multiscale']:.1f}")
        print(f"    优先级: {priority.name:15s} (期望: {test_case['expected']})")
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
    
    # 信号叠加分析
    signal_analysis = selector.get_signal_analysis(uncertainties)
    print(f"\n高价值信号叠加分析:")
    for signal_type, count in signal_analysis.items():
        print(f"  {signal_type:15s}: {count:2d} ({count/num_samples*100:5.1f}%)")
    
    # 选择top-8样本
    budget = 8
    selected_indices, selected_priorities = selector.select_samples(
        uncertainties,
        budget=budget,
        allow_noise=False
    )
    
    print(f"\n选择 top-{budget} 样本:")
    for i, (idx, priority) in enumerate(zip(selected_indices, selected_priorities)):
        priority_name = Priority(priority.item()).name
        
        # 计算该样本有几个high
        is_e_high = uncertainties['exploration'][idx] > 0.5
        is_b_high = uncertainties['boundary'][idx] > 0.6
        is_m_high = uncertainties['multiscale'][idx] > 0.5
        high_count = sum([is_e_high, is_b_high, is_m_high])
        
        print(f"  #{i+1}: 样本{idx:2d}, 优先级={priority_name:12s} ({high_count}个high)")
        print(f"       E={uncertainties['exploration'][idx]:.3f}{'✓' if is_e_high else ' '}, "
              f"B={uncertainties['boundary'][idx]:.3f}{'✓' if is_b_high else ' '}, "
              f"D={uncertainties['density'][idx]:.3f}, "
              f"M={uncertainties['multiscale'][idx]:.3f}{'✓' if is_m_high else ' '}")
    
    print()
    print("=" * 70)
    print("✓ 测试完成!")
    print("=" * 70)


if __name__ == "__main__":
    test_cascading_selector_v1()

