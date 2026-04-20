"""
配置文件 - 集中管理所有参数
按照模块文件分区组织，方便查找和修改

使用方式: from config import Config, 然后 Config.param_name 访问
运行方式: python main.py（零参数，所有配置在此文件修改）
"""


class Config:
    """主动学习实验配置"""
    
    # ============================================================
    # 实验配置 (Experiment Settings)
    # - 每次实验前必须检查这里！
    # ============================================================
    
    # 实验名称（结果保存时自动加上时间戳：exp_name_YYYYMMDD_HHMM）
    exp_name = 'closed_loop_full'
    
    # 结果保存目录（相对于 active_learning/ 目录）
    results_dir = '../results'
    
    # 消融实验控制
    # - True: 随机采样（Random Sampling baseline）
    # - False: 使用主动学习策略
    random_sampling = False
    
    # 参与融合的特征不确定性种类（消融实验核心配置）
    # 完整版: ['exploration', 'boundary', 'density', 'multiscale']
    # 消融实验示例:
    #   - E only:     ['exploration']
    #   - E + B:      ['exploration', 'boundary']
    #   - E + B + M:  ['exploration', 'boundary', 'multiscale']
    #   - E + B + D + M (完整版): ['exploration', 'boundary', 'density', 'multiscale']
    active_feature_uncertainties = ['exploration', 'boundary', 'density', 'multiscale']
    
    # ============================================================
    # 1. feature_extractor.py - DINOv3特征提取器
    # ============================================================
    
    # DINOv3 模型配置
    model_size = 'base'             # 'small'(384维) / 'base'(768维) / 'large'(1024维)
    feature_dim = 768               # 特征维度（与model_size对应）
    feature_layers = [3, 6, 9, 11]  # 要提取的中间层（用于多尺度不确定性）
    
    # 预训练权重路径（相对于 dinov3/ 目录）
    pretrained_weights_dir = 'pretrained_models'
    
    # ============================================================
    # 2. dataset.py - 数据集管理
    # ============================================================
    
    # 数据集路径（AutoDL服务器）
    data_root = '/root/autodl-tmp/dinov3/data/imagenet100_split'
    # data_root = '/root/autodl-tmp/dinov3/data/ImageNet100'
    
    # 数据集基本信息
    num_classes = 100               # 类别数
    image_size = 224                # 输入图像尺寸
    
    # ImageNet 归一化参数
    normalize_mean = [0.485, 0.456, 0.406]
    normalize_std = [0.229, 0.224, 0.225]
    
    # 数据加载
    batch_size = 256                 # 批大小（推理/特征提取用）
    num_workers = 8                 # 数据加载线程数
    
    # ====== 百分比采样配置（推荐使用）======
    # 设置最终使用训练集的百分比和轮数，自动计算每轮样本数
    use_percentage_sampling = True   # True: 使用百分比配置，False: 使用固定数值
    final_data_percentage = 5.0      # 最终使用训练集的百分比（如 1.0, 5.0, 10.0）
    # 注意: total_rounds 在下方 active_learner.py 区域定义
    
    # ====== 固定数值采样配置（use_percentage_sampling=False 时使用）======
    # 初始标注池
    # 策略1：按类别均匀采样（分层采样）
    initial_samples_per_class = 2   # 每类初始标注样本数（设为0则禁用此策略）
    
    # 策略2：完全随机采样（更接近真实场景）
    use_random_initial_pool = True  # True: 使用完全随机初始池
    initial_random_samples = 200     # 完全随机时的初始样本数（use_percentage_sampling=False时使用）
    
    seed = 42                       # 随机种子
    
    # ============================================================
    # 3. memory_bank.py - Memory Bank (存储已标注样本特征)
    # ============================================================
    
    # Memory Bank 会自动使用以下配置:
    # - feature_dim: 来自上面的 feature_dim
    # - num_classes: 来自上面的 num_classes
    # - layers: 来自上面的 feature_layers
    
    # 设备配置
    device = 'cuda'                 # 'cuda' / 'cpu'
    
    # ============================================================
    # 4. uncertainty.py - 4种特征不确定性计算
    # ============================================================
    
    # K近邻参数
    k_neighbors = 10                # KNN的K值
    
    # 特征不确定性权重（开环模式下 compute_all_uncertainties 的默认权重）
    # 注意：闭环模式使用 attention_modulation，会动态调整权重
    feature_uncertainty_weights = {
        'exploration': 0.25,        # 探索未知区域
        'boundary': 0.25,           # 决策边界
        'density': 0.25,            # 密度/噪声过滤
        'multiscale': 0.25          # 多层一致性
    }
    
    # ============================================================
    # 5. prediction_uncertainty.py - 预测不确定性（闭环模式）
    # ============================================================
    
    # 是否使用预测不确定性（True=闭环, False=开环，random sampling时设置为False）
    use_prediction_uncertainty = True
    
    # 预测不确定性模式: 'entropy', 'margin', 'combined'
    # - 'entropy': 只使用熵（推荐，物理意义明确）
    # - 'margin': 只使用间隔（Top-2概率差）
    # - 'combined': 4种加权组合
    prediction_uncertainty_mode = 'entropy'
    
    # 各模式的权重配置
    PREDICTION_UNCERTAINTY_WEIGHTS = {
        'entropy': {
            'entropy': 1.0,
            'margin': 0.0,
            'confidence': 0.0,
            'variance': 0.0
        },
        'margin': {
            'entropy': 0.0,
            'margin': 1.0,
            'confidence': 0.0,
            'variance': 0.0
        },
        'combined': {
            'entropy': 0.4,
            'margin': 0.3,
            'confidence': 0.2,
            'variance': 0.1
        }
    }
    
    # 融合策略: 'attention', 'multiply', 'add'
    # - 'attention': 动态注意力调制
    # - 'multiply': 简单相乘
    # - 'add': 简单相加
    fusion_strategy = 'attention'
    
    # 融合权重（attention 策略的基础权重，用于 add/multiply 策略）
    feature_uncertainty_weight = 0.7    # 特征不确定性权重
    prediction_uncertainty_weight = 0.3  # 预测不确定性权重
    
    # attention_modulation 动态权重配置
    # 原始配置: exploration=0.25, boundary=0.25, multiscale=0.25, density=0.25
    # 新配置:   exploration=0.40, boundary=0.25, multiscale=0.20, density=0.15
    attention_weights = {
        'exploration': {'base': 0.25, 'boost': 0.15}, 
        'boundary':    {'base': 0.25, 'boost': 0.15}, 
        'multiscale':  {'base': 0.25, 'boost': -0.10},
        'density':     {'base': 0.25, 'boost': -0.20} 
    }
    # attention_weights = {
    #     'exploration': {'base': 0.40, 'boost': 0.05},   # [0.40, 0.45]
    #     'boundary':    {'base': 0.25, 'boost': 0.05},   # [0.25, 0.30]
    #     'multiscale':  {'base': 0.20, 'boost': -0.05},  # [0.15, 0.20]
    #     'density':     {'base': 0.15, 'boost': -0.05}   # [0.10, 0.15]
    # }
    
    # 动态权重调整（根据训练轮次）
    # - True: 早期强调exploration，后期逐渐转向boundary
    # - False: 使用固定的attention_weights
    use_dynamic_weights = False
    
    # 动态权重配置（仅 use_dynamic_weights=True 时生效）
    # 权重变化: early_phase_ratio 比例内使用early权重，之后线性过渡到late权重
    dynamic_weights_config = {
        'early_phase_ratio': 0.25,  # 前40%轮次为早期阶段
        'early': {
            'exploration': 0.50,   # 早期强调探索
            'boundary':    0.20,
            'multiscale':  0.15,
            'density':     0.15
        },
        'late': {
            'exploration': 0.25,   # 后期平衡
            'boundary':    0.30,   # 后期强调边界
            'multiscale':  0.25,
            'density':     0.20
        }
    }
    
    # 调制因子范围（预测不确定性高时放大特征不确定性）
    modulation_factor_range = (1.0, 1.5)  # [1.0, 1.5]
    
    # ============================================================
    # 6. cascading_selector_v1.py - 级联选择器 V1
    # ============================================================
    
    # 不确定性阈值（用于判断 high/low）
    # 值越大，条件越严格，通过的样本越少
    exploration_threshold = 0.5     # 探索性阈值：相似度 < 阈值 视为未探索
    boundary_threshold = 0.6        # 边界阈值：熵 > 阈值 视为边界样本
    density_threshold = 0.5         # 密度阈值：标准差 > 阈值 视为孤立/噪声
    multiscale_threshold = 0.5      # 多尺度阈值：一致性 > 阈值 视为复杂样本
    
    # ============================================================
    # 7. classifier.py - 分类头
    # ============================================================
    
    # 分类器类型: 'simple' / 'mlp'
    # - 'simple': 单层线性（768 → 100）
    # - 'mlp': 两层MLP（768 → 512 → 100）
    classifier_type = 'simple'
    
    # MLP隐藏层维度（仅 classifier_type='mlp' 时使用）
    classifier_hidden_dim = 512
    
    # Dropout率
    classifier_dropout = 0.1
    
    # 训练参数
    classifier_lr = 1e-3            # 学习率
    classifier_epochs = 20          # 每轮主动学习中分类器训练的 epoch 数
    # classifier_epochs = 50          # 每轮主动学习中分类器训练的 epoch 数
    
    # ============================================================
    # 8. active_learner.py - 主动学习主循环
    # ============================================================
    
    # 主动学习循环配置
    total_rounds = 10               # 总轮数（初始池 + 10轮 = 11份）
    
    # 以下参数在 use_percentage_sampling=True 时由 main.py 自动计算
    # 在 use_percentage_sampling=False 时需要手动设置
    budget_per_round = 200          # 每轮选择的样本数（百分比模式下自动覆盖）
    initial_samples = 200           # 初始样本数（百分比模式下自动覆盖）
    
    # 闭环模式噪声过滤阈值
    # density > noise_threshold 的样本被视为可能噪声，会被过滤
    noise_threshold = 0.7
    
    # 日志配置
    log_interval = 5                # 训练时日志打印间隔（每多少个epoch打印一次）
    save_checkpoints = True         # 是否保存检查点（模型权重等）
    
    # 是否记录每个类别的验证准确率（用于分析类别覆盖对性能的影响）
    record_per_class_accuracy = True


# ============================================================
# 派生参数（自动计算，不需要手动修改）
# ============================================================

# 特征维度映射
_feature_dim_map = {'small': 384, 'base': 768, 'large': 1024}
Config.feature_dim = _feature_dim_map.get(Config.model_size, 768)

# 注意: 以下参数在 use_percentage_sampling=True 时由 main.py 动态计算
# 这里提供默认值用于 use_percentage_sampling=False 的情况
if not Config.use_percentage_sampling:
    # 固定数值模式：使用手动设置的值
    Config.initial_labeled_count = Config.initial_samples_per_class * Config.num_classes  # 200
    Config.final_labeled_count = (
        Config.initial_labeled_count + 
        Config.budget_per_round * Config.total_rounds
    )  # 4200
    Config.final_labeled_ratio = Config.final_labeled_count / 101347 * 100  # ~4.14%
else:
    # 百分比模式：这些值将在 main.py 加载数据集后动态计算
    # 这里设置占位符，避免 AttributeError
    Config.initial_labeled_count = None  # 由 main.py 计算
    Config.final_labeled_count = None    # 由 main.py 计算
    Config.final_labeled_ratio = Config.final_data_percentage  # 目标百分比


# ============================================================
# 预设配置（快速切换实验设置）
# ============================================================

def set_random_sampling():
    """切换为 Random Sampling 基线实验"""
    Config.exp_name = 'exp_random_sampling'
    Config.random_sampling = True
    Config.use_prediction_uncertainty = False


def set_closed_loop_full():
    """切换为闭环完整版（4种不确定性）"""
    Config.exp_name = 'exp_closed_loop_full'
    Config.random_sampling = False
    Config.use_prediction_uncertainty = True
    Config.active_feature_uncertainties = ['exploration', 'boundary', 'density', 'multiscale']


def set_closed_loop_E_only():
    """切换为闭环消融：只用 Exploration"""
    Config.exp_name = 'exp_closed_loop_E'
    Config.random_sampling = False
    Config.use_prediction_uncertainty = True
    Config.active_feature_uncertainties = ['exploration']


def set_closed_loop_E_B():
    """切换为闭环消融：E + B"""
    Config.exp_name = 'exp_closed_loop_E_B'
    Config.random_sampling = False
    Config.use_prediction_uncertainty = True
    Config.active_feature_uncertainties = ['exploration', 'boundary']


def set_closed_loop_E_B_M():
    """切换为闭环消融：E + B + M"""
    Config.exp_name = 'exp_closed_loop_E_B_M'
    Config.random_sampling = False
    Config.use_prediction_uncertainty = True
    Config.active_feature_uncertainties = ['exploration', 'boundary', 'multiscale']


def set_open_loop():
    """切换为开环模式（纯特征不确定性）"""
    Config.exp_name = 'exp_open_loop'
    Config.random_sampling = False
    Config.use_prediction_uncertainty = False
    Config.active_feature_uncertainties = ['exploration', 'boundary', 'density', 'multiscale']


# ============================================================
# 打印配置信息
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("当前配置")
    print("=" * 60)
    
    print("\n【实验设置】")
    print(f"  实验名称: {Config.exp_name}")
    print(f"  随机采样: {Config.random_sampling}")
    print(f"  闭环模式: {Config.use_prediction_uncertainty}")
    print(f"  参与融合的不确定性: {Config.active_feature_uncertainties}")
    
    print("\n【1. feature_extractor.py】")
    print(f"  模型规模: {Config.model_size}")
    print(f"  特征维度: {Config.feature_dim}")
    print(f"  提取层: {Config.feature_layers}")
    
    print("\n【2. dataset.py】")
    print(f"  数据集路径: {Config.data_root}")
    print(f"  类别数: {Config.num_classes}")
    print(f"  批大小: {Config.batch_size}")
    if Config.use_percentage_sampling:
        print(f"  采样模式: 百分比模式")
        print(f"  最终数据百分比: {Config.final_data_percentage}%")
    else:
        print(f"  采样模式: 固定数值模式")
        print(f"  初始每类样本: {Config.initial_samples_per_class}")
    
    print("\n【3. memory_bank.py】")
    print(f"  设备: {Config.device}")
    
    print("\n【4. uncertainty.py】")
    print(f"  K近邻: {Config.k_neighbors}")
    print(f"  特征不确定性权重: {Config.feature_uncertainty_weights}")
    
    print("\n【5. prediction_uncertainty.py】")
    print(f"  预测不确定性模式: {Config.prediction_uncertainty_mode}")
    print(f"  融合策略: {Config.fusion_strategy}")
    
    print("\n【6. cascading_selector_v1.py】")
    print(f"  探索阈值: {Config.exploration_threshold}")
    print(f"  边界阈值: {Config.boundary_threshold}")
    print(f"  密度阈值: {Config.density_threshold}")
    print(f"  多尺度阈值: {Config.multiscale_threshold}")
    
    print("\n【7. classifier.py】")
    print(f"  分类器类型: {Config.classifier_type}")
    print(f"  学习率: {Config.classifier_lr}")
    print(f"  训练轮数: {Config.classifier_epochs}")
    
    print("\n【8. active_learner.py】")
    print(f"  总轮数: {Config.total_rounds}")
    print(f"  噪声阈值: {Config.noise_threshold}")
    
    print("\n【派生参数】")
    if Config.use_percentage_sampling:
        print(f"  采样模式: 百分比模式")
        print(f"  目标百分比: {Config.final_data_percentage}%")
        print(f"  总轮数: {Config.total_rounds} (初始+{Config.total_rounds}轮 = {Config.total_rounds + 1}份)")
        print(f"  注意: 具体样本数将在加载数据集后自动计算")
    else:
        print(f"  采样模式: 固定数值模式")
        print(f"  每轮预算: {Config.budget_per_round}")
        print(f"  初始标注: {Config.initial_labeled_count} 样本")
        print(f"  最终标注: {Config.final_labeled_count} 样本")
        print(f"  标注比例: {Config.final_labeled_ratio:.2f}%")
    
    print("\n" + "=" * 60)
