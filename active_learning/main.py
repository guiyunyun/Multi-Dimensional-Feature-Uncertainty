"""
主动学习实验入口
运行方式: python main.py
所有配置在 config.py 中修改
"""

from dataset import load_imagenet100, create_initial_labeled_pool, create_random_initial_pool, ActiveLearningDataset
from active_learner import ActiveLearner
from cascading_selector_v1 import CascadingSelectorV1
from uncertainty import UncertaintyEstimator
from memory_bank import MemoryBank
from classifier import SimpleClassifier, MLPClassifier
from feature_extractor import MultiLayerDINOv3
from config import Config
import sys
from pathlib import Path
import logging
from datetime import datetime

import torch

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def setup_logger(save_dir: Path):
    """
    设置日志记录器，同时输出到终端和文件

    Args:
        save_dir: 保存目录（已包含时间戳）

    Returns:
        logger: 日志记录器
        log_file: 日志文件路径
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # 日志文件路径
    log_file = save_dir / 'log.txt'

    # 创建logger
    logger = logging.getLogger('ActiveLearning')
    logger.setLevel(logging.INFO)

    # 清除已有的handlers
    logger.handlers = []

    # 创建formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 文件handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 终端handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger, log_file


def log_config(logger):
    """记录当前配置"""
    logger.info("")
    logger.info("=" * 70)
    logger.info("Experiment Configuration")
    logger.info("=" * 70)

    logger.info(f"【Experiment Settings】")
    logger.info(f"  Experiment Name: {Config.exp_name}")
    logger.info(f"  Random Sampling: {Config.random_sampling}")
    logger.info(f"  Closed-Loop Mode: {Config.use_prediction_uncertainty}")
    logger.info(
        f"  Active Feature Uncertainties: {Config.active_feature_uncertainties}")

    logger.info(f"【Model Configuration】")
    logger.info(f"  DINOv3 Size: {Config.model_size} (feature dim: {Config.feature_dim})")
    logger.info(f"  Extraction Layers: {Config.feature_layers}")
    logger.info(f"  Classifier: {Config.classifier_type}")

    logger.info(f"【Data Configuration】")
    logger.info(f"  Dataset: {Config.data_root}")
    logger.info(f"  Number of Classes: {Config.num_classes}")
    logger.info(f"  Batch Size: {Config.batch_size}")

    logger.info(f"【Active Learning Configuration】")
    # 根据采样模式显示不同的信息
    if Config.use_percentage_sampling:
        logger.info(f"  Sampling Mode: Percentage-based")
        logger.info(f"  Target Percentage: {Config.final_data_percentage}%")
        logger.info(f"  Total Rounds: {Config.total_rounds}")
        logger.info(f"  (Detailed sample counts will be calculated after loading dataset)")
    else:
        # 固定数值模式
        use_random_initial = getattr(Config, 'use_random_initial_pool', False)
        if use_random_initial:
            initial_count = getattr(Config, 'initial_random_samples', 100)
            logger.info(f"  Initial Pool: {initial_count} random samples (no class balance)")
        else:
            logger.info(
                f"  Initial Pool: {Config.initial_samples_per_class} per class = {Config.initial_labeled_count} total")
        logger.info(f"  Budget per Round: {Config.budget_per_round}")
        logger.info(f"  Total Rounds: {Config.total_rounds}")
        # 计算最终标注数量
        if use_random_initial:
            initial_count = getattr(Config, 'initial_random_samples', 100)
            final_count = initial_count + Config.budget_per_round * Config.total_rounds
            final_ratio = final_count / 101347 * 100  # ImageNet-100训练集大小
            logger.info(f"  Final Labelled: {final_count} ({final_ratio:.2f}%)")
        else:
            logger.info(
                f"  Final Labelled: {Config.final_labeled_count} ({Config.final_labeled_ratio:.2f}%)")

    if Config.use_prediction_uncertainty:
        logger.info(f"【Prediction Uncertainty】")
        logger.info(f"  Mode: {Config.prediction_uncertainty_mode}")
        logger.info(f"  Fusion Strategy: {Config.fusion_strategy}")
        logger.info(f"  Noise Threshold: {Config.noise_threshold}")

    logger.info("=" * 70)
    logger.info("")


def main():
    """主函数"""
    # ========== 0. 准备工作 ==========
    # 生成带时间戳的保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    save_dir = Path(__file__).parent / Config.results_dir / \
        f"{Config.exp_name}_{timestamp}"

    # 设置日志
    logger, log_file = setup_logger(save_dir)

    logger.info("")
    logger.info("=" * 70)
    logger.info("Active Learning Experiment - DINOv3 + ImageNet-100")
    logger.info("=" * 70)
    logger.info(f"Result Save Directory: {save_dir}")
    logger.info("")

    # 记录配置
    log_config(logger)

    # 设置设备
    device = Config.device if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using Device: {device}")
    if device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info("")

    # 设置随机种子
    torch.manual_seed(Config.seed)

    # ========== 1. 加载数据集 ==========
    logger.info("1. Load Dataset...")

    import io
    from contextlib import redirect_stdout

    f = io.StringIO()
    with redirect_stdout(f):
        train_dataset, val_dataset = load_imagenet100(
            data_root=Config.data_root,
            batch_size=Config.batch_size,
            num_workers=Config.num_workers
        )
    output = f.getvalue()
    for line in output.strip().split('\n'):
        if line:
            logger.info(f"  {line}")

    # 创建验证集DataLoader
    from torch.utils.data import DataLoader
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True
    )
    
    # ========== 1.5 百分比采样：自动计算样本数 ==========
    if Config.use_percentage_sampling:
        dataset_size = len(train_dataset)
        
        # 计算最终总样本数
        total_budget = int(dataset_size * Config.final_data_percentage / 100)
        
        # 均分为 (total_rounds + 1) 份：初始池 1 份 + 每轮 1 份
        samples_per_round = total_budget // (Config.total_rounds + 1)
        
        # 更新 Config（供其他模块使用）
        Config.initial_samples = samples_per_round
        Config.budget_per_round = samples_per_round
        Config.initial_random_samples = samples_per_round  # 兼容随机初始池
        Config.use_random_initial_pool = True  # 百分比模式默认使用随机初始池
        
        # 更新派生参数
        Config.initial_labeled_count = samples_per_round
        Config.final_labeled_count = samples_per_round * (Config.total_rounds + 1)
        Config.final_labeled_ratio = Config.final_labeled_count / dataset_size * 100
        
        logger.info("")
        logger.info("  [Percentage Sampling Mode]")
        logger.info(f"  Dataset Size: {dataset_size}")
        logger.info(f"  Target Percentage: {Config.final_data_percentage}%")
        logger.info(f"  Total Budget: {total_budget} samples")
        logger.info(f"  Divided into: {Config.total_rounds + 1} parts (1 initial + {Config.total_rounds} rounds)")
        logger.info(f"  Samples per Part: {samples_per_round}")
        logger.info(f"  Actual Final Samples: {Config.final_labeled_count} ({Config.final_labeled_ratio:.2f}%)")
    
    logger.info("")

    # ========== 2. 创建初始labeled pool ==========
    logger.info("2. Create Initial Labelled Pool...")
    
    # 根据配置选择初始采样策略
    use_random_initial = getattr(Config, 'use_random_initial_pool', False)
    
    f = io.StringIO()
    with redirect_stdout(f):
        if use_random_initial:
            # 完全随机初始采样（不保证类别覆盖）
            logger.info("  Using Random Initial Pool (no class balance guarantee)")
            initial_indices = create_random_initial_pool(
                train_dataset,
                num_samples=getattr(Config, 'initial_random_samples', 100),
                seed=Config.seed
            )
        else:
            # 按类别均匀采样（分层采样）
            logger.info("  Using Stratified Initial Pool (class balanced)")
            initial_indices = create_initial_labeled_pool(
                train_dataset,
                num_samples_per_class=Config.initial_samples_per_class,
                num_classes=Config.num_classes,
                seed=Config.seed
            )
    output = f.getvalue()
    for line in output.strip().split('\n'):
        if line:
            logger.info(f"  {line}")

    f = io.StringIO()
    with redirect_stdout(f):
        al_dataset = ActiveLearningDataset(
            dataset=train_dataset,
            initial_labeled_indices=initial_indices,
            batch_size=Config.batch_size,
            num_workers=Config.num_workers
        )
    output = f.getvalue()
    for line in output.strip().split('\n'):
        if line:
            logger.info(f"  {line}")
    logger.info("")

    # ========== 3. 创建模型组件 ==========
    logger.info("3. Create Model Components...")

    # 特征提取器（冻结的DINOv3）
    logger.info("  - Load DINOv3 Feature Extractor...")
    feature_extractor = MultiLayerDINOv3(
        model_size=Config.model_size,
        pretrained=True,
        layers=Config.feature_layers
    )

    # 分类头（可训练）
    logger.info("  - Create Classifier Head...")
    if Config.classifier_type == 'mlp':
        classifier = MLPClassifier(
            input_dim=Config.feature_dim,
            hidden_dim=Config.classifier_hidden_dim,
            num_classes=Config.num_classes,
            dropout=Config.classifier_dropout
        )
    else:
        classifier = SimpleClassifier(
            input_dim=Config.feature_dim,
            num_classes=Config.num_classes,
            dropout=Config.classifier_dropout
        )
    logger.info(
        f"    Classifier Parameters: {sum(p.numel() for p in classifier.parameters()):,}")

    # Memory Bank
    logger.info("  - Create Memory Bank...")
    memory_bank = MemoryBank(
        feature_dim=Config.feature_dim,
        num_classes=Config.num_classes,
        device=device,
        layers=Config.feature_layers
    )

    # 不确定性估计器
    logger.info("  - Create Uncertainty Estimator...")
    uncertainty_estimator = UncertaintyEstimator(
        memory_bank=memory_bank,
        k_neighbors=Config.k_neighbors
    )

    # 样本选择器
    logger.info("  - Create Sample Selector...")
    selector = CascadingSelectorV1(
        density_threshold=Config.density_threshold,
        exploration_threshold=Config.exploration_threshold,
        boundary_threshold=Config.boundary_threshold,
        multiscale_threshold=Config.multiscale_threshold
    )
    logger.info("")

    # ========== 4. 创建主动学习器 ==========
    logger.info("4. Create Active Learner...")

    # 显示模式信息
    if Config.random_sampling:
        logger.info("  Mode: Random Sampling (Random Sampling Baseline)")
    elif Config.use_prediction_uncertainty:
        logger.info("  Mode: Closed-Loop (Feature + Prediction Uncertainty)")
        logger.info(f"  Fusion Strategy: {Config.fusion_strategy}")
        logger.info(
            f"  Active Feature Uncertainties: {Config.active_feature_uncertainties}")
    else:
        logger.info("  Mode: Open-Loop (Pure Feature Uncertainty)")

    active_learner = ActiveLearner(
        feature_extractor=feature_extractor,
        classifier=classifier,
        memory_bank=memory_bank,
        uncertainty_estimator=uncertainty_estimator,
        selector=selector,
        device=device,
        num_classes=Config.num_classes,
        use_prediction_uncertainty=Config.use_prediction_uncertainty,
        fusion_strategy=Config.fusion_strategy,
        random_sampling=Config.random_sampling
    )
    logger.info("")

    # ========== 4.5 初始化 Memory Bank（让 Round 1 也能用特征不确定性）==========
    # 如果不是随机采样模式，用初始标注样本的特征填充 Memory Bank
    if not Config.random_sampling:
        logger.info("4.5. Initialize Memory Bank with initial samples...")
        logger.info("  (This allows feature uncertainty to work in Round 1)")
        
        f = io.StringIO()
        with redirect_stdout(f):
            initial_loader = al_dataset.get_labeled_loader(shuffle=False)
            active_learner.update_memory_bank(initial_loader)
        output = f.getvalue()
        for line in output.strip().split('\n'):
            if line:
                logger.info(f"  {line}")
        
        logger.info(f"  Memory Bank initialized with {memory_bank.num_samples} samples")
        logger.info("")

    # ========== 5. 运行主动学习循环 ==========
    logger.info("5. Start Active Learning Loop...")
    logger.info(f"  Total Rounds: {Config.total_rounds}")
    logger.info(f"  Budget per Round: {Config.budget_per_round}")
    logger.info("")

    for round_num in range(1, Config.total_rounds + 1):
        # 运行一轮（捕获输出）
        f = io.StringIO()
        with redirect_stdout(f):
            metrics = active_learner.run_one_round(
                round_num=round_num,
                al_dataset=al_dataset,
                val_loader=val_loader,
                budget=Config.budget_per_round,
                train_epochs=Config.classifier_epochs,
                learning_rate=Config.classifier_lr
            )
        output = f.getvalue()
        for line in output.strip().split('\n'):
            if line:
                logger.info(line)

        # 打印总结
        logger.info("")
        logger.info(f"Round {round_num} Summary:")
        logger.info(f"  Validation Accuracy: {metrics['val_accuracy']:.2f}%")
        logger.info(f"  Labelled Samples: {metrics['labeled_samples']}")
        logger.info(f"  Unlabelled Samples: {metrics['unlabeled_samples']}")
        logger.info("")

    # ========== 6. 保存结果 ==========
    logger.info("")
    logger.info("=" * 70)
    logger.info("6. Save Results...")

    # 保存历史记录
    active_learner.save_history(save_dir / 'history.json')

    # 根据配置决定是否保存检查点
    if Config.save_checkpoints:
        active_learner.save_checkpoint(save_dir / 'checkpoint.pth')

    # 保存配置（便于复现）
    import json
    config_dict = {k: v for k, v in vars(
        Config).items() if not k.startswith('_')}
    # 过滤掉不能序列化的对象
    config_dict = {k: v for k, v in config_dict.items()
                   if isinstance(v, (int, float, str, bool, list, dict, type(None)))}
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)

    logger.info(f"  Results Saved to: {save_dir}")
    logger.info(f"  - {save_dir / 'history.json'}")
    logger.info(f"  - {save_dir / 'checkpoint.pth'}")
    logger.info(f"  - {save_dir / 'config.json'}")
    logger.info(f"  - {log_file}")
    logger.info("")

    # ========== 7. 最终评估 ==========
    logger.info("7. Final Evaluation...")
    from classifier import evaluate
    final_metrics = evaluate(
        active_learner.classifier,
        active_learner.feature_extractor,
        val_loader,
        device
    )

    logger.info("")
    logger.info("=" * 70)
    logger.info("Final Results:")
    logger.info(f"  Validation Accuracy: {final_metrics['accuracy']:.2f}%")
    logger.info(f"  Validation Loss: {final_metrics['loss']:.4f}")
    logger.info(f"  Total Labelled Samples: {len(al_dataset.labeled_indices)}")
    logger.info(
        f"  Labelled Ratio: {len(al_dataset.labeled_indices)/len(train_dataset)*100:.2f}%")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Experiment Completed!")


if __name__ == "__main__":
    main()
