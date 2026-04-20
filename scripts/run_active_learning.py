"""
运行完整的主动学习实验
"""

import sys
from pathlib import Path
import argparse
import logging
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from active_learning import (
    MultiLayerDINOv3,
    SimpleClassifier,
    MemoryBank,
    UncertaintyEstimator,
    CascadingSelectorV1,
    ActiveLearner,
    load_imagenet100,
    create_initial_labeled_pool,
    ActiveLearningDataset
)


def setup_logger(save_dir: Path, exp_name: str):
    """
    设置日志记录器，同时输出到终端和文件
    
    Args:
        save_dir: 保存目录
        exp_name: 实验名称
    
    Returns:
        logger: 日志记录器
    """
    # 创建保存目录
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建日志文件路径（带时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = save_dir / f"log_{timestamp}.txt"
    
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
    
    logger.info("=" * 70)
    logger.info(f"日志文件已创建: {log_file}")
    logger.info("=" * 70)
    
    return logger, log_file


def main(args):
    """主函数"""
    # ========== 0. 设置日志 ==========
    save_dir = project_root / 'results' / args.exp_name
    logger, log_file = setup_logger(save_dir, args.exp_name)
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("主动学习实验 - DINOv3 + ImageNet-100")
    logger.info("=" * 70)
    logger.info("")
    
    # 记录所有参数
    logger.info("实验参数:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    logger.info("")
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用设备: {device}")
    logger.info("")
    
    # ========== 1. 加载数据集 ==========
    logger.info("1. 加载数据集...")
    
    # 临时捕获load_imagenet100的输出
    import io
    from contextlib import redirect_stdout
    
    f = io.StringIO()
    with redirect_stdout(f):
        train_dataset, val_dataset = load_imagenet100(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
    output = f.getvalue()
    for line in output.strip().split('\n'):
        if line:
            logger.info(f"  {line}")
    
    # 创建验证集DataLoader
    from torch.utils.data import DataLoader
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    logger.info("")
    
    # ========== 2. 创建初始labeled pool ==========
    logger.info("2. 创建初始labeled pool...")
    
    f = io.StringIO()
    with redirect_stdout(f):
        initial_indices = create_initial_labeled_pool(
            train_dataset,
            num_samples_per_class=args.init_samples_per_class,
            num_classes=100,
            seed=args.seed
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
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
    output = f.getvalue()
    for line in output.strip().split('\n'):
        if line:
            logger.info(f"  {line}")
    logger.info("")
    
    # ========== 3. 创建模型组件 ==========
    logger.info("3. 创建模型组件...")
    
    # 特征提取器（冻结的DINOv3）
    logger.info("  - 加载DINOv3特征提取器...")
    feature_extractor = MultiLayerDINOv3(
        model_size=args.model_size,
        pretrained=True,
        layers=args.layers
    )
    
    # 分类头（可训练）
    logger.info("  - 创建分类头...")
    classifier = SimpleClassifier(
        input_dim=768 if args.model_size == 'base' else 1024,
        num_classes=100,
        dropout=args.dropout
    )
    
    # Memory Bank
    logger.info("  - 创建Memory Bank...")
    memory_bank = MemoryBank(
        feature_dim=768 if args.model_size == 'base' else 1024,
        num_classes=100,
        device=device,
        layers=args.layers
    )
    
    # 不确定性估计器
    logger.info("  - 创建不确定性估计器...")
    uncertainty_estimator = UncertaintyEstimator(
        memory_bank=memory_bank,
        k_neighbors=args.k_neighbors
    )
    
    # 样本选择器
    logger.info("  - 创建样本选择器...")
    selector = CascadingSelectorV1(
        density_threshold=args.density_threshold,
        exploration_threshold=args.exploration_threshold,
        boundary_threshold=args.boundary_threshold,
        multiscale_threshold=args.multiscale_threshold
    )
    logger.info("")
    
    # ========== 4. 创建主动学习器 ==========
    logger.info("4. 创建主动学习器...")
    
    # 显示模式信息
    if args.use_prediction_uncertainty:
        logger.info("  模式: 闭环（特征 + 预测不确定性）")
        logger.info(f"  融合策略: {args.fusion_strategy}")
    else:
        logger.info("  模式: 开环（纯特征不确定性）")
    
    active_learner = ActiveLearner(
        feature_extractor=feature_extractor,
        classifier=classifier,
        memory_bank=memory_bank,
        uncertainty_estimator=uncertainty_estimator,
        selector=selector,
        device=device,
        num_classes=100,
        use_prediction_uncertainty=args.use_prediction_uncertainty,
        fusion_strategy=args.fusion_strategy
    )
    logger.info("")
    
    # ========== 5. 运行主动学习循环 ==========
    logger.info("5. 开始主动学习循环...")
    logger.info(f"  总轮数: {args.num_rounds}")
    logger.info(f"  每轮预算: {args.budget_per_round}")
    logger.info("")
    
    for round_num in range(1, args.num_rounds + 1):
        # 捕获run_one_round的输出
        f = io.StringIO()
        with redirect_stdout(f):
            metrics = active_learner.run_one_round(
                round_num=round_num,
                al_dataset=al_dataset,
                val_loader=val_loader,
                budget=args.budget_per_round,
                train_epochs=args.train_epochs,
                learning_rate=args.learning_rate
            )
        output = f.getvalue()
        for line in output.strip().split('\n'):
            if line:
                logger.info(line)
        
        # 打印总结
        logger.info("")
        logger.info(f"Round {round_num} 总结:")
        logger.info(f"  验证准确率: {metrics['val_accuracy']:.2f}%")
        logger.info(f"  已标注样本: {metrics['labeled_samples']}")
        logger.info(f"  未标注样本: {metrics['unlabeled_samples']}")
        logger.info("")
    
    # ========== 6. 保存结果 ==========
    logger.info("")
    logger.info("=" * 70)
    logger.info("6. 保存结果...")
    
    # 保存历史记录
    active_learner.save_history(save_dir / 'history.json')
    active_learner.save_checkpoint(save_dir / 'checkpoint.pth')
    
    logger.info(f"✓ 结果已保存到: {save_dir}")
    logger.info(f"  - {save_dir / 'history.json'}")
    logger.info(f"  - {save_dir / 'checkpoint.pth'}")
    logger.info(f"  - {log_file}")
    logger.info("")
    
    # ========== 7. 最终评估 ==========
    logger.info("7. 最终评估...")
    from active_learning.classifier import evaluate
    final_metrics = evaluate(
        active_learner.classifier,
        active_learner.feature_extractor,
        val_loader,
        device
    )
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("最终结果:")
    logger.info(f"  验证准确率: {final_metrics['accuracy']:.2f}%")
    logger.info(f"  验证损失: {final_metrics['loss']:.4f}")
    logger.info(f"  总标注样本: {len(al_dataset.labeled_indices)}")
    logger.info(f"  标注比例: {len(al_dataset.labeled_indices)/len(train_dataset)*100:.2f}%")
    logger.info("=" * 70)
    logger.info("")
    logger.info("✓ 实验完成!")
    logger.info("")
    logger.info(f"📊 查看结果: cat {save_dir / 'history.json'}")
    logger.info(f"📄 查看日志: cat {log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='主动学习实验')
    
    # 数据参数
    parser.add_argument('--data_root', type=str, 
                        default='/root/autodl-tmp/dinov3/data/imagenet100_split',
                        help='数据集根目录')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批大小')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    
    # 模型参数
    parser.add_argument('--model_size', type=str, default='base',
                        choices=['small', 'base', 'large'],
                        help='DINOv3模型大小')
    parser.add_argument('--layers', type=int, nargs='+', default=[3, 6, 9, 11],
                        help='提取特征的层')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='分类头的dropout')
    
    # 主动学习参数
    parser.add_argument('--num_rounds', type=int, default=10,
                        help='主动学习轮数')
    parser.add_argument('--budget_per_round', type=int, default=100,
                        help='每轮选择的样本数')
    parser.add_argument('--init_samples_per_class', type=int, default=2,
                        help='初始每类样本数')
    parser.add_argument('--k_neighbors', type=int, default=10,
                        help='KNN的K值')
    
    # 不确定性阈值
    parser.add_argument('--density_threshold', type=float, default=0.5,
                        help='密度不确定性阈值')
    parser.add_argument('--exploration_threshold', type=float, default=0.5,
                        help='探索性不确定性阈值')
    parser.add_argument('--boundary_threshold', type=float, default=0.6,
                        help='边界不确定性阈值')
    parser.add_argument('--multiscale_threshold', type=float, default=0.5,
                        help='多尺度不确定性阈值')
    
    # 训练参数
    parser.add_argument('--train_epochs', type=int, default=10,
                        help='每轮训练的epoch数')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='学习率')
    
    # 预测不确定性（闭环）参数
    parser.add_argument('--use_prediction_uncertainty', type=lambda x: x.lower() == 'true',
                        default=False,
                        help='是否使用预测不确定性（闭环模式）')
    parser.add_argument('--fusion_strategy', type=str, default='attention',
                        choices=['attention', 'multiply', 'add'],
                        help='不确定性融合策略')
    
    # 其他
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--exp_name', type=str, default='exp_default',
                        help='实验名称')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    main(args)

