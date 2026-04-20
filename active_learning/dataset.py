"""
Dataset and DataLoader for Active Learning
- ImageNet-100数据加载
- 主动学习的labeled/unlabeled pool管理
"""

from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np


def _get_config():
    """获取 Config 对象"""
    try:
        from .config import Config
    except ImportError:
        from config import Config
    return Config


class ActiveLearningDataset:
    """
    主动学习数据集管理器
    
    功能:
    - 管理labeled pool和unlabeled pool
    - 支持样本迁移（unlabeled → labeled）
    - 提供DataLoader
    """
    
    def __init__(
        self,
        dataset: Dataset,
        initial_labeled_indices: Optional[List[int]] = None,
        batch_size: int = None,
        num_workers: int = None
    ):
        """
        Args:
            dataset: 完整的数据集
            initial_labeled_indices: 初始已标注样本的索引
            batch_size: 批大小
            num_workers: 数据加载线程数
        """
        Config = _get_config()
        
        # 从 Config 读取默认值
        if batch_size is None:
            batch_size = getattr(Config, 'batch_size', 32)
        if num_workers is None:
            num_workers = getattr(Config, 'num_workers', 4)
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # 初始化labeled和unlabeled索引
        all_indices = list(range(len(dataset)))
        
        if initial_labeled_indices is None:
            # 如果没有提供，默认所有样本都是unlabeled
            self.labeled_indices = []
            self.unlabeled_indices = all_indices
        else:
            self.labeled_indices = initial_labeled_indices
            self.unlabeled_indices = [i for i in all_indices if i not in initial_labeled_indices]
        
        print(f"  ActiveLearningDataset initialized")
        print(f"  Total samples: {len(dataset)}")
        print(f"  Labeled samples: {len(self.labeled_indices)}")
        print(f"  Unlabeled samples: {len(self.unlabeled_indices)}")
    
    def add_labeled_samples(self, indices: List[int]):
        """
        添加新的已标注样本
        
        Args:
            indices: 要标注的样本索引（在unlabeled_indices中的相对索引）
        """
        # 将相对索引转换为绝对索引
        absolute_indices = [self.unlabeled_indices[i] for i in indices]
        
        # 移动到labeled pool
        self.labeled_indices.extend(absolute_indices)
        
        # 从unlabeled pool中移除
        self.unlabeled_indices = [
            idx for idx in self.unlabeled_indices 
            if idx not in absolute_indices
        ]
        
        print(f"  Added {len(indices)} samples to labeled pool")
        print(f"  - Labeled samples: {len(self.labeled_indices)}")
        print(f"  - Unlabeled samples: {len(self.unlabeled_indices)}")
    
    # shuffle: bool = True 表示是否打乱数据集，训练时ture防止模型记住样本顺序，推理验证时false方便追踪结果
    def get_labeled_loader(self, shuffle: bool = True) -> DataLoader:
        """获取labeled pool的DataLoader"""
        if len(self.labeled_indices) == 0:
            return None
        
        # Subset: 从数据集中选择指定索引的子集
        # pin_memory: 是否将数据加载到GPU内存中
        labeled_dataset = Subset(self.dataset, self.labeled_indices)
        return DataLoader(
            labeled_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_unlabeled_loader(self, shuffle: bool = False) -> DataLoader:
        """获取unlabeled pool的DataLoader"""
        if len(self.unlabeled_indices) == 0:
            return None
        
        unlabeled_dataset = Subset(self.dataset, self.unlabeled_indices)
        return DataLoader(
            unlabeled_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_statistics(self) -> dict:
        """获取统计信息"""
        return {
            'total_samples': len(self.dataset),
            'labeled_samples': len(self.labeled_indices),
            'unlabeled_samples': len(self.unlabeled_indices),
            'labeled_ratio': len(self.labeled_indices) / len(self.dataset) * 100
        }


def get_imagenet100_transforms(split: str = 'train'):
    """
    获取ImageNet-100的数据增强
    
    Args:
        split: 'train' or 'val'
    
    Returns:
        transform: torchvision transforms
    """
    Config = _get_config()
    
    # 从 Config 读取参数
    image_size = getattr(Config, 'image_size', 224)
    normalize_mean = getattr(Config, 'normalize_mean', [0.485, 0.456, 0.406])
    normalize_std = getattr(Config, 'normalize_std', [0.229, 0.224, 0.225])
    
    if split == 'train':
        # 训练时的数据增强
        transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std)
        ])
    else:
        # 验证/测试时
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std)
        ])
    
    return transform


def load_imagenet100(
    data_root: str = None,
    batch_size: int = None,
    num_workers: int = None
) -> Tuple[Dataset, Dataset]:
    """
    加载ImageNet-100数据集
    
    Args:
        data_root: 数据集根目录
        batch_size: 批大小
        num_workers: 数据加载线程数
    
    Returns:
        train_dataset: 训练集
        val_dataset: 验证集
    """
    Config = _get_config()
    
    # 从 Config 读取默认值
    if data_root is None:
        data_root = getattr(Config, 'data_root', '/root/autodl-tmp/dinov3/data/ImageNet100')
    if batch_size is None:
        batch_size = getattr(Config, 'batch_size', 32)
    if num_workers is None:
        num_workers = getattr(Config, 'num_workers', 4)
    
    data_root = Path(data_root)
    
    # 检查数据集是否存在
    train_dir = data_root / 'train'
    val_dir = data_root / 'val'
    
    if not train_dir.exists():
        raise FileNotFoundError(f"train directory does not exist: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"validation directory does not exist: {val_dir}")
    
    # 加载训练集
    train_transform = get_imagenet100_transforms('train')
    train_dataset = ImageFolder(train_dir, transform=train_transform)
    
    # 加载验证集
    val_transform = get_imagenet100_transforms('val')
    val_dataset = ImageFolder(val_dir, transform=val_transform)
    
    print(f"  ImageNet-100 dataset loaded")
    print(f"  - Train dataset: {len(train_dataset)} samples")
    print(f"  - Validation dataset: {len(val_dataset)} samples")
    print(f"  - Number of classes: {len(train_dataset.classes)}")
    
    return train_dataset, val_dataset


def create_initial_labeled_pool(
    dataset: Dataset,
    num_samples_per_class: int = None,
    num_classes: int = None,
    seed: int = None
) -> List[int]:
    """
    创建初始的labeled pool（每个类别随机选择若干样本）
    
    Args:
        dataset: 数据集
        num_samples_per_class: 每个类别选择的样本数
        num_classes: 类别数
        seed: 随机种子
    
    Returns:
        initial_indices: 初始已标注样本的索引
    """
    Config = _get_config()
    
    # 从 Config 读取默认值
    if num_samples_per_class is None:
        num_samples_per_class = getattr(Config, 'initial_samples_per_class', 2)
    if num_classes is None:
        num_classes = getattr(Config, 'num_classes', 100)
    if seed is None:
        seed = getattr(Config, 'seed', 42)
    
    np.random.seed(seed)
    
    # 获取每个类别的样本索引
    # 优化：直接从ImageFolder的targets获取标签，不触发图片加载
    print("  Analyzing class distribution...")
    if hasattr(dataset, 'targets'):
        # ImageFolder有targets属性，直接使用
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'targets'):
        # 可能是Subset，获取底层dataset的targets
        labels = np.array(dataset.dataset.targets)
    else:
        # 兜底方案：遍历（会很慢）
        print("  Unable to get labels directly, using slow method...")
        from tqdm import tqdm
        labels = []
        for idx in tqdm(range(len(dataset)), desc="  读取标签"):
            _, label = dataset[idx]
            labels.append(label)
        labels = np.array(labels)
    
    class_to_indices = {i: [] for i in range(num_classes)}
    for idx, label in enumerate(labels):
        class_to_indices[int(label)].append(idx)
    
    # 从每个类别中随机选择样本
    print("  Selecting initial samples...")
    initial_indices = []
    for class_id in range(num_classes):
        indices = class_to_indices[class_id]
        if len(indices) >= num_samples_per_class:
            selected = np.random.choice(
                indices, 
                size=num_samples_per_class, 
                replace=False
            )
            initial_indices.extend(selected.tolist())
    
    print(f"  Initial labeled pool created")
    print(f"  - Number of samples per class: {num_samples_per_class}")
    print(f"  - Total samples: {len(initial_indices)}")
    
    return initial_indices


def create_random_initial_pool(
    dataset: Dataset,
    num_samples: int = None,
    seed: int = None
) -> List[int]:
    """
    创建完全随机的初始labeled pool（不保证类别覆盖）
    
    这更接近真实的主动学习场景：开始时没有任何标签信息，
    随机选择一些样本进行标注。
    
    Args:
        dataset: 数据集
        num_samples: 要选择的样本总数
        seed: 随机种子
    
    Returns:
        initial_indices: 初始已标注样本的索引
    """
    Config = _get_config()
    
    # 从 Config 读取默认值
    if num_samples is None:
        num_samples = getattr(Config, 'initial_random_samples', 100)
    if seed is None:
        seed = getattr(Config, 'seed', 42)
    
    np.random.seed(seed)
    
    # 完全随机选择
    total_samples = len(dataset)
    initial_indices = np.random.choice(
        total_samples,
        size=min(num_samples, total_samples),
        replace=False
    ).tolist()
    
    # 统计类别覆盖情况
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
        selected_labels = labels[initial_indices]
        num_classes_covered = len(set(selected_labels))
        
        print(f"  Random initial labeled pool created")
        print(f"  - Total samples: {len(initial_indices)}")
        print(f"  - Classes covered: {num_classes_covered}/{len(set(labels))}")
        
        # 如果覆盖率低，打印警告
        if num_classes_covered < len(set(labels)) * 0.8:
            print(f"  Warning: Low class coverage! Some classes may not be represented.")
    else:
        print(f"  Random initial labeled pool created")
        print(f"  - Total samples: {len(initial_indices)}")
    
    return initial_indices


def test_dataset():
    """测试数据加载器"""
    Config = _get_config()
    
    print("=" * 60)
    print("测试数据加载器")
    print("=" * 60)
    print()
    
    print(f"配置来源: config.py")
    print(f"  data_root: {Config.data_root}")
    print(f"  batch_size: {Config.batch_size}")
    print(f"  num_classes: {Config.num_classes}")
    print(f"  initial_samples_per_class: {Config.initial_samples_per_class}")
    print()
    
    try:
        # 1. 加载数据集（参数从 Config 读取）
        print("1. 加载ImageNet-100:")
        train_dataset, val_dataset = load_imagenet100()
        print()
        
        # 2. 创建初始labeled pool（参数从 Config 读取）
        print("2. 创建初始labeled pool:")
        initial_indices = create_initial_labeled_pool(train_dataset)
        print()
        
        # 3. 创建ActiveLearningDataset（参数从 Config 读取）
        print("3. 创建ActiveLearningDataset:")
        al_dataset = ActiveLearningDataset(
            dataset=train_dataset,
            initial_labeled_indices=initial_indices
        )
        print()
        
        # 4. 获取DataLoader
        print("4. 测试DataLoader:")
        labeled_loader = al_dataset.get_labeled_loader()
        unlabeled_loader = al_dataset.get_unlabeled_loader()
        
        # 测试labeled loader
        if labeled_loader:
            images, labels = next(iter(labeled_loader))
            print(f"  Labeled batch:")
            print(f"    images shape: {images.shape}")
            print(f"    labels shape: {labels.shape}")
            print(f"    labels range: [{labels.min()}, {labels.max()}]")
        
        # 测试unlabeled loader
        if unlabeled_loader:
            images, labels = next(iter(unlabeled_loader))
            print(f"  Unlabeled batch:")
            print(f"    images shape: {images.shape}")
            print(f"    labels shape: {labels.shape} (用于模拟标注)")
        print()
        
        # 5. 测试添加样本
        print("5. 测试添加样本:")
        print(f"  添加前: labeled={len(al_dataset.labeled_indices)}, unlabeled={len(al_dataset.unlabeled_indices)}")
        al_dataset.add_labeled_samples([0, 1, 2, 3, 4])
        print(f"  添加后: labeled={len(al_dataset.labeled_indices)}, unlabeled={len(al_dataset.unlabeled_indices)}")
        print()
        
        # 6. 统计信息
        print("6. 统计信息:")
        stats = al_dataset.get_statistics()
        for key, value in stats.items():
            if 'ratio' in key:
                print(f"  {key}: {value:.2f}%")
            else:
                print(f"  {key}: {value}")
        
    except FileNotFoundError as e:
        print(f"⚠️ 数据集未找到: {e}")
        print(f"  请确保已经运行了 scripts/prepare_imagenet100.py")
        print(f"  或手动下载ImageNet-100数据集")
    
    print()
    print("=" * 60)
    print("✓ 测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    test_dataset()

