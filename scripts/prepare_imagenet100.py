"""
准备ImageNet-100数据集
- 将所有图片split成训练集和验证集（80/20）
- 创建标准的ImageFolder格式
"""

import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

def split_imagenet100(
    source_dir="/root/autodl-tmp/dinov3/data/imagenet100",
    train_ratio=0.8,
    seed=42
):
    """
    将ImageNet-100分割成train/val
    
    Args:
        source_dir: 原始数据目录（所有图片在一起）
        train_ratio: 训练集比例
        seed: 随机种子
    """
    random.seed(seed)
    
    source_path = Path(source_dir)
    base_dir = source_path.parent
    
    # 创建新的train/val目录
    train_dir = base_dir / "imagenet100_split" / "train"
    val_dir = base_dir / "imagenet100_split" / "val"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Source: {source_path}")
    print(f"Train: {train_dir}")
    print(f"Val: {val_dir}")
    print()
    
    # 获取所有类别
    classes = sorted([d for d in source_path.iterdir() if d.is_dir()])
    
    print(f"找到 {len(classes)} 个类别")
    print()
    
    total_train = 0
    total_val = 0
    
    for class_dir in tqdm(classes, desc="处理类别"):
        class_name = class_dir.name
        
        # 获取该类的所有图片
        images = sorted(list(class_dir.glob("*.JPEG")))
        
        if len(images) == 0:
            print(f"警告: {class_name} 没有图片，跳过")
            continue
        
        # 随机打乱
        random.shuffle(images)
        
        # 分割
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # 创建类别目录
        train_class_dir = train_dir / class_name
        val_class_dir = val_dir / class_name
        train_class_dir.mkdir(exist_ok=True)
        val_class_dir.mkdir(exist_ok=True)
        
        # 复制训练集图片
        for img in train_images:
            shutil.copy2(img, train_class_dir / img.name)
        
        # 复制验证集图片
        for img in val_images:
            shutil.copy2(img, val_class_dir / img.name)
        
        total_train += len(train_images)
        total_val += len(val_images)
    
    print()
    print("=" * 60)
    print("分割完成！")
    print("=" * 60)
    print(f"训练集: {total_train} 张图片")
    print(f"验证集: {total_val} 张图片")
    print(f"总计: {total_train + total_val} 张图片")
    print()
    print(f"训练集目录: {train_dir}")
    print(f"验证集目录: {val_dir}")


if __name__ == "__main__":
    split_imagenet100()

