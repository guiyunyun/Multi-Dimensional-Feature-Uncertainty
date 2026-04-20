#!/usr/bin/env python3
"""
通过 PyTorch Hub 下载 DINOv3 权重
PyTorch Hub 会自动下载并缓存模型权重
"""
import torch
import os
import shutil
from pathlib import Path

# 创建权重目录
weights_dir = Path("/root/dinov3/weights")
weights_dir.mkdir(exist_ok=True)

# PyTorch Hub 缓存目录
hub_cache = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"

# 要下载的模型列表
models = [
    ("dinov3_vits16", "ViT-S/16"),
    ("dinov3_vitb16", "ViT-B/16"),
    ("dinov3_vitl16", "ViT-L/16 ⭐主力模型"),
]

print("=" * 60)
print("通过 PyTorch Hub 下载 DINOv3 权重")
print("=" * 60)

for model_name, desc in models:
    print(f"\n[{models.index((model_name, desc)) + 1}/{len(models)}] 正在下载 {desc}...")
    print(f"模型名: {model_name}")
    
    try:
        # 加载模型 (会自动下载权重)
        model = torch.hub.load(
            'facebookresearch/dinov3',
            model_name,
            pretrained=True,
            force_reload=False,  # 使用缓存
            skip_validation=True
        )
        
        print(f"✅ {desc} 下载成功!")
        
        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   参数量: {total_params / 1e6:.1f}M")
        
    except Exception as e:
        print(f"❌ {desc} 下载失败: {e}")

print("\n" + "=" * 60)
print("权重文件已缓存到:")
print(f"  {hub_cache}")
print("\n如需手动访问权重文件，请查看上述目录")
print("=" * 60)

# 列出已下载的权重
if hub_cache.exists():
    print("\n已下载的权重文件:")
    for f in sorted(hub_cache.glob("*.pth")):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name} ({size_mb:.1f} MB)")

