#!/usr/bin/env python3
"""
DINOv3 基础功能测试脚本
用于验证环境配置和模型加载是否正常

Week 1, Day 1-2 任务
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from pathlib import Path

def test_environment():
    """测试基础环境"""
    print("=" * 60)
    print("1. 测试基础环境")
    print("=" * 60)
    
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"GPU数量: {torch.cuda.device_count()}")
    print()

def test_model_loading():
    """测试模型加载"""
    print("=" * 60)
    print("2. 测试DINOv3模型加载")
    print("=" * 60)
    
    # 检查预训练模型文件
    model_dir = Path(__file__).parent.parent / "pretrained_models"
    
    models = {
        'vits16': 'dinov3_vits16_pretrain_lvd1689m-08c60483.pth',
        'vitb16': 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
        'vitl16': 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
    }
    
    print(f"预训练模型目录: {model_dir}")
    for name, filename in models.items():
        model_path = model_dir / filename
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ {name:8s}: {filename} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {name:8s}: 未找到 {filename}")
    print()
    
    # 尝试加载ViT-B/16模型（最平衡的选择）
    print("尝试加载 ViT-B/16 模型...")
    try:
        from dinov3.models import vision_transformer as vit
        
        # 创建模型
        model = vit.vit_base(
            img_size=224,
            patch_size=16,
            init_values=1.0,
            block_chunks=0,
            num_register_tokens=0  # 先不用register tokens
        )
        
        print(f"  ✓ 模型创建成功")
        print(f"  - 参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        
        # 加载预训练权重
        vitb_path = model_dir / models['vitb16']
        if vitb_path.exists():
            state_dict = torch.load(vitb_path, map_location='cpu')
            
            # 清理state_dict的key（可能包含'module.'前缀）
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k.replace('module.', '')
                new_state_dict[new_k] = v
            
            # 只加载匹配的权重
            model_dict = model.state_dict()
            filtered_dict = {k: v for k, v in new_state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            
            model.load_state_dict(filtered_dict, strict=False)
            print(f"  ✓ 预训练权重加载成功 ({len(filtered_dict)}/{len(model_dict)} layers)")
        
        model.eval()
        
        return model
    
    except Exception as e:
        print(f"  ✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_feature_extraction(model):
    """测试特征提取"""
    print("=" * 60)
    print("3. 测试特征提取")
    print("=" * 60)
    
    if model is None:
        print("跳过（模型未加载）")
        return
    
    try:
        # 创建测试输入
        batch_size = 2
        image = torch.randn(batch_size, 3, 224, 224)
        print(f"输入shape: {image.shape}")
        
        # 如果有GPU，移到GPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        image = image.to(device)
        
        # 前向传播
        with torch.no_grad():
            output = model(image)
        
        print(f"输出shape: {output.shape}")
        
        # 解析输出
        if len(output.shape) == 3:  # [B, N, D]
            batch, num_tokens, dim = output.shape
            
            print(f"\n  特征详情:")
            print(f"  - Batch size: {batch}")
            print(f"  - Token数量: {num_tokens}")
            print(f"  - 特征维度: {dim}")
            
            # CLS token
            cls_token = output[:, 0]
            print(f"\n  CLS token shape: {cls_token.shape}  # 全局特征")
            
            # Patch tokens
            if num_tokens > 1:
                patch_tokens = output[:, 1:]
                num_patches = patch_tokens.shape[1]
                print(f"  Patch tokens shape: {patch_tokens.shape}  # 密集特征")
                print(f"  - Patch数量: {num_patches}")
                print(f"  - 预期: 14x14 = 196 patches (对于224x224图像)")
                
                # 验证patch数量
                expected_patches = (224 // 16) ** 2  # 224/16 = 14, 14*14 = 196
                if num_patches == expected_patches:
                    print(f"  ✓ Patch数量正确!")
                else:
                    print(f"  ⚠ Patch数量不匹配 (实际{num_patches}, 预期{expected_patches})")
        
        print(f"\n  ✓ 特征提取成功!")
        print(f"  - 设备: {device}")
        print(f"  - 数据类型: {output.dtype}")
        print(f"  - 特征范围: [{output.min():.3f}, {output.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"  ✗ 特征提取失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_processing(model):
    """测试批量处理"""
    print("=" * 60)
    print("4. 测试批量处理性能")
    print("=" * 60)
    
    if model is None or not torch.cuda.is_available():
        print("跳过（需要GPU）")
        return
    
    try:
        import time
        
        device = 'cuda'
        model = model.to(device)
        
        batch_sizes = [1, 4, 8, 16, 32]
        
        print(f"{'Batch Size':>12} | {'Time (ms)':>12} | {'Throughput (img/s)':>20}")
        print("-" * 50)
        
        for bs in batch_sizes:
            # 预热
            dummy_input = torch.randn(bs, 3, 224, 224, device=device)
            with torch.no_grad():
                _ = model(dummy_input)
            
            # 测试
            torch.cuda.synchronize()
            start = time.time()
            
            with torch.no_grad():
                _ = model(dummy_input)
            
            torch.cuda.synchronize()
            elapsed = (time.time() - start) * 1000  # ms
            throughput = bs / (elapsed / 1000)  # images/sec
            
            print(f"{bs:>12} | {elapsed:>12.2f} | {throughput:>20.2f}")
        
        print("\n  ✓ 批量处理测试完成!")
        
    except Exception as e:
        print(f"  ✗ 批量处理测试失败: {e}")

def main():
    """主函数"""
    print("\n")
    print("*" * 60)
    print("*" + " " * 58 + "*")
    print("*" + "  DINOv3 环境验证脚本".center(56) + "  *")
    print("*" + " " * 58 + "*")
    print("*" * 60)
    print()
    
    # 测试1: 环境
    test_environment()
    
    # 测试2: 模型加载
    model = test_model_loading()
    
    # 测试3: 特征提取
    success = test_feature_extraction(model)
    
    # 测试4: 批量处理（可选）
    if success and torch.cuda.is_available():
        test_batch_processing(model)
    
    # 总结
    print()
    print("=" * 60)
    print("测试总结")
    print("=" * 60)
    
    if model is not None and success:
        print("✓ 所有测试通过!")
        print("✓ DINOv3环境配置正确，可以开始研究工作")
        print()
        print("下一步:")
        print("  1. 运行 scripts/prepare_cifar100.py 准备数据集")
        print("  2. 运行 scripts/extract_features.py 提取特征")
        print("  3. 查看 RESEARCH_ROADMAP.md 了解详细计划")
    else:
        print("✗ 部分测试失败")
        print("请检查:")
        print("  1. 预训练模型是否正确下载")
        print("  2. dinov3包是否正确安装")
        print("  3. CUDA环境是否配置")
    
    print()

if __name__ == "__main__":
    main()
