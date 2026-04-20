"""
多层DINOv3特征提取器
- 提取ViT不同Transformer层的特征
- 用于计算多尺度不确定性
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path


class MultiLayerDINOv3(nn.Module):
    """
    多层DINOv3特征提取器
    
    提取ViT不同层的特征用于多尺度不确定性计算
    """
    
    def __init__(self, model_size=None, pretrained=True, layers=None):
        """
        Args:
            model_size: 模型大小
                - 'small': ViT-S (384 dim)
                - 'base': ViT-B (768 dim)
                - 'large': ViT-L (1024 dim)
            pretrained: 是否加载预训练权重
            layers: 要提取的层索引，例如 [3, 6, 9, 11]
                    如果为None，自动根据模型深度选择
        """
        super().__init__()
        
        # 添加项目路径到sys.path
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        # 从 Config 读取默认值
        try:
            from .config import Config
        except ImportError:
            from config import Config
        
        # 如果参数为 None，从 Config 读取
        if model_size is None:
            model_size = getattr(Config, 'model_size', 'base')
        if layers is None:
            layers = getattr(Config, 'feature_layers', [3, 6, 9, 11])
        
        # 获取权重目录
        weights_dir = getattr(Config, 'pretrained_weights_dir', 'pretrained_models')
        
        # 导入DINOv3官方加载函数
        from dinov3.hub import backbones
        
        # 模型配置
        model_configs = {
            'small': {
                'func': backbones.dinov3_vits16,
                'embed_dim': 384,
                'weight_file': 'dinov3_vits16_pretrain_lvd1689m-08c60483.pth'
            },
            'base': {
                'func': backbones.dinov3_vitb16,
                'embed_dim': 768,
                'weight_file': 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'
            },
            'large': {
                'func': backbones.dinov3_vitl16,
                'embed_dim': 1024,
                'weight_file': 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
            }
        }
        
        if model_size not in model_configs:
            raise ValueError(f"model_size must be one of {list(model_configs.keys())}")
        
        config = model_configs[model_size]
        
        # 加载模型（使用官方函数）
        print(f"Loading DINOv3 ViT-{model_size.upper()}/16...")
        
        if pretrained:
            # 使用本地权重路径
            weight_path = project_root / weights_dir / config['weight_file']
            if not weight_path.exists():
                raise FileNotFoundError(f"Weight file not found: {weight_path}")
            
            print(f"Loading weight: {weight_path.name}")
            # 传入本地权重路径（config['func'] 是 backbones.dinov3_vitb16 函数，附带传参）
            self.backbone = config['func'](pretrained=True, weights=str(weight_path))
            print("Weight loaded successfully")
        else:
            self.backbone = config['func'](pretrained=False)
        
        self.backbone.eval()
        
        # 冻结backbone参数（torch.Tensor类方法，选择是否计算梯度）
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # 获取模型配置
        self.model_size = model_size
        self.embed_dim = config['embed_dim']
        self.num_blocks = len(self.backbone.blocks)
        
        # 设置要提取的层
        if layers is None:
            # 自动选择4层：早期、中前期、中后期、最后一层
            self.layers = [
                self.num_blocks // 4,           # ~25% 深度
                self.num_blocks // 2,           # ~50% 深度
                3 * self.num_blocks // 4,       # ~75% 深度
                self.num_blocks - 1             # 最后一层
            ]
        else:
            self.layers = layers
        
        print(f"Model loaded successfully: DINOv3 ViT-{self.model_size.upper()}/16")
        print(f"  - Embed dimension: {self.embed_dim}")
        print(f"  - Transformer layers: {self.num_blocks}")
        print(f"  - Extracted layers: {self.layers}")
        print(f"  - Parameters frozen: True")
        print()
    
    # 修饰器：torch.no_grad()，禁用计算梯度
    @torch.no_grad()
    def forward(self, images, return_cls_only=False, pool_patches=False):
        """
        提取多层特征
        
        Args:
            images: [B, 3, H, W] 输入图像
            return_cls_only: 是否只返回CLS token（用于分类）
            pool_patches: 是否对patch特征进行平均池化（节省内存）
        
        Returns:
            如果 return_cls_only=True:
                cls_token: [B, embed_dim] CLS token特征
            否则:
                features_dict: {
                    'layer_3': [B, embed_dim] if pool_patches else [B, num_patches, embed_dim],
                    'layer_6': [B, embed_dim] if pool_patches else [B, num_patches, embed_dim],
                    'layer_9': [B, embed_dim] if pool_patches else [B, num_patches, embed_dim],
                    'layer_11': [B, embed_dim] if pool_patches else [B, num_patches, embed_dim],
                    'cls': [B, embed_dim]  # 最后一层的CLS token
                }
        """
        # 提取多层特征
        features_list = self.backbone.get_intermediate_layers(
            images,
            n=self.layers,
            reshape=False,  # 保持 [B, N+1, C] 格式 (1 cls + N patches)
            return_class_token=True,
            norm=True
        )
        
        if return_cls_only:
            # 只返回最后一层的CLS token用于分类
            last_features = features_list[-1]
            if isinstance(last_features, tuple):
                last_features = last_features[0]
            cls_token = last_features[:, 0]  # [B, embed_dim]取cls token
            return cls_token
        
        # 构建多层特征字典
        multi_layer_features = {}
        
        for i, layer_idx in enumerate(self.layers):
            feat = features_list[i]
            
            # 处理可能的tuple返回值
            if isinstance(feat, tuple):
                feat = feat[0]
            
            # feat shape: [B, N+1, embed_dim]
            # N+1 = 1 (cls) + num_patches
            
            cls_token = feat[:, 0]  # [B, embed_dim]
            patch_tokens = feat[:, 1:]  # [B, num_patches, embed_dim]
            
            # 可选：对patch tokens进行平均池化（取平均数，只留空间语义，大幅节省内存）
            if pool_patches:
                # [B, num_patches, embed_dim] -> [B, embed_dim]
                patch_tokens = patch_tokens.mean(dim=1)
            
            # 存储patch tokens（用于多尺度不确定性）
            multi_layer_features[f'layer_{layer_idx}'] = patch_tokens
            
            # 最后一层的CLS token用于分类
            if i == len(self.layers) - 1:
                multi_layer_features['cls'] = cls_token
        
        return multi_layer_features
    
    def get_global_features(self, images):
        """
        获取全局特征（最后一层CLS token）
        
        Args:
            images: [B, 3, H, W]
        
        Returns:
            features: [B, embed_dim]
        """
        return self.forward(images, return_cls_only=True)
    
    def get_patch_features(self, images, layer_idx=-1):
        """
        获取特定层的patch特征
        
        Args:
            images: [B, 3, H, W]
            layer_idx: 层索引（-1表示最后一层）
        
        Returns:
            patch_features: [B, num_patches, embed_dim]
        """
        features_dict = self.forward(images, return_cls_only=False)
        
        if layer_idx == -1:
            layer_idx = self.layers[-1]
        
        layer_key = f'layer_{layer_idx}'
        return features_dict[layer_key]


def test_feature_extractor():
    """测试特征提取器"""
    # 导入 Config
    try:
        from .config import Config
    except ImportError:
        from config import Config
    
    print("=" * 60)
    print("测试多层DINOv3特征提取器")
    print("=" * 60)
    print()
    
    print(f"配置来源: config.py")
    print(f"  model_size: {Config.model_size}")
    print(f"  feature_layers: {Config.feature_layers}")
    print()
    
    # 创建提取器（参数从 Config 读取）
    model = MultiLayerDINOv3(
        # model_size 和 layers 不传，自动从 Config 读取
        pretrained=True
    )
    model = model.cuda()
    
    # 测试输入
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224).cuda()
    
    print(f"输入shape: {images.shape}")
    print()
    
    # 测试1: 多层特征提取
    print("1. 多层特征提取:")
    features = model(images, return_cls_only=False)
    
    for key, value in features.items():
        print(f"  {key}: {value.shape}")
    
    # 测试2: 只提取CLS token
    print("\n2. 只提取CLS token (用于分类):")
    cls_features = model.get_global_features(images)
    print(f"  cls_features: {cls_features.shape}")
    
    # 测试3: 提取特定层的patch features
    print("\n3. 提取最后一层的patch features:")
    patch_features = model.get_patch_features(images, layer_idx=-1)
    print(f"  patch_features: {patch_features.shape}")
    
    # 验证patch数量
    H, W = 224, 224
    patch_size = 16  # DINOv3 patch_size=16
    expected_patches = (H // patch_size) * (W // patch_size)
    print(f"\n预期patch数量: {expected_patches}")
    print(f"实际patch数量: {patch_features.shape[1]}")
    
    print("\n" + "=" * 60)
    print("✓ 测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    test_feature_extractor()

