# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
DINOv3 Vision Transformer 实现

这个文件定义了 DINOv3 使用的 Vision Transformer (ViT) 架构
核心类是 DinoVisionTransformer，它是整个模型的骨干网络

主要功能：
1. 将图像分割成 patches 并进行 embedding
2. 通过多层 Transformer blocks 提取特征
3. 支持提取中间层特征（用于主动学习等下游任务）
4. 支持不同规模的模型（small/base/large/giant 等）

===============================================================================
学习指南：你需要重点理解的部分
===============================================================================

对于主动学习研究，你需要掌握：

1. 核心概念（必须理解）：
   - ViT 的数据流：图像 -> Patches -> Transformer -> 特征
   - CLS Token 的作用：全局图像表示
   - 多层特征的意义：浅层=纹理，中层=结构，深层=语义
   
2. 关键方法（重点阅读）：
   - get_intermediate_layers()：提取多层特征（主动学习最常用）
   - prepare_tokens_with_masks()：理解 token 的组织方式
   - __init__()：理解模型的组成部分
   
3. 模型规模（需要知道）：
   - vit_small: 384 维, 12 层, 21M 参数
   - vit_base: 768 维, 12 层, 86M 参数（推荐用于主动学习）
   - vit_large: 1024 维, 24 层, 300M 参数

4. 可以跳过的部分（不影响理解）：
   - RoPE 位置编码的数学细节
   - forward_features_list()：多尺度训练相关
   - 权重初始化函数的实现细节
   - 超大模型（giant/7b）的配置

5. 在主动学习中的使用：
   ```python
   # 创建模型
   model = vit_base()
   # 加载预训练权重
   model.load_state_dict(torch.load('pretrained_weights.pth'))
   model.eval()
   
   # 提取多层特征
   images = torch.randn(4, 3, 224, 224)  # batch=4
   features = model.get_intermediate_layers(
       images, 
       n=[3, 6, 9, 11],  # 提取第 4, 7, 10, 12 层
       return_class_token=True,  # 返回 CLS token
       norm=True  # 归一化
   )
   # features: [(patches_3, cls_3), (patches_6, cls_6), ...]
   # cls_3: [4, 768] - 第 4 层的全局特征
   ```

学习建议：
- 第一遍：通读注释，理解整体流程
- 第二遍：重点研究 get_intermediate_layers()
- 第三遍：对照你的 feature_extractor.py 代码理解如何使用
- 动手实践：运行示例代码，观察输入输出形状

===============================================================================
"""

import logging
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn.init
from torch import Tensor, nn

from dinov3.layers import LayerScale, Mlp, PatchEmbed, RMSNorm, RopePositionEmbedding, SelfAttentionBlock, SwiGLUFFN
from dinov3.utils import named_apply

logger = logging.getLogger("dinov3")

# 前馈网络层的选项字典
# mlp: 标准的多层感知机(multilayer perceptron)
# swiglu: 使用 SwiGLU 激活函数的 FFN（性能更好）
ffn_layer_dict = {
    "mlp": Mlp,
    "swiglu": SwiGLUFFN,
    "swiglu32": partial(SwiGLUFFN, align_to=32),
    "swiglu64": partial(SwiGLUFFN, align_to=64),
    "swiglu128": partial(SwiGLUFFN, align_to=128),
}

# 归一化层的选项字典
# layernorm: 标准的 Layer Normalization
# rmsnorm: Root Mean Square Normalization（更高效）
norm_layer_dict = {
    "layernorm": partial(nn.LayerNorm, eps=1e-6),
    "layernormbf16": partial(nn.LayerNorm, eps=1e-5),
    "rmsnorm": RMSNorm,
}

# 数据类型字典（用于混合精度训练）
dtype_dict = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


"""
权重初始化函数（用于模型训练，使用预训练模型时可以跳过）

作用：为 ViT 的各个模块初始化权重
参数：
    module: 要初始化的神经网络模块
    name: 模块的名称（可选，用于调试）

说明：这个函数会被递归应用到模型的所有子模块
"""
def init_weights_vit(module: nn.Module, name: str = ""):
    # 线性层：使用截断正态分布初始化权重
    if isinstance(module, nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        if hasattr(module, "bias_mask") and module.bias_mask is not None:
            o = module.out_features
            module.bias_mask.fill_(1)
            module.bias_mask[o // 3 : 2 * o // 3].fill_(0)
    # 各种归一化层和特殊层：使用默认的参数重置方法
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    if isinstance(module, LayerScale):
        module.reset_parameters()
    if isinstance(module, PatchEmbed):
        module.reset_parameters()
    if isinstance(module, RMSNorm):
        module.reset_parameters()


"""
DINOv3 Vision Transformer 核心类

这是 DINOv3 的骨干网络，负责将图像转换为特征表示
主要流程：图像 -> Patches -> Transformer Blocks -> 特征向量
"""
class DinoVisionTransformer(nn.Module):
    """
    初始化 DINOv3 Vision Transformer
    
    作用：创建模型的所有组件（patch embedding、transformer blocks、normalization等）
    
    核心参数（你需要理解的）：
        img_size: 输入图像大小，默认 224（即 224x224 像素）
        patch_size: 每个 patch 的大小，默认 16（即每个 patch 是 16x16 像素）
                   224/16 = 14，所以图像会被分成 14x14 = 196 个 patches
        in_chans: 输入图像通道数，默认 3（RGB 图像）
        embed_dim: 特征维度（每个 patch 被编码成多少维的向量）
                  small=384, base=768, large=1024, giant=1536
        depth: Transformer 层数（有多少个 Transformer blocks）
              small/base=12, large=24, giant=40
              你在主动学习中会从这些层中提取中间特征
        num_heads: 多头注意力的头数
                  small=6, base=12, large=16, giant=24
    
    位置编码相关参数（RoPE - Rotary Position Embedding）：
        pos_embed_rope_*: 这些参数控制位置编码的行为
                         用于让模型知道每个 patch 在图像中的位置
                         可以暂时不深入理解细节
    
    其他参数（可以暂时忽略）：
        ffn_ratio: 前馈网络的扩展比例，默认 4.0
        qkv_bias: Query/Key/Value 是否使用 bias
        drop_path_rate: DropPath 正则化比例
        layerscale_init: LayerScale 初始化值
        norm_layer: 归一化层类型（"layernorm" 或 "rmsnorm"）
        ffn_layer: 前馈网络类型（"mlp" 或 "swiglu"）
        n_storage_tokens: Register tokens 的数量（DINOv3 的特性）
        untie_*: 是否对不同类型的 token 使用不同的归一化层
    """
    def __init__(
        self,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        pos_embed_rope_base: float = 100.0,
        pos_embed_rope_min_period: float | None = None,
        pos_embed_rope_max_period: float | None = None,
        pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        pos_embed_rope_shift_coords: float | None = None,
        pos_embed_rope_jitter_coords: float | None = None,
        pos_embed_rope_rescale_coords: float | None = None,
        pos_embed_rope_dtype: str = "bf16",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        layerscale_init: float | None = None,
        norm_layer: str = "layernorm",
        ffn_layer: str = "mlp",
        ffn_bias: bool = True,
        proj_bias: bool = True,
        n_storage_tokens: int = 0,
        mask_k_bias: bool = False,
        untie_cls_and_patch_norms: bool = False,
        untie_global_and_local_cls_norm: bool = False,
        device: Any | None = None,
        **ignored_kwargs,
    ):
        super().__init__()
        if len(ignored_kwargs) > 0:
            logger.warning(f"Ignored kwargs: {ignored_kwargs}")
        del ignored_kwargs

        # 选择归一化层类型
        norm_layer_cls = norm_layer_dict[norm_layer]

        # 保存基本参数
        self.num_features = self.embed_dim = embed_dim  # 特征维度（与其他模型保持接口一致）
        self.n_blocks = depth  # Transformer 层数
        self.num_heads = num_heads  # 注意力头数
        self.patch_size = patch_size  # Patch 大小

        # 第1步：Patch Embedding 层
        # 作用：将图像 [B, 3, 224, 224] 转换为 patches [B, H, W, embed_dim]
        # 例如：[B, 3, 224, 224] -> [B, 14, 14, 768]（当 patch_size=16, embed_dim=768）
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,  # 不展平，保持 [B, H, W, D] 形状
        )

        # 第2步：CLS Token（分类标记）
        # 作用：这是一个可学习的特殊 token，会被添加到所有 patches 前面
        # 最终这个 token 的特征会作为整张图像的全局表示
        # 形状：[1, 1, embed_dim]，会在前向传播时扩展到 [B, 1, embed_dim]
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, device=device))
        
        # 第3步：Storage Tokens（Register Tokens，DINOv3 的创新）
        # 作用：额外的可学习 tokens，帮助稳定训练和提高性能
        # 在推理时通常不使用，但会保留在模型中
        self.n_storage_tokens = n_storage_tokens
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, n_storage_tokens, embed_dim, device=device))
        # 第4步：位置编码（RoPE - Rotary Position Embedding）
        # 作用：让模型知道每个 patch 在图像中的空间位置
        # 为什么需要：Transformer 本身没有位置信息，需要手动添加
        # RoPE 是一种旋转位置编码，比传统的绝对位置编码更有效
        logger.info(f"using base={pos_embed_rope_base} for rope new")
        logger.info(f"using min_period={pos_embed_rope_min_period} for rope new")
        logger.info(f"using max_period={pos_embed_rope_max_period} for rope new")
        logger.info(f"using normalize_coords={pos_embed_rope_normalize_coords} for rope new")
        logger.info(f"using shift_coords={pos_embed_rope_shift_coords} for rope new")
        logger.info(f"using rescale_coords={pos_embed_rope_rescale_coords} for rope new")
        logger.info(f"using jitter_coords={pos_embed_rope_jitter_coords} for rope new")
        logger.info(f"using dtype={pos_embed_rope_dtype} for rope new")
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=dtype_dict[pos_embed_rope_dtype],
            device=device,
        )
        
        # 第5步：创建 Transformer Blocks（核心部分）
        # 作用：这是模型的主体，由 depth 个相同结构的 block 堆叠而成
        # 每个 block 包含：Self-Attention + Feed-Forward Network (FFN)
        # 例如：base 模型有 12 个 blocks，large 模型有 24 个
        logger.info(f"using {ffn_layer} layer as FFN")
        ffn_layer_cls = ffn_layer_dict[ffn_layer]
        ffn_ratio_sequence = [ffn_ratio] * depth  # 每层使用相同的 FFN 比例
        
        # 创建 depth 个 Transformer blocks
        blocks_list = [
            SelfAttentionBlock(
                dim=embed_dim,  # 特征维度
                num_heads=num_heads,  # 注意力头数
                ffn_ratio=ffn_ratio_sequence[i],  # FFN 扩展比例
                qkv_bias=qkv_bias,  # QKV 是否使用 bias
                proj_bias=proj_bias,  # 投影层是否使用 bias
                ffn_bias=ffn_bias,  # FFN 是否使用 bias
                drop_path=drop_path_rate,  # DropPath 正则化
                norm_layer=norm_layer_cls,  # 归一化层类型
                act_layer=nn.GELU,  # 激活函数
                ffn_layer=ffn_layer_cls,  # FFN 层类型
                init_values=layerscale_init,  # LayerScale 初始化
                mask_k_bias=mask_k_bias,  # Key 的 mask bias
                device=device,
            )
            for i in range(depth)  # 创建 depth 个 block
        ]

        self.chunked_blocks = False  # 是否使用分块处理（用于超大模型）
        self.blocks = nn.ModuleList(blocks_list)  # 将所有 blocks 包装成 ModuleList

        # 第6步：最终归一化层（Normalization）
        # 作用：在提取特征前对所有 token 进行归一化，稳定特征分布
        # 默认情况下，所有 token（CLS + Storage + Patches）共用一个归一化层
        self.norm = norm_layer_cls(embed_dim)

        # 可选：为 CLS token 和 Patch tokens 使用不同的归一化层
        # 作用：CLS token 和 Patch tokens 的分布可能不同，分开归一化可能更好
        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        if untie_cls_and_patch_norms:
            self.cls_norm = norm_layer_cls(embed_dim)
        else:
            self.cls_norm = None

        # 可选：为全局和局部 CLS token 使用不同的归一化层
        # 作用：训练时可能使用多尺度裁剪，全局和局部的 CLS token 需要分开处理
        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm
        if untie_global_and_local_cls_norm:
            self.local_cls_norm = norm_layer_cls(embed_dim)
        else:
            self.local_cls_norm = None
        
        # 第7步：分类头（这里只是占位符）
        # 实际使用时，DINOv3 通常直接使用 CLS token 的特征，不需要额外的分类头
        self.head = nn.Identity()
        
        # 第8步：Mask Token（用于掩码图像建模训练，推理时通常不用）
        # 作用：在自监督训练时，被掩码的 patches 会被替换为这个 token
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim, device=device))

    """
    初始化模型权重
    
    作用：为模型的各个可学习参数设置初始值
    说明：使用预训练模型时，这个方法已经被调用过了，不需要再次调用
    
    初始化内容：
        1. 位置编码的权重
        2. CLS token（正态分布，标准差 0.02）
        3. Storage tokens（如果有的话）
        4. Mask token（全零初始化）
        5. 所有子模块的权重（通过 init_weights_vit 函数）
    """
    def init_weights(self):
        # 初始化位置编码
        self.rope_embed._init_weights()
        # 初始化 CLS token
        nn.init.normal_(self.cls_token, std=0.02)
        # 初始化 Storage tokens（如果有）
        if self.n_storage_tokens > 0:
            nn.init.normal_(self.storage_tokens, std=0.02)
        # 初始化 Mask token（全零）
        nn.init.zeros_(self.mask_token)
        # 递归初始化所有子模块
        named_apply(init_weights_vit, self)

    """
    准备输入 tokens（支持掩码）
    
    作用：将输入图像转换为 token 序列，并添加 CLS token 和 Storage tokens
    
    参数：
        x: 输入图像张量 [B, 3, H, W]，例如 [B, 3, 224, 224]
        masks: 可选的掩码张量，用于掩码图像建模训练
    
    返回：
        x: Token 序列 [B, N, D]
           N = 1(CLS) + n_storage_tokens + num_patches
           例如：[B, 197, 768] (base 模型，无 storage tokens)
        (H, W): Patch 网格的高度和宽度，例如 (14, 14)
    
    处理流程：
        图像 [B, 3, 224, 224]
        -> Patch Embed -> [B, 14, 14, 768]
        -> Flatten -> [B, 196, 768]
        -> 添加 CLS + Storage -> [B, 1+0+196, 768] = [B, 197, 768]
    """
    def prepare_tokens_with_masks(self, x: Tensor, masks=None) -> Tuple[Tensor, Tuple[int]]:
        # 第1步：Patch Embedding
        # 输入：[B, 3, 224, 224]
        # 输出：[B, H, W, embed_dim]，例如 [B, 14, 14, 768]
        x = self.patch_embed(x)
        B, H, W, _ = x.shape
        
        # 第2步：展平 patches
        # [B, H, W, D] -> [B, H*W, D]，例如 [B, 196, 768]
        x = x.flatten(1, 2)

        # 第3步：应用掩码（如果有）
        # 在训练时，某些 patches 会被掩码替换（用于自监督学习）
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
            cls_token = self.cls_token
        else:
            # 推理时不使用掩码，但仍需要让计算图包含 mask_token（用于梯度传播）
            cls_token = self.cls_token + 0 * self.mask_token
        
        # 第4步：准备 Storage tokens（如果有）
        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens
        else:
            # 创建空的 storage tokens（占位符）
            storage_tokens = torch.empty(
                1,
                0,
                cls_token.shape[-1],
                dtype=cls_token.dtype,
                device=cls_token.device,
            )

        # 第5步：拼接所有 tokens
        # 顺序：[CLS token] + [Storage tokens] + [Patch tokens]
        # 形状：[B, 1, D] + [B, n_storage, D] + [B, 196, D] = [B, 197, D]
        x = torch.cat(
            [
                cls_token.expand(B, -1, -1),  # 扩展 CLS token 到 batch
                storage_tokens.expand(B, -1, -1),  # 扩展 Storage tokens 到 batch
                x,  # Patch tokens
            ],
            dim=1,
        )

        # 返回：token 序列 + patch 网格尺寸
        return x, (H, W)

    def forward_features_list(self, x_list: List[Tensor], masks_list: List[Tensor]) -> List[Dict[str, Tensor]]:
        x = []
        rope = []
        for t_x, t_masks in zip(x_list, masks_list):
            t2_x, hw_tuple = self.prepare_tokens_with_masks(t_x, t_masks)
            x.append(t2_x)
            rope.append(hw_tuple)
        for _, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = [self.rope_embed(H=H, W=W) for H, W in rope]
            else:
                rope_sincos = [None for r in rope]
            x = blk(x, rope_sincos)
        all_x = x
        output = []
        for idx, (x, masks) in enumerate(zip(all_x, masks_list)):
            if self.untie_cls_and_patch_norms or self.untie_global_and_local_cls_norm:
                if self.untie_global_and_local_cls_norm and self.training and idx == 1:
                    # Assume second entry of list corresponds to local crops.
                    # We only ever apply this during training.
                    x_norm_cls_reg = self.local_cls_norm(x[:, : self.n_storage_tokens + 1])
                elif self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(x[:, : self.n_storage_tokens + 1])
                else:
                    x_norm_cls_reg = self.norm(x[:, : self.n_storage_tokens + 1])
                x_norm_patch = self.norm(x[:, self.n_storage_tokens + 1 :])
            else:
                x_norm = self.norm(x)
                x_norm_cls_reg = x_norm[:, : self.n_storage_tokens + 1]
                x_norm_patch = x_norm[:, self.n_storage_tokens + 1 :]
            output.append(
                {
                    "x_norm_clstoken": x_norm_cls_reg[:, 0],
                    "x_storage_tokens": x_norm_cls_reg[:, 1:],
                    "x_norm_patchtokens": x_norm_patch,
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    """
    前向传播提取特征（完整版本）
    
    作用：运行完整的前向传播，返回所有类型的特征
    
    参数：
        x: 输入图像，可以是：
           - 单个张量 [B, 3, H, W]
           - 张量列表（用于多尺度训练）
        masks: 可选的掩码（用于掩码图像建模训练）
    
    返回：
        字典或字典列表，包含：
        - x_norm_clstoken: 归一化后的 CLS token 特征 [B, D]
        - x_storage_tokens: Storage tokens 特征
        - x_norm_patchtokens: 归一化后的 patch tokens [B, num_patches, D]
        - x_prenorm: 归一化前的所有 tokens
        - masks: 掩码信息
    
    说明：这个方法主要用于训练，推理时通常使用 get_intermediate_layers
    """
    def forward_features(self, x: Tensor | List[Tensor], masks: Optional[Tensor] = None) -> List[Dict[str, Tensor]]:
        # 如果输入是单个张量，转换为列表格式
        if isinstance(x, torch.Tensor):
            return self.forward_features_list([x], [masks])[0]
        else:
            # 如果已经是列表，直接处理（用于多尺度训练）
            return self.forward_features_list(x, masks)

    """
    提取中间层特征（内部实现，不使用分块）
    
    作用：遍历所有 Transformer blocks，提取指定层的输出
    
    参数：
        x: 输入图像 [B, 3, H, W]
        n: 要提取的层数或层索引列表
           - 如果是 int：提取最后 n 层，例如 n=4 提取第 9,10,11,12 层（共12层）
           - 如果是 list：提取指定索引的层，例如 n=[3,6,9,11] 提取第 4,7,10,12 层
    
    返回：
        output: 包含指定层特征的列表，每个元素形状为 [B, N, D]
                N = 1 + n_storage + num_patches
    
    说明：这个方法会运行完整的前向传播，只是在指定层保存输出
    """
    def _get_intermediate_layers_not_chunked(self, x: Tensor, n: int = 1) -> List[Tensor]:
        # 第1步：准备 tokens
        x, (H, W) = self.prepare_tokens_with_masks(x)
        
        # 第2步：确定要提取哪些层
        # 如果 n 是整数，取最后 n 层；如果是列表，取指定的层
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        
        # 第3步：遍历所有 Transformer blocks
        for i, blk in enumerate(self.blocks):
            # 计算位置编码（RoPE）
            if self.rope_embed is not None:
                rope_sincos = self.rope_embed(H=H, W=W)
            else:
                rope_sincos = None
            # 通过当前 block 进行前向传播
            x = blk(x, rope_sincos)
            # 如果当前层是需要提取的层，保存输出
            if i in blocks_to_take:
                output.append(x)
        
        # 确保提取了所有需要的层
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    """
    提取中间层特征（主动学习中最重要的方法）
    
    作用：从 Transformer 的指定层提取特征，用于下游任务
    这是你在主动学习中会频繁使用的方法！
    
    参数：
        x: 输入图像 [B, 3, H, W]，例如 [B, 3, 224, 224]
        
        n: 要提取的层（重要参数）
           - int: 提取最后 n 层，例如 n=4 提取第 9,10,11,12 层
           - list: 提取指定层，例如 n=[3,6,9,11] 提取第 4,7,10,12 层
           注意：层索引从 0 开始，第 0 层是第一个 Transformer block
        
        reshape: 是否将 patch tokens 重塑为 2D 特征图
                 False: 保持 [B, num_patches, D] 形状
                 True: 重塑为 [B, D, H', W'] 形状（用于密集预测任务）
        
        return_class_token: 是否返回 CLS token 的特征
                           True: 返回每层的 CLS token [B, D]
                           False: 只返回 patch tokens
        
        return_extra_tokens: 是否返回 storage tokens
                            通常设置为 False
        
        norm: 是否对输出进行归一化
              True: 应用 LayerNorm（推荐，特征更稳定）
              False: 返回原始特征
    
    返回：
        根据参数组合，返回不同的内容：
        - 只要 patches: tuple([patch_features_layer1, patch_features_layer2, ...])
        - 要 CLS: tuple([(patches_layer1, cls_layer1), (patches_layer2, cls_layer2), ...])
    
    使用示例（主动学习场景）：
        # 提取最后 4 层的 CLS token 特征
        features = model.get_intermediate_layers(
            images,                    # [B, 3, 224, 224]
            n=4,                       # 提取最后 4 层
            return_class_token=True,   # 返回 CLS token
            norm=True                  # 归一化
        )
        # features: [(patches_9, cls_9), (patches_10, cls_10), ...]
        # cls_9: [B, 768] - 第 9 层的全局特征
    """
    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        *,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        # 第1步：提取中间层的原始输出
        outputs = self._get_intermediate_layers_not_chunked(x, n)
        
        # 第2步：归一化（如果需要）
        if norm:
            outputs_normed = []
            for out in outputs:
                # 如果 CLS 和 Patch 使用不同的归一化层
                if self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(out[:, : self.n_storage_tokens + 1])
                    x_norm_patch = self.norm(out[:, self.n_storage_tokens + 1 :])
                    outputs_normed.append(torch.cat((x_norm_cls_reg, x_norm_patch), dim=1))
                else:
                    # 默认：所有 token 使用相同的归一化
                    outputs_normed.append(self.norm(out))
            outputs = outputs_normed
        
        # 第3步：分离不同类型的 tokens
        # CLS token: 每层的第 0 个 token，形状 [B, D]
        class_tokens = [out[:, 0] for out in outputs]
        # Storage tokens: 第 1 到 n_storage_tokens 个 token
        extra_tokens = [out[:, 1 : self.n_storage_tokens + 1] for out in outputs]
        # Patch tokens: 剩余的所有 tokens，形状 [B, num_patches, D]
        outputs = [out[:, self.n_storage_tokens + 1 :] for out in outputs]
        
        # 第4步：重塑 patch tokens（如果需要）
        if reshape:
            B, _, h, w = x.shape
            # 从 [B, num_patches, D] 重塑为 [B, D, H', W']
            # 用于密集预测任务（如分割）
            outputs = [
                out.reshape(B, h // self.patch_size, w // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        
        # 第5步：根据参数返回不同的组合
        if not return_class_token and not return_extra_tokens:
            # 只返回 patch tokens
            return tuple(outputs)
        elif return_class_token and not return_extra_tokens:
            # 返回 (patch tokens, CLS token) 的元组
            # 这是主动学习中最常用的模式
            return tuple(zip(outputs, class_tokens))
        elif not return_class_token and return_extra_tokens:
            # 返回 (patch tokens, storage tokens)
            return tuple(zip(outputs, extra_tokens))
        elif return_class_token and return_extra_tokens:
            # 返回所有：(patch tokens, CLS token, storage tokens)
            return tuple(zip(outputs, class_tokens, extra_tokens))

    """
    整体前向传播
    
    作用：模型的主入口函数
    
    参数：
        *args, **kwargs: 传递给 forward_features 的参数
        is_training: 是否处于训练模式
    
    返回：
        - 训练模式：返回完整的特征字典（包含所有 tokens）
        - 推理模式：只返回 CLS token 的特征 [B, D]
    
    说明：
        - 训练时需要所有信息（CLS + patches + storage）
        - 推理时通常只需要 CLS token 作为图像的全局表示
        - 在主动学习中，通常直接使用 get_intermediate_layers 而不是这个方法
    """
    def forward(self, *args, is_training: bool = False, **kwargs) -> List[Dict[str, Tensor]] | Tensor:
        # 提取特征
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            # 训练模式：返回完整特征字典
            return ret
        else:
            # 推理模式：只返回 CLS token 特征
            return self.head(ret["x_norm_clstoken"])


"""
创建 ViT-Small 模型

规模：约 21M 参数
配置：
    - embed_dim: 384 维特征
    - depth: 12 层 Transformer
    - num_heads: 6 个注意力头
    - 每个头的维度: 384/6 = 64

适用场景：
    - 计算资源有限
    - 需要快速推理
    - 数据集较小

参数：
    patch_size: Patch 大小，默认 16
    **kwargs: 其他参数传递给 DinoVisionTransformer

返回：
    DinoVisionTransformer 模型实例
"""
def vit_small(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=4,
        **kwargs,
    )
    return model


"""
创建 ViT-Base 模型（推荐用于主动学习）

规模：约 86M 参数
配置：
    - embed_dim: 768 维特征
    - depth: 12 层 Transformer
    - num_heads: 12 个注意力头
    - 每个头的维度: 768/12 = 64

适用场景：
    - 性能和速度的平衡点
    - 大多数主动学习研究使用这个规模
    - 可以在单张 GPU 上训练和推理

参数：
    patch_size: Patch 大小，默认 16
    **kwargs: 其他参数传递给 DinoVisionTransformer

返回：
    DinoVisionTransformer 模型实例

使用示例：
    # 创建 base 模型
    model = vit_base(patch_size=16)
    # 加载预训练权重
    model.load_state_dict(torch.load('dinov3_vitb16_pretrain.pth'))
"""
def vit_base(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_ratio=4,
        **kwargs,
    )
    return model


"""
创建 ViT-Large 模型

规模：约 300M 参数
配置：
    - embed_dim: 1024 维特征
    - depth: 24 层 Transformer（是 base 的两倍）
    - num_heads: 16 个注意力头
    - 每个头的维度: 1024/16 = 64

适用场景：
    - 追求最高性能
    - 计算资源充足
    - 大规模数据集

参数：
    patch_size: Patch 大小，默认 16
    **kwargs: 其他参数传递给 DinoVisionTransformer

返回：
    DinoVisionTransformer 模型实例

注意：
    - 需要更多显存（推理约需 16GB）
    - 推理速度较慢
    - 在小数据集上可能过拟合
"""
def vit_large(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        **kwargs,
    )
    return model


"""
创建 ViT-SO400M 模型

规模：约 400M 参数
配置：
    - embed_dim: 1152 维特征
    - depth: 27 层 Transformer
    - num_heads: 18 个注意力头
    
说明：这是介于 Large 和 Giant 之间的中等超大模型
主动学习通常不需要这么大的模型
"""
def vit_so400m(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1152,
        depth=27,
        num_heads=18,
        ffn_ratio=3.777777778,
        **kwargs,
    )
    return model


"""
创建 ViT-Huge2 模型

规模：约 600M+ 参数
配置：
    - embed_dim: 1280 维特征
    - depth: 32 层 Transformer
    - num_heads: 20 个注意力头

注意：需要大量显存（24GB+），主要用于超大规模数据集
"""
def vit_huge2(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1280,
        depth=32,
        num_heads=20,
        ffn_ratio=4,
        **kwargs,
    )
    return model


"""
创建 ViT-Giant2 模型

规模：约 1B+ 参数
配置：
    - embed_dim: 1536 维特征
    - depth: 40 层 Transformer
    - num_heads: 24 个注意力头
    - 每个头的维度: 1536/24 = 64

说明：接近 ViT-Giant 的配置，是非常大的模型
需要多卡训练和推理，主要用于研究和超大规模任务
"""
def vit_giant2(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        ffn_ratio=4,
        **kwargs,
    )
    return model


"""
创建 ViT-7B 模型（最大规模）

规模：约 7B（70亿）参数
配置：
    - embed_dim: 4096 维特征
    - depth: 40 层 Transformer
    - num_heads: 32 个注意力头
    - 每个头的维度: 4096/32 = 128

说明：这是 DINOv3 的最大模型，需要多张 A100/H100 才能运行
主要用于前沿研究，一般研究和应用不会使用这个规模
"""
def vit_7b(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=4096,
        depth=40,
        num_heads=32,
        ffn_ratio=3,
        **kwargs,
    )
    return model
