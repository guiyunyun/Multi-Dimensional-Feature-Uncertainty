# 🎓 DINOv3主动学习项目 - 完整学习指南

> **目标**：从零开始理解整个主动学习项目，掌握DINOv3特征提取和4种不确定性计算的原理与实现

---

## 📋 目录

1. [项目概述](#1-项目概述)
2. [学习路线图](#2-学习路线图)
3. [核心概念](#3-核心概念)
4. [文件结构](#4-文件结构)
5. [代码学习顺序](#5-代码学习顺序)
6. [关键代码解析](#6-关键代码解析)
7. [DINOv3源码学习](#7-dinov3源码学习)
8. [实验与调优](#8-实验与调优)
9. [常见问题](#9-常见问题)

---

## 1. 项目概述

### 1.1 研究目标

**用最少的标注数据，达到最好的分类性能**

- **数据集**：ImageNet-100 (101,347张图，100个类别)
- **冻结骨干**：DINOv3-ViT-Base/16（参数不更新，只提取特征）
- **标注比例**：0.2% → 4% (200张 → 4,200张)
- **目标准确率**：70-80% (vs 全量训练~85%)

### 1.2 核心创新

**四种不确定性的级联选择机制：**

```
未标注样本 
  ↓
[提取多层特征]
  ↓
┌─────────────────────────────────────┐
│ 1. Density (密度不确定性)            │ 过滤噪声
│    → KNN相似度标准差                 │
├─────────────────────────────────────┤
│ 2. Exploration (探索性不确定性)      │ 探索新区域
│    → 到labeled样本的最小距离         │
├─────────────────────────────────────┤
│ 3. Boundary (边界不确定性)           │ 决策边界
│    → KNN标签不一致性                 │
├─────────────────────────────────────┤
│ 4. Multi-Scale (多尺度特征不确定性)  │ 语义复杂度
│    → 层间一致性 ū·(1-Var)            │
└─────────────────────────────────────┘
  ↓
[Cascading Selector]
  ↓
选出最有价值的样本
```

### 1.3 技术栈

- **深度学习框架**：PyTorch 2.0+
- **视觉模型**：DINOv3 (Facebook AI)
- **数据集**：ImageNet-100
- **环境**：AutoDL (4090D GPU, 80GB内存)

---

## 2. 学习路线图

### 🎯 第一阶段：理解基础概念（1-2天）

```mermaid
graph LR
    A[什么是主动学习] --> B[什么是DINOv3]
    B --> C[什么是特征空间]
    C --> D[什么是不确定性]
```

**学习材料：**
1. 阅读 `active_learning/README.md` - 项目整体介绍
2. 理解主动学习流程图
3. 了解DINOv3的Vision Transformer架构

---

### 🎯 第二阶段：数据流理解（1天）

**学习目标：** 理解数据如何从图片变成特征，再到选择样本

**学习路径：**
```
图片文件 → DataLoader → DINOv3 → 特征向量 → 不确定性计算 → 样本选择
```

**推荐顺序：**
1. 📄 [`active_learning/dataset.py`](#file-dataset) - 数据加载
2. 📄 [`active_learning/feature_extractor.py`](#file-feature-extractor) - 特征提取
3. 📄 [`active_learning/memory_bank.py`](#file-memory-bank) - 特征存储

---

### 🎯 第三阶段：核心算法（2-3天）

**学习目标：** 深入理解4种不确定性的计算原理

**学习路径：**
1. 📄 [`active_learning/uncertainty.py`](#file-uncertainty) ⭐⭐⭐
   - 重点：每种不确定性的公式和直觉
   - 动手：修改参数，观察效果
2. 📄 [`active_learning/cascading_selector.py`](#file-selector) ⭐⭐
   - 重点：级联选择的决策树逻辑
   - 对比：V0/V1/V2三个版本的差异

---

### 🎯 第四阶段：主循环与训练（1天）

**学习目标：** 理解完整的主动学习循环

**学习路径：**
1. 📄 [`active_learning/classifier.py`](#file-classifier) - 简单分类头
2. 📄 [`active_learning/active_learner.py`](#file-active-learner) - 主循环
3. 📄 [`scripts/run_active_learning.py`](#file-run-script) - 运行脚本

---

### 🎯 第五阶段：DINOv3源码（选修，2-3天）

**学习目标：** 理解Vision Transformer和DINOv3的实现

**学习路径：**
1. 📄 `dinov3/models/vision_transformer.py` - ViT架构
2. 📄 `dinov3/hub/backbones.py` - 模型加载
3. 📄 `dinov3/layers/` - Attention/MLP等组件

---

## 3. 核心概念

### 3.1 主动学习 (Active Learning)

**定义：** 模型主动选择最有价值的样本进行标注

**传统方法 vs 主动学习：**

| 方法 | 选样本方式 | 标注量 | 性能 |
|-----|----------|--------|------|
| **随机采样** | 随机选 | 100% | 85% |
| **主动学习** | 智能选 | 4% | 75% |

**核心思想：** 不是所有数据都同等重要！

```python
# 哪些样本最有价值？
1. 离已知样本远的（探索新区域）
2. 在决策边界上的（区分困难样本）
3. 代表性强的（不要离群点）
4. 语义复杂的（多层特征不一致）
```

---

### 3.2 DINOv3 特征提取器

**DINOv3 = DINO v3 (Self-Distillation with NO labels)**

**关键特性：**
- ✅ **自监督学习**：不需要标注，在海量图片上学习
- ✅ **通用特征**：适用于分类、检测、分割等任务
- ✅ **多层特征**：不同层捕获不同粒度的信息

**架构：Vision Transformer (ViT)**

```
输入图片 [224×224×3]
    ↓
[Patch Embedding] 切成 14×14 = 196 个patches
    ↓
[CLS Token] + [Patch Tokens]
    ↓
[Transformer Blocks] × 12层
    ↓
Layer 3: 早期特征（边缘、纹理）
Layer 6: 中期特征（部件、形状）
Layer 9: 高期特征（物体、语义）
Layer 11: 最终特征（全局理解）
    ↓
输出：[CLS Token] [768维] → 用于分类
     [Patch Tokens] [196×768] → 用于定位
```

**为什么冻结？**
- DINOv3已经在超大规模数据上训练过
- 特征质量已经很高，微调收益小
- 冻结后只训练分类头，速度快、不易过拟合

---

### 3.3 特征空间与距离

**特征空间：** 每张图变成768维向量的空间

```python
# 原始空间（不可比）
图1: [224, 224, 3] RGB像素
图2: [224, 224, 3] RGB像素
→ 像素差异不代表语义相似度

# 特征空间（可比）
图1: [768] 特征向量
图2: [768] 特征向量
→ 余弦距离 ≈ 语义相似度
```

**距离度量：余弦相似度**

```python
similarity = cos(θ) = A·B / (||A|| ||B||)
distance = 1 - similarity

相似度高 (0.9) → 距离小 (0.1) → 语义接近
相似度低 (0.1) → 距离大 (0.9) → 语义不同
```

---

### 3.4 四种不确定性详解

#### 3.4.1 Density Uncertainty (密度不确定性)

**目的：** 过滤噪声和离群点

**原理：** 如果一个样本周围的邻居距离分散度很大，说明它可能是噪声

```python
# 伪代码
knn_similarities = find_k_nearest_neighbors(sample, k=10)
# knn_similarities = [0.95, 0.93, 0.91, ..., 0.12, 0.10]
#                     ^^^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^
#                     聚类中心：低方差          离群点：高方差

density_uncertainty = std(knn_similarities)

# 低方差 → 在稠密区域 → 代表性强 ✅
# 高方差 → 在稀疏区域 → 可能噪声 ❌
```

**应用：** 阈值过滤，避免标注噪声样本

---

#### 3.4.2 Exploration Uncertainty (探索性不确定性)

**目的：** 探索未知区域

**原理：** 离已标注样本越远，越值得探索

```python
# 伪代码
distances = compute_distances(unlabeled_sample, labeled_samples)
exploration_uncertainty = min(distances)

# 距离小 → 附近有标注 → 信息已知 ❌
# 距离大 → 远离标注区 → 新信息 ✅
```

**直觉：**
```
特征空间示意图
┌─────────────────────────────┐
│  ●labeled   ●   ●            │
│    ●   ●  ●                  │
│                              │
│           ? ← 高探索不确定性  │
│                              │
│  ●   ●                       │
│    ●   ● ? ← 低探索不确定性  │
└─────────────────────────────┘
```

---

#### 3.4.3 Boundary Uncertainty (边界不确定性)

**目的：** 找到决策边界附近的样本

**原理：** 如果最近的K个邻居标签不一致，说明在边界上

```python
# 伪代码
knn_labels = find_k_nearest_labels(sample, k=10)
# knn_labels = [狗, 狗, 猫, 狗, 猫, 猫, 狗, ...]
#              → 标签混乱 → 在决策边界上

label_counts = count(knn_labels)
boundary_uncertainty = 1 - max(label_counts) / k

# 标签一致 (狗×10) → 确定区域 ❌
# 标签混乱 (狗×5, 猫×5) → 边界区域 ✅
```

**重要性：** 边界样本最能帮助模型学习分类规则

---

#### 3.4.4 Multi-Scale Feature Uncertainty (多尺度特征不确定性)

**目的：** 识别语义复杂的样本

**原理：** 不同层的特征应该一致，如果不一致说明样本复杂

```python
# 伪代码
# 计算每层的exploration uncertainty
u_layer3 = compute_exploration(layer_3_features)  # 0.8
u_layer6 = compute_exploration(layer_6_features)  # 0.7
u_layer9 = compute_exploration(layer_9_features)  # 0.9
u_layer11 = compute_exploration(layer_11_features) # 0.75

mean_u = mean([0.8, 0.7, 0.9, 0.75]) = 0.7875
variance = var([0.8, 0.7, 0.9, 0.75]) = 0.0069

multiscale_uncertainty = mean_u × (1 - variance)
                       = 0.7875 × 0.9931
                       = 0.782

# 一致性高（低方差）→ 简单样本 ❌
# 一致性低（高方差）→ 复杂样本 ✅
```

**直觉：**
```
简单样本（猫的清晰照片）：
- Layer 3: 确定（纹理一致）
- Layer 6: 确定（耳朵、胡须清晰）
- Layer 9: 确定（整体形状明确）
→ 方差小 → 低不确定性

复杂样本（猫狗重叠、遮挡）：
- Layer 3: 不确定（纹理混乱）
- Layer 6: 不确定（部件模糊）
- Layer 9: 不确定（物体混淆）
→ 方差大 → 高不确定性
```

---

## 4. 文件结构

### 4.1 项目目录树

```
dinov3/
├── active_learning/              # 主动学习核心代码 ⭐
│   ├── __init__.py               # 包初始化
│   ├── dataset.py                # 数据加载与管理
│   ├── feature_extractor.py      # DINOv3多层特征提取
│   ├── memory_bank.py            # 特征存储与KNN计算
│   ├── uncertainty.py            # 4种不确定性计算 ⭐⭐⭐
│   ├── cascading_selector.py     # 级联选择器 V0
│   ├── cascading_selector_v1.py  # 级联选择器 V1（综合判断）
│   ├── cascading_selector_v2.py  # 级联选择器 V2（加权分数）
│   ├── classifier.py             # 简单分类头
│   ├── active_learner.py         # 主动学习主循环 ⭐⭐
│   └── README.md                 # 详细技术文档
│
├── scripts/                      # 运行脚本
│   ├── run_active_learning.py    # 主运行脚本 ⭐
│   ├── view_results.py           # 结果查看工具
│   ├── prepare_imagenet100.py    # 数据集准备
│   └── README.md                 # 使用说明
│
├── dinov3/                       # DINOv3官方源码
│   ├── models/
│   │   └── vision_transformer.py # ViT架构实现
│   ├── hub/
│   │   └── backbones.py          # 模型加载接口
│   ├── layers/                   # 网络层实现
│   │   ├── attention.py          # Self-Attention
│   │   ├── mlp.py                # MLP
│   │   └── patch_embed.py        # Patch Embedding
│   └── ...
│
├── data/                         # 数据目录
│   └── imagenet100_split/
│       ├── train/                # 训练集 (101,347张)
│       └── val/                  # 验证集 (25,342张)
│
├── pretrained_models/            # 预训练权重
│   └── dinov3_vitb16_pretrain.pth
│
├── results/                      # 实验结果
│   ├── quick_test_v2/            # 快速测试结果
│   │   ├── history.json          # 训练历史
│   │   ├── checkpoint.pth        # 模型检查点
│   │   └── log_*.txt             # 运行日志
│   └── exp_full_paper_v1/        # 完整实验结果
│
├── LEARNING_GUIDE.md             # 本学习指南 ⭐
└── README.md                     # 项目总览
```

---

## 5. 代码学习顺序

### 📖 第1天：数据流（从图片到特征）

<a name="file-dataset"></a>
#### 📄 文件1：`active_learning/dataset.py` (352行)

**学习重点：**
1. `ImageNet100Dataset` 如何加载图片
2. `ActiveLearningDataset` 如何管理labeled/unlabeled pool
3. `create_initial_labeled_pool` 如何创建初始标注集

**关键代码位置：**
- **第121-156行**：数据增强（transforms）
- **第158-200行**：加载ImageNet-100
- **第202-244行**：创建初始labeled pool（优化版）
- **第16-119行**：ActiveLearningDataset类

**动手练习：**
```python
# 运行测试
cd /root/autodl-tmp/dinov3
python -m active_learning.dataset

# 理解输出
# - 数据集大小
# - labeled/unlabeled如何分割
# - DataLoader如何工作
```

**核心理解：**
```python
# 关键操作
labeled_indices = [0, 5, 10, ...]      # 已标注样本索引
unlabeled_indices = [1, 2, 3, 4, ...]  # 未标注样本索引

# 迁移样本
selected = [1, 3]  # 从unlabeled中选中
labeled_indices.extend([1, 3])         # 移到labeled
unlabeled_indices.remove([1, 3])       # 从unlabeled移除
```

---

<a name="file-feature-extractor"></a>
#### 📄 文件2：`active_learning/feature_extractor.py` (260行)

**学习重点：**
1. DINOv3模型如何加载
2. 多层特征如何提取
3. Patch特征池化优化

**关键代码位置：**
- **第39-111行**：`MultiLayerDINOv3.__init__` - 模型加载
- **第112-176行**：`forward` - 特征提取（支持池化）
- **第175-186行**：`get_global_features` - 获取CLS特征

**核心流程：**
```python
images: [B, 3, 224, 224]
    ↓
backbone.get_intermediate_layers(images, n=[3,6,9,11])
    ↓
Layer 3: [B, 197, 768]  # 1 CLS + 196 patches
Layer 6: [B, 197, 768]
Layer 9: [B, 197, 768]
Layer 11: [B, 197, 768]
    ↓
[分离 CLS 和 Patches]
    ↓
CLS: [B, 768]           # 用于分类
Patches: [B, 196, 768]  # 用于多尺度不确定性
    ↓
[可选池化] pool_patches=True
    ↓
Patches: [B, 768]       # 平均池化后（节省内存）
```

**动手练习：**
```python
# 提取单张图的特征
import torch
from active_learning import MultiLayerDINOv3

extractor = MultiLayerDINOv3(pretrained=True)
dummy_image = torch.randn(1, 3, 224, 224)

# 只要CLS特征
cls_feat = extractor(dummy_image, return_cls_only=True)
print(cls_feat.shape)  # [1, 768]

# 多层特征
features = extractor(dummy_image, return_cls_only=False, pool_patches=True)
for k, v in features.items():
    print(f"{k}: {v.shape}")
```

**重点理解：**
- 为什么要多层特征？→ 捕获多尺度信息
- 为什么要池化？→ 节省内存（320GB → 1.5GB）
- 为什么冻结？→ DINOv3特征已足够好

---

<a name="file-memory-bank"></a>
#### 📄 文件3：`active_learning/memory_bank.py` (444行)

**学习重点：**
1. 如何存储已标注样本的特征
2. KNN（K近邻）如何高效计算
3. 距离、密度、边界如何计算

**关键代码位置：**
- **第60-119行**：`add_samples` - 添加样本到Memory Bank
- **第121-152行**：`compute_knn_distances` - KNN距离
- **第154-183行**：`compute_knn_labels` - KNN标签
- **第185-216行**：`compute_knn_similarities` - KNN相似度
- **第218-241行**：`compute_density` - 密度不确定性
- **第242-298行**：`compute_multi_layer_distances` - 多层距离

**核心数据结构：**
```python
class MemoryBank:
    self.cls_features: Tensor [N, 768]           # 已标注样本的CLS特征
    self.multi_layer_features: Dict {
        'layer_3': Tensor [N, 768],              # 池化后的layer特征
        'layer_6': Tensor [N, 768],
        'layer_9': Tensor [N, 768],
        'layer_11': Tensor [N, 768],
    }
    self.labels: Tensor [N]                      # 已标注样本的标签
    self.num_samples: int                        # 已标注样本数
```

**KNN计算原理：**
```python
# 1. 计算相似度矩阵
query: [B, 768]
memory: [N, 768]
similarity = query @ memory.T  # [B, N]

# 2. 找Top-K
topk_similarities, topk_indices = torch.topk(similarity, k=10, dim=1)
# topk_similarities: [B, 10]
# topk_indices: [B, 10]

# 3. 获取邻居标签
knn_labels = self.labels[topk_indices]  # [B, 10]
```

**动手练习：**
```python
from active_learning import MemoryBank
import torch

# 创建Memory Bank
mb = MemoryBank(feature_dim=768, num_classes=100)

# 添加样本
cls_feats = torch.randn(10, 768)
layer_feats = {'layer_3': torch.randn(10, 768)}
labels = torch.randint(0, 100, (10,))
mb.add_samples(cls_feats, layer_feats, labels)

# 查询KNN
query = torch.randn(5, 768)
distances = mb.compute_knn_distances(query, k=3)
print(distances.shape)  # [5, 3]
```

---

### 📖 第2-3天：核心算法（4种不确定性）

<a name="file-uncertainty"></a>
#### 📄 文件4：`active_learning/uncertainty.py` (421行) ⭐⭐⭐

**这是最核心的文件！需要花2-3天深入理解**

**学习重点：**
1. 每种不确定性的公式推导
2. 为什么这样设计能work
3. 参数如何影响结果

**关键代码位置：**

| 不确定性类型 | 代码行数 | 核心公式 |
|------------|---------|---------|
| **Density** | 110-151行 | `std(KNN_similarities)` |
| **Exploration** | 62-86行 | `min(distances_to_labeled)` |
| **Boundary** | 88-108行 | `1 - max(label_counts)/K` |
| **Multi-Scale** | 177-235行 | `mean_u × (1 - var_u)` |
| **综合计算** | 237-295行 | `compute_all_uncertainties` |

**逐个攻克：**

##### 4.1 Density Uncertainty (110-151行)

```python
def compute_density_uncertainty(self, cls_features, normalize=True):
    """
    步骤：
    1. 找K个最近邻
    2. 计算相似度 [B, K]
    3. 计算标准差 std(similarities)
    4. 归一化到[0, 1]
    """
    # 核心逻辑
    knn_similarities = self.memory_bank.compute_knn_similarities(
        cls_features, k=self.k
    )  # [B, K] 每行是K个相似度
    
    density_uncertainty = knn_similarities.std(dim=1)  # [B]
    # 标准差大 → 邻居距离分散 → 可能噪声
    # 标准差小 → 邻居距离一致 → 稠密区域
```

**直觉理解：**
```
样本A的10个邻居相似度：[0.95, 0.94, 0.93, 0.92, 0.91, ...]
→ std很小 → 在聚类中心 → 代表性强 ✅

样本B的10个邻居相似度：[0.9, 0.8, 0.5, 0.3, 0.1, ...]
→ std很大 → 远离聚类 → 可能噪声 ❌
```

##### 4.2 Exploration Uncertainty (62-86行)

```python
def compute_exploration_uncertainty(self, cls_features, normalize=True):
    """
    步骤：
    1. 计算到所有labeled样本的距离
    2. 取最小距离（最近的labeled样本）
    3. 归一化
    """
    # 核心逻辑
    distances = self.memory_bank.compute_knn_distances(
        cls_features, k=1
    )  # [B, 1] 最近邻距离
    
    exploration_uncertainty = distances.squeeze(1)  # [B]
    # 距离大 → 远离labeled → 探索新区域 ✅
    # 距离小 → 靠近labeled → 已知区域 ❌
```

**可视化：**
```
特征空间
┌──────────────────────────┐
│  ●labeled  ●  ●           │
│    ●  ●  ●                │
│                           │
│           A (dist=0.8)    │  ← 高探索不确定性
│                           │
│  ●  ●                     │
│    ● B (dist=0.1)         │  ← 低探索不确定性
│      ●  ●                 │
└──────────────────────────┘
```

##### 4.3 Boundary Uncertainty (88-108行)

```python
def compute_boundary_uncertainty(self, cls_features, normalize=True):
    """
    步骤：
    1. 找K个最近邻的标签
    2. 统计标签分布
    3. 计算熵或不一致性
    """
    # 核心逻辑
    knn_labels = self.memory_bank.compute_knn_labels(
        cls_features, k=self.k
    )  # [B, K]
    
    # 统计每个标签的出现次数
    max_label_count = mode(knn_labels, dim=1)  # 最多的标签数
    boundary_uncertainty = 1 - (max_label_count / K)
    
    # max_count = 10 → 标签一致 → 确定区域 → u = 0
    # max_count = 5 → 标签混乱 → 边界区域 → u = 0.5
```

**示例：**
```
样本C的10个邻居标签：[猫, 猫, 猫, 猫, 猫, 猫, 猫, 猫, 猫, 猫]
→ max_count = 10 → u = 0 → 确定是猫 ❌

样本D的10个邻居标签：[猫, 狗, 猫, 狗, 猫, 狗, 猫, 狗, 猫, 狗]
→ max_count = 5 → u = 0.5 → 在猫狗边界上 ✅
```

##### 4.4 Multi-Scale Uncertainty (177-235行)

**最复杂的一个，需要重点理解！**

```python
def compute_multiscale_uncertainty(self, multi_layer_features, normalize=True):
    """
    步骤：
    1. 计算每层的exploration uncertainty
    2. 计算平均值和方差
    3. 公式: ū × (1 - Var)
    """
    # 核心逻辑
    layer_distances = self.memory_bank.compute_multi_layer_distances(
        multi_layer_features
    )  # {'layer_3': [B], 'layer_6': [B], ...}
    
    # 堆叠成矩阵
    uncertainties_per_layer = torch.stack([
        layer_distances['layer_3'],
        layer_distances['layer_6'],
        layer_distances['layer_9'],
        layer_distances['layer_11']
    ], dim=1)  # [B, 4]
    
    # 计算统计量
    mean_u = uncertainties_per_layer.mean(dim=1)  # [B]
    var_u = uncertainties_per_layer.var(dim=1)    # [B]
    
    # 公式
    multiscale_uncertainty = mean_u * (1 - var_u)
    # mean高 + var低 → 一致的高不确定性 → 整体复杂 ✅
    # mean低 或 var高 → 不一致或简单 → 低不确定性 ❌
```

**公式直觉：**
```
情况1：简单样本（清晰的猫）
Layer 3: 0.2 (纹理简单)
Layer 6: 0.1 (形状清晰)
Layer 9: 0.15 (语义明确)
Layer 11: 0.12 (全局简单)
→ mean = 0.14, var = 0.002
→ u = 0.14 × 0.998 = 0.14 ❌

情况2：复杂样本（猫狗重叠）
Layer 3: 0.8 (纹理混乱)
Layer 6: 0.75 (形状模糊)
Layer 9: 0.85 (语义复杂)
Layer 11: 0.78 (全局复杂)
→ mean = 0.795, var = 0.002
→ u = 0.795 × 0.998 = 0.79 ✅

情况3：不一致样本（需要过滤）
Layer 3: 0.9 (局部复杂)
Layer 6: 0.2 (中期简单)
Layer 9: 0.8 (高期复杂)
Layer 11: 0.1 (全局简单)
→ mean = 0.5, var = 0.15
→ u = 0.5 × 0.85 = 0.425 ⚠️
```

**动手练习：修改参数观察效果**
```python
# 修改K值
estimator = UncertaintyEstimator(memory_bank, k_neighbors=5)   # 少邻居
estimator = UncertaintyEstimator(memory_bank, k_neighbors=20)  # 多邻居

# 观察：K如何影响密度和边界计算？

# 修改normalize
uncertainties = estimator.compute_all_uncertainties(
    ..., normalize=False  # 不归一化，看原始值范围
)
```

---

<a name="file-selector"></a>
#### 📄 文件5：`active_learning/cascading_selector*.py` (3个版本)

**学习重点：**
1. 如何组合4种不确定性
2. 三种策略的优劣对比
3. 如何设计更好的选择策略

**三个版本对比：**

| 版本 | 文件 | 策略 | 优点 | 缺点 |
|-----|------|------|------|------|
| **V0** | `cascading_selector.py` | 短路过滤 | 明确逻辑 | 可能过滤有价值样本 |
| **V1** | `cascading_selector_v1.py` | 综合判断 | 信号叠加 | 参数多 |
| **V2** | `cascading_selector_v2.py` | 加权分数 | 灵活可调 | 权重敏感 |

**V0：短路过滤（Baseline）**
```python
def evaluate_sample(u_density, u_exploration, u_boundary, u_multiscale):
    # 第1关：密度过滤
    if u_density > 0.5:
        return Priority.NOISE  # 直接淘汰
    
    # 第2关：探索优先
    if u_exploration > 0.5:
        return Priority.HIGH
    
    # 第3关：边界重要
    if u_boundary > 0.6:
        return Priority.HIGH
    
    # 第4关：多尺度
    if u_multiscale > 0.5:
        return Priority.MEDIUM
    
    return Priority.LOW
```

**V1：综合判断（推荐）**
```python
def evaluate_sample(u_density, u_exploration, u_boundary, u_multiscale):
    # 计数有几个高信号
    high_signals = [
        u_exploration > 0.5,
        u_boundary > 0.6,
        u_multiscale > 0.5
    ].count(True)
    
    # 信号叠加
    if high_signals >= 3:
        return Priority.VERY_HIGH
    elif high_signals == 2:
        return Priority.HIGH
    elif high_signals == 1:
        if u_exploration > 0.5:
            return Priority.HIGH
        else:
            return Priority.MEDIUM
    
    # 密度作为后备
    if u_density < 0.5:
        return Priority.MEDIUM
    else:
        return Priority.LOW
```

**V2：加权分数**
```python
def evaluate_sample(u_density, u_exploration, u_boundary, u_multiscale):
    # 价值分数
    value_score = (
        0.4 * u_exploration +
        0.3 * u_boundary +
        0.3 * u_multiscale
    )
    
    # 密度作为置信度惩罚
    confidence = 1.0 - (u_density * 0.5)
    
    # 最终分数
    final_score = value_score * confidence
    
    # 分数映射到优先级
    if final_score > 0.7: return Priority.VERY_HIGH
    elif final_score > 0.5: return Priority.HIGH
    elif final_score > 0.3: return Priority.MEDIUM
    else: return Priority.LOW
```

**对比实验设计：**
```bash
# 运行三个实验
python scripts/run_active_learning.py --exp_name v0 --selector v0
python scripts/run_active_learning.py --exp_name v1 --selector v1
python scripts/run_active_learning.py --exp_name v2 --selector v2

# 对比结果
python scripts/compare_results.py --exps v0 v1 v2
```

---

### 📖 第4天：主循环与训练

<a name="file-classifier"></a>
#### 📄 文件6：`active_learning/classifier.py` (155行)

**学习重点：**
1. 简单分类头的设计
2. 为什么不用复杂网络
3. 训练和评估流程

**两种分类头：**
```python
# 1. 线性分类头（最简单）
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim=768, num_classes=100):
        self.fc = nn.Linear(768, 100)
    
    def forward(self, features):
        return self.fc(features)  # [B, 768] → [B, 100]

# 2. MLP分类头（稍复杂）
class MLPClassifier(nn.Module):
    def __init__(self, input_dim=768, num_classes=100, hidden_dim=512):
        self.fc1 = nn.Linear(768, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(512, 100)
    
    def forward(self, features):
        x = self.fc1(features)     # [B, 768] → [B, 512]
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)            # [B, 512] → [B, 100]
        return x
```

**为什么简单就够？**
- DINOv3特征已经很强（冻结骨干）
- 只需要学习"类别边界"，不需要学习"特征提取"
- 避免过拟合（标注样本少）

**训练流程：**
```python
def train_epoch(classifier, feature_extractor, dataloader, optimizer, device):
    for images, labels in dataloader:
        # 1. 提取特征（冻结）
        with torch.no_grad():
            features = feature_extractor(images, return_cls_only=True)
        
        # 2. 分类（可训练）
        logits = classifier(features)
        
        # 3. 计算损失
        loss = F.cross_entropy(logits, labels)
        
        # 4. 更新参数（只更新分类头）
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

<a name="file-active-learner"></a>
#### 📄 文件7：`active_learning/active_learner.py` (446行) ⭐⭐

**学习重点：**
1. 主动学习的完整循环
2. 如何协调各个组件
3. 如何保存和恢复实验

**主循环结构：**
```python
class ActiveLearner:
    def run_one_round(self, round_num, ...):
        # 1. 训练分类头（用已标注样本）
        labeled_loader = dataset.get_labeled_loader()
        self.train_classifier(labeled_loader)
        
        # 2. 评估当前性能
        val_accuracy = self.evaluate(val_loader)
        
        # 3. 选择新样本
        unlabeled_loader = dataset.get_unlabeled_loader()
        selected_indices = self.select_samples(unlabeled_loader, budget)
        
        # 4. 标注样本（移动到labeled pool）
        dataset.add_labeled_samples(selected_indices)
        
        # 5. 更新Memory Bank
        self.update_memory_bank(labeled_loader)
        
        # 6. 记录历史
        self.history['rounds'].append(round_num)
        self.history['val_accuracy'].append(val_accuracy)
```

**关键方法解析：**

##### 7.1 `extract_features_from_loader` (108-171行)
```python
def extract_features_from_loader(self, dataloader, desc="提取特征"):
    """从DataLoader批量提取特征"""
    all_cls_features = []
    all_multi_layer_features = {}
    
    for batch in tqdm(dataloader, desc=desc):
        images, labels = batch
        images = images.to(device)
        
        # 提取特征（使用池化节省内存）
        features = self.feature_extractor(
            images, 
            return_cls_only=False, 
            pool_patches=True  # 重要优化！
        )
        
        # 收集特征（移到CPU节省GPU显存）
        all_cls_features.append(features['cls'].cpu())
        ...
    
    # 合并所有batch
    return {
        'cls': torch.cat(all_cls_features, dim=0),
        'layer_3': torch.cat(all_layer3_features, dim=0),
        ...
    }
```

##### 7.2 `select_samples` (173-214行)
```python
def select_samples(self, unlabeled_loader, budget):
    """从unlabeled pool选择样本"""
    # 1. 提取特征
    unlabeled_features = self.extract_features_from_loader(
        unlabeled_loader, desc="提取未标注样本"
    )
    
    # 2. 计算不确定性
    uncertainties = self.uncertainty_estimator.compute_all_uncertainties(
        cls_features=unlabeled_features['cls'],
        multi_layer_features={k: v for k, v in unlabeled_features.items() if k != 'cls'}
    )
    
    # 3. 级联选择
    selected_indices, priorities = self.selector.select_samples(
        uncertainties, budget
    )
    
    return selected_indices, priorities
```

##### 7.3 `update_memory_bank` (216-240行)
```python
def update_memory_bank(self, labeled_loader):
    """更新Memory Bank"""
    # 提取labeled样本的特征
    labeled_features = self.extract_features_from_loader(
        labeled_loader, desc="提取已标注样本"
    )
    
    # 添加到Memory Bank
    self.memory_bank.add_samples(
        cls_features=labeled_features['cls'],
        multi_layer_features={...},
        labels=labeled_features['labels']
    )
```

**完整流程图：**
```
[Round 1]
├─ 训练: 200张labeled → 分类头权重更新
├─ 评估: Val准确率 40.9%
├─ 提取特征: 101,147张unlabeled (6-7分钟)
├─ 计算不确定性: 4种不确定性
├─ 选择样本: 挑出50张
├─ 标注: 移到labeled pool
└─ 更新: Memory Bank加入50张新特征

[Round 2]
├─ 训练: 250张labeled → 分类头权重更新
├─ 评估: Val准确率 37.8%
├─ 提取特征: 101,097张unlabeled
├─ 计算不确定性: 4种不确定性
├─ 选择样本: 挑出50张
├─ 标注: 移到labeled pool
└─ 更新: Memory Bank加入50张新特征

[Round 3]
...
```

---

<a name="file-run-script"></a>
#### 📄 文件8：`scripts/run_active_learning.py` (350行) ⭐

**学习重点：**
1. 如何整合所有组件
2. 参数如何传递
3. 实验如何记录

**脚本结构：**
```python
def main(args):
    # 0. 设置日志
    logger = setup_logger(args.exp_name)
    
    # 1. 加载数据集
    train_dataset, val_dataset = load_imagenet100(args.data_root)
    
    # 2. 创建初始labeled pool
    initial_indices = create_initial_labeled_pool(
        train_dataset, 
        num_samples_per_class=args.init_samples_per_class
    )
    al_dataset = ActiveLearningDataset(train_dataset, initial_indices)
    
    # 3. 创建模型组件
    feature_extractor = MultiLayerDINOv3(pretrained=True)
    classifier = SimpleClassifier(num_classes=100)
    memory_bank = MemoryBank(feature_dim=768, num_classes=100)
    uncertainty_estimator = UncertaintyEstimator(memory_bank)
    selector = CascadingSelectorV1()
    
    # 4. 创建主动学习器
    active_learner = ActiveLearner(
        feature_extractor, classifier, memory_bank,
        uncertainty_estimator, selector
    )
    
    # 5. 运行主动学习循环
    for round_num in range(1, args.num_rounds + 1):
        metrics = active_learner.run_one_round(
            round_num, al_dataset, val_loader,
            budget=args.budget_per_round,
            train_epochs=args.train_epochs
        )
        logger.info(f"Round {round_num}: Val Acc = {metrics['val_accuracy']:.2f}%")
    
    # 6. 保存结果
    active_learner.save_history(f"results/{args.exp_name}/history.json")
    active_learner.save_checkpoint(f"results/{args.exp_name}/checkpoint.pth")
```

**参数说明：**
```python
# 数据参数
--data_root: 数据集路径
--batch_size: 批大小（影响内存和速度）
--num_workers: 数据加载线程数

# 主动学习参数
--num_rounds: 总轮数（10-20）
--budget_per_round: 每轮选择样本数（100-200）
--init_samples_per_class: 初始每类样本数（2）
--k_neighbors: KNN的K值（10）

# 不确定性阈值
--density_threshold: 密度阈值（0.5）
--exploration_threshold: 探索阈值（0.5）
--boundary_threshold: 边界阈值（0.6）
--multiscale_threshold: 多尺度阈值（0.5）

# 训练参数
--train_epochs: 每轮训练epoch数（10-20）
--learning_rate: 学习率（0.001）
```

**运行示例：**
```bash
# 快速测试
python scripts/run_active_learning.py \
    --exp_name quick_test \
    --num_rounds 3 \
    --budget_per_round 50 \
    --train_epochs 5 \
    --batch_size 4

# 完整实验
python scripts/run_active_learning.py \
    --exp_name exp_full \
    --num_rounds 20 \
    --budget_per_round 200 \
    --train_epochs 20 \
    --batch_size 4
```

---

## 6. 关键代码解析

### 6.1 特征提取的优化历程

**问题演进：**

#### 版本1：原始实现（OOM）
```python
# ❌ 问题：占用320GB内存
def forward(self, images):
    features = backbone.get_intermediate_layers(images, n=[3,6,9,11])
    # features['layer_3']: [B, 257, 768] = 保留所有patch
    return features

# 提取101,147张图：
# 4层 × 101,147 × 257 × 768 × 4字节 = 320GB ❌
```

#### 版本2：添加池化选项
```python
# ✅ 解决：可选池化
def forward(self, images, pool_patches=False):
    features = backbone.get_intermediate_layers(images, n=[3,6,9,11])
    
    if pool_patches:
        # 平均池化: [B, 257, 768] → [B, 768]
        for key in features:
            if key != 'cls':
                features[key] = features[key].mean(dim=1)
    
    return features

# 提取101,147张图：
# 4层 × 101,147 × 768 × 4字节 = 1.2GB ✅
```

### 6.2 Memory Bank的KNN优化

**挑战：** 如何快速计算101,147个query对250个memory的KNN？

**方案：矩阵乘法 + TopK**
```python
# 朴素方法：O(N×M×D) = 慢
for query in queries:  # N=101,147
    for memory in memories:  # M=250
        distance = compute_distance(query, memory)  # D=768
    topk = sort(distances)[:K]

# 优化方法：O(N×M + N×K×logK) = 快
# 1. 批量计算相似度矩阵
query: [N, D] = [101147, 768]
memory: [M, D] = [250, 768]
similarity = query @ memory.T  # [N, M] = [101147, 250]
# 单次矩阵乘法，GPU加速！

# 2. TopK（每行独立）
topk_sim, topk_idx = torch.topk(similarity, k=10, dim=1)
# [101147, 10] 只需要O(N×K×logK)
```

**实际性能：**
- 朴素方法：~10分钟
- 优化方法：~5秒（120倍加速）

### 6.3 不确定性归一化的重要性

**为什么要归一化？**

```python
# ❌ 不归一化：尺度不同，无法比较
u_density = [0.001, 0.003, 0.002, ...]      # 范围：0-0.01
u_exploration = [0.5, 0.8, 0.3, ...]        # 范围：0-1
u_boundary = [0.2, 0.9, 0.5, ...]           # 范围：0-1
u_multiscale = [0.6, 0.7, 0.8, ...]         # 范围：0-1
# 如果直接相加，density几乎不起作用！

# ✅ 归一化：都在[0, 1]范围
u_density = (u - u.min()) / (u.max() - u.min())
# 现在可以公平比较和组合
```

### 6.4 Cascading Selector的设计哲学

**为什么不直接加权平均？**

```python
# ❌ 简单加权：可能选到噪声
final_score = 0.25 * u_density + 0.25 * u_exploration + ...
# 即使density很高（噪声），但其他不确定性高可能也被选中

# ✅ 级联过滤：先排除噪声
if u_density > threshold:
    return NOISE  # 直接淘汰
# 然后再考虑其他不确定性
```

**级联 vs 并联：**

| 方法 | 逻辑 | 优点 | 缺点 |
|-----|------|------|------|
| **并联（加权）** | 所有信号同等重要 | 灵活 | 噪声敏感 |
| **级联（短路）** | 有优先级顺序 | 鲁棒 | 可能过滤有价值样本 |
| **混合（V1/V2）** | 综合判断 | 平衡 | 参数多 |

---

## 7. DINOv3源码学习

### 7.1 Vision Transformer基础

**ViT架构图：**
```
输入图片 [224×224×3]
    ↓
┌────────────────────────┐
│ Patch Embedding        │
│ - 16×16 patches        │
│ - Linear projection    │
└────────────────────────┘
    ↓
[196 patches] + [1 CLS token]
    ↓
┌────────────────────────┐
│ Position Embedding     │
│ - Learnable位置编码    │
└────────────────────────┘
    ↓
┌────────────────────────┐
│ Transformer Block × 12 │
│  ├─ Layer Norm         │
│  ├─ Multi-Head Attention│
│  ├─ Layer Norm         │
│  └─ MLP                │
└────────────────────────┘
    ↓
[CLS Token] [768] → 分类
[Patch Tokens] [196×768] → 定位
```

### 7.2 关键文件导读

#### 📄 `dinov3/models/vision_transformer.py`

**核心类：`DinoVisionTransformer`**

```python
class DinoVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, ...):
        # 1. Patch Embedding
        self.patch_embed = PatchEmbed(...)  # 图片→patches
        
        # 2. CLS Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 3. Position Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # 4. Transformer Blocks
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=12, ...)
            for _ in range(depth)  # depth=12 for ViT-Base
        ])
        
        # 5. Normalization
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # [B, 3, 224, 224] → [B, 196, 768]
        
        # 添加CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, 768]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 197, 768]
        
        # 添加位置编码
        x = x + self.pos_embed
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Normalization
        x = self.norm(x)
        return x
```

**学习要点：**
1. **为什么是16×16 patch？** → 平衡计算量和精度
2. **CLS token的作用？** → 聚合全局信息
3. **Position embedding为什么重要？** → Transformer没有位置信息

#### 📄 `dinov3/layers/attention.py`

**Multi-Head Self-Attention实现**

```python
class Attention(nn.Module):
    def forward(self, x):
        B, N, C = x.shape  # [B, 197, 768]
        
        # 1. 计算Q, K, V
        qkv = self.qkv(x)  # [B, 197, 2304] (768×3)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        # q: [B, 12, 197, 64] (12 heads, 64 dim per head)
        
        # 2. Scaled Dot-Product Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn: [B, 12, 197, 197] - 每个token对所有token的注意力
        
        attn = attn.softmax(dim=-1)
        
        # 3. 加权求和
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # x: [B, 197, 768]
        
        return x
```

**直觉理解：**
```
Query (我是谁): "我是CLS token"
Key (你们是谁): ["我是patch1", "我是patch2", ...]
Value (信息): [patch1的特征, patch2的特征, ...]

Attention (注意力权重):
CLS关注patch1: 0.1
CLS关注patch2: 0.3
CLS关注patch3: 0.05
...

输出 = 0.1 × patch1 + 0.3 × patch2 + 0.05 × patch3 + ...
```

### 7.3 get_intermediate_layers详解

**我们项目中最重要的接口！**

```python
# 在 vision_transformer.py 中
def get_intermediate_layers(
    self, 
    x, 
    n=[3, 6, 9, 11],  # 要提取的层
    reshape=False,     # 是否reshape成HxW
    return_class_token=True,
    norm=True
):
    """
    提取中间层特征
    
    Returns:
        如果n=[3,6,9,11]，返回4个特征：
        [
            layer_3_output: [B, 197, 768],
            layer_6_output: [B, 197, 768],
            layer_9_output: [B, 197, 768],
            layer_11_output: [B, 197, 768]
        ]
    """
    # Patch embedding
    x = self.patch_embed(x)
    x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
    x = x + self.pos_embed
    
    output = []
    for i, blk in enumerate(self.blocks):
        x = blk(x)
        if i in n:
            # 提取这一层的输出
            output.append(self.norm(x) if norm else x)
    
    return output
```

**为什么选[3, 6, 9, 11]层？**
```
Layer 0-2:  低层特征（边缘、纹理）
Layer 3:    早期语义（形状、颜色块）✅
Layer 4-5:  过渡层
Layer 6:    中期语义（部件、结构）✅
Layer 7-8:  过渡层
Layer 9:    高期语义（物体、场景）✅
Layer 10:   过渡层
Layer 11:   最终语义（全局理解）✅
```

---

## 8. 实验与调优

### 8.1 实验设计

#### 实验1：Baseline对比

**目的：** 证明主动学习优于随机采样

```bash
# 1. 主动学习（您的方法）
python scripts/run_active_learning.py \
    --exp_name exp_active \
    --selector v1

# 2. 随机采样
python scripts/run_active_learning.py \
    --exp_name exp_random \
    --selector random

# 3. 对比
python scripts/compare_results.py --exps exp_active exp_random
```

**预期结果：**
```
标注量: 1,200张 (1.18%)

随机采样: 50-55% 准确率
主动学习: 55-65% 准确率 ← 应该更好
```

#### 实验2：Selector对比

**目的：** 找到最好的样本选择策略

```bash
# V0: 短路过滤
python scripts/run_active_learning.py --exp_name v0 --selector v0

# V1: 综合判断
python scripts/run_active_learning.py --exp_name v1 --selector v1

# V2: 加权分数
python scripts/run_active_learning.py --exp_name v2 --selector v2
```

**对比维度：**
- 最终准确率
- 样本多样性
- 类别平衡性
- 噪声过滤效果

#### 实验3：不确定性消融实验

**目的：** 验证每种不确定性的贡献

```bash
# 只用Exploration
python scripts/run_active_learning.py --exp_name only_exploration \
    --density_weight 0 --exploration_weight 1 --boundary_weight 0 --multiscale_weight 0

# 只用Boundary
python scripts/run_active_learning.py --exp_name only_boundary \
    --density_weight 0 --exploration_weight 0 --boundary_weight 1 --multiscale_weight 0

# 只用Multi-Scale
python scripts/run_active_learning.py --exp_name only_multiscale \
    --density_weight 0 --exploration_weight 0 --boundary_weight 0 --multiscale_weight 1

# 全部组合
python scripts/run_active_learning.py --exp_name all_combined \
    --density_weight 0.2 --exploration_weight 0.3 --boundary_weight 0.3 --multiscale_weight 0.2
```

### 8.2 参数调优

#### 调优1：K近邻数量

**默认：K=10**

```bash
# 测试不同K值
for K in 5 10 15 20; do
    python scripts/run_active_learning.py \
        --exp_name k${K} \
        --k_neighbors $K
done

# 观察：K如何影响密度和边界计算
```

**K值的影响：**
- K太小（5）：噪声敏感，边界不稳定
- K适中（10）：平衡
- K太大（20）：过于平滑，边界模糊

#### 调优2：阈值调整

**密度阈值（density_threshold）**
```bash
# 保守（过滤更多）
--density_threshold 0.3

# 中等（默认）
--density_threshold 0.5

# 激进（过滤更少）
--density_threshold 0.7
```

**观察指标：**
- 选中样本的质量
- 噪声比例
- 类别分布

#### 调优3：训练参数

**学习率调度：**
```python
# 固定学习率（简单）
optimizer = Adam(classifier.parameters(), lr=0.001)

# 学习率衰减（更好）
optimizer = Adam(classifier.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
# Epoch 0-4:  lr = 0.001
# Epoch 5-9:  lr = 0.0005
# Epoch 10-14: lr = 0.00025
```

**早停（Early Stopping）：**
```python
best_val_acc = 0
patience = 5
no_improve = 0

for epoch in range(epochs):
    train_loss = train_epoch(...)
    val_acc = evaluate(...)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        no_improve = 0
        save_checkpoint(...)
    else:
        no_improve += 1
        if no_improve >= patience:
            print("Early stopping!")
            break
```

### 8.3 后续优化方向

#### 优化1：加入预测不确定性

**当前：只用特征不确定性**

```python
# 现在的做法
uncertainties = estimator.compute_all_uncertainties(features)
selected = selector.select_samples(uncertainties, budget)
```

**优化：融合预测不确定性**

```python
# 1. 提取特征
features = feature_extractor(unlabeled_images)

# 2. 特征不确定性（当前方法）
feature_uncertainty = estimator.compute_all_uncertainties(features)

# 3. 预测不确定性（新增）
with torch.no_grad():
    logits = classifier(features['cls'])
    probs = F.softmax(logits, dim=1)
    
    # 方法1：熵（Entropy）
    entropy = -(probs * probs.log()).sum(dim=1)
    pred_uncertainty = entropy / math.log(num_classes)  # 归一化
    
    # 方法2：最大概率（Confidence）
    max_prob, _ = probs.max(dim=1)
    pred_uncertainty = 1 - max_prob
    
    # 方法3：Margin（Top2差距）
    top2_probs, _ = torch.topk(probs, k=2, dim=1)
    margin = top2_probs[:, 0] - top2_probs[:, 1]
    pred_uncertainty = 1 - margin

# 4. 融合
final_uncertainty = (
    0.7 * feature_uncertainty['combined'] +
    0.3 * pred_uncertainty
)

# 5. 选择样本
selected = selector.select_samples({'combined': final_uncertainty}, budget)
```

**何时用预测不确定性？**
- 后期轮次（分类器已较准）
- 类别不平衡时
- 与特征不确定性互补

#### 优化2：动态阈值调整

**当前：固定阈值**
```python
density_threshold = 0.5  # 固定
```

**优化：根据轮次动态调整**

```python
def get_dynamic_threshold(round_num, total_rounds):
    """
    早期：保守（高阈值），过滤噪声
    后期：激进（低阈值），尽量选样本
    """
    # 线性衰减
    alpha = round_num / total_rounds
    density_threshold = 0.7 - 0.3 * alpha
    # Round 1:  0.7 (严格过滤)
    # Round 10: 0.4 (宽松过滤)
    
    return density_threshold
```

#### 优化3：类别平衡采样

**当前问题：** 可能选出的样本类别不均衡

```python
# 当前：纯不确定性选择
selected = topk_samples(uncertainties, k=100)
# 可能：猫×50, 狗×30, 鸟×20（不均衡）
```

**优化：强制类别平衡**

```python
def balanced_select(uncertainties, labels, budget, num_classes):
    """每个类别选择 budget/num_classes 个样本"""
    samples_per_class = budget // num_classes
    selected = []
    
    for class_id in range(num_classes):
        # 找出该类的样本
        class_mask = (labels == class_id)
        class_uncertainties = uncertainties[class_mask]
        
        # 从该类中选top-k
        topk = torch.topk(class_uncertainties, k=samples_per_class)
        selected.extend(topk.indices)
    
    return selected
```

#### 优化4：核心集方法（Coreset）

**核心思想：** 选出的样本应该"代表"整个分布

```python
def coreset_select(features, uncertainties, budget):
    """
    结合不确定性和多样性
    1. 先用不确定性筛选候选集（2×budget）
    2. 再用贪心算法选择多样的子集
    """
    # 1. 候选集
    candidates = topk_uncertain_samples(uncertainties, k=2*budget)
    candidate_features = features[candidates]
    
    # 2. 贪心选择（最大化距离）
    selected = []
    selected.append(candidates[0])  # 最不确定的
    
    for _ in range(budget - 1):
        # 计算每个候选到已选样本的最小距离
        distances = compute_min_distances(
            candidate_features, 
            features[selected]
        )
        # 选择距离最大的（最不相似的）
        next_sample = candidates[distances.argmax()]
        selected.append(next_sample)
    
    return selected
```

---

## 9. 常见问题

### Q1: 为什么验证准确率在第2轮下降？

**可能原因：**
1. **过拟合**：样本太少（200-300），分类头记住训练集
2. **分布偏移**：选出的样本不代表验证集分布
3. **随机性**：不同轮次的batch采样导致

**解决方法：**
```python
# 1. 增加正则化
classifier = SimpleClassifier(dropout=0.3)  # 增大dropout

# 2. 减少训练epoch
--train_epochs 5  # 避免过拟合

# 3. 使用数据增强
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4),  # 更强的增强
    ...
])

# 4. 多次运行取平均
for seed in [42, 43, 44]:
    python scripts/run_active_learning.py --seed $seed
```

### Q2: 如何判断Memory Bank是否工作正常？

**检查方法：**
```python
# 在 uncertainty.py 中添加调试代码
def compute_exploration_uncertainty(self, cls_features):
    if self.memory_bank.num_samples == 0:
        print("⚠️ Memory Bank为空！")
    else:
        print(f"✓ Memory Bank: {self.memory_bank.num_samples} 样本")
        print(f"  类别分布: {self.memory_bank.get_statistics()}")
    
    distances = self.memory_bank.compute_knn_distances(cls_features, k=1)
    print(f"  距离范围: [{distances.min():.4f}, {distances.max():.4f}]")
    
    return distances
```

**正常输出：**
```
✓ Memory Bank: 250 样本
  类别分布: {0: 3, 1: 2, 2: 3, ..., 99: 2}
  距离范围: [0.0521, 0.9876]
```

**异常情况：**
```
⚠️ 问题1：Memory Bank为空
→ 检查 update_memory_bank 是否被调用

⚠️ 问题2：距离范围异常 [0.0, 0.0]
→ 特征没有归一化

⚠️ 问题3：类别分布极度不均 {0: 200, 1: 0, ...}
→ 样本选择有问题
```

### Q3: 显存/内存不够怎么办？

**问题诊断：**
```bash
# 查看显存
nvidia-smi

# 查看内存
free -h

# 查看进程占用
htop
```

**解决方案：**

| 资源 | 优化方法 | 效果 |
|-----|---------|------|
| **GPU显存** | `batch_size=4→2` | 显存减半 |
| | `pool_patches=True` | 已启用 ✅ |
| | `torch.cuda.empty_cache()` | 清理碎片 |
| **CPU内存** | `num_workers=2→1` | 内存减半 |
| | 减少提取层数 `--layers 9 11` | 内存减半 |
| | 分批提取特征 | 见下方 |

**分批提取特征（终极方案）：**
```python
def extract_features_in_chunks(self, dataloader, chunk_size=10000):
    """分批提取，避免一次性加载全部"""
    all_features = []
    
    chunk_buffer = []
    for batch in dataloader:
        chunk_buffer.append(batch)
        
        if len(chunk_buffer) >= chunk_size:
            # 处理一个chunk
            features = self._extract_chunk(chunk_buffer)
            all_features.append(features)
            
            # 清空buffer和GPU缓存
            chunk_buffer = []
            torch.cuda.empty_cache()
    
    # 处理最后的chunk
    if chunk_buffer:
        features = self._extract_chunk(chunk_buffer)
        all_features.append(features)
    
    # 合并所有chunk
    return self._merge_features(all_features)
```

### Q4: 如何理解不同层的特征？

**可视化方法：**
```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_layer_features(multi_layer_features, labels):
    """
    用PCA降维到2D，可视化不同层的特征分布
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    for i, (layer_name, features) in enumerate(multi_layer_features.items()):
        if layer_name == 'cls':
            continue
        
        # PCA降维
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features.cpu().numpy())
        
        # 绘制
        ax = axes[i]
        scatter = ax.scatter(
            features_2d[:, 0], 
            features_2d[:, 1], 
            c=labels.cpu().numpy(), 
            cmap='tab10',
            alpha=0.6
        )
        ax.set_title(f'{layer_name}')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
    
    plt.colorbar(scatter, ax=axes[-1])
    plt.tight_layout()
    plt.savefig('layer_features_visualization.png')
    plt.close()
```

**观察：**
- Layer 3：特征分散，类别混乱（低层）
- Layer 6：开始聚类，但有重叠（中层）
- Layer 9：聚类明显，边界清晰（高层）
- Layer 11：聚类紧密，分离良好（最终层）

### Q5: 如何复现实验结果？

**固定随机性：**
```python
import torch
import numpy as np
import random

def set_seed(seed=42):
    """固定所有随机种子"""
    # Python
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 在脚本开头调用
set_seed(42)
```

**记录实验：**
```python
experiment_config = {
    'seed': args.seed,
    'model': 'DINOv3-ViT-Base/16',
    'dataset': 'ImageNet-100',
    'num_rounds': args.num_rounds,
    'budget_per_round': args.budget_per_round,
    'selector': args.selector,
    'k_neighbors': args.k_neighbors,
    'thresholds': {
        'density': args.density_threshold,
        'exploration': args.exploration_threshold,
        'boundary': args.boundary_threshold,
        'multiscale': args.multiscale_threshold
    },
    'training': {
        'epochs': args.train_epochs,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size
    }
}

# 保存到实验目录
with open(f'results/{args.exp_name}/config.json', 'w') as f:
    json.dump(experiment_config, f, indent=2)
```

---

## 附录：学习资源

### 论文推荐

#### 主动学习
1. **Active Learning Literature Survey** (2009)
   - 主动学习综述，入门必读
2. **Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds** (ICLR 2020)
   - 深度学习时代的主动学习
3. **Learning Loss for Active Learning** (CVPR 2019)
   - 学习预测损失作为不确定性

#### Vision Transformer
1. **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale** (ICLR 2021)
   - ViT原论文
2. **DINOv2: Learning Robust Visual Features without Supervision** (2023)
   - DINOv2/v3原论文

#### 不确定性估计
1. **Uncertainty in Deep Learning** (PhD Thesis, Yarin Gal)
   - 不确定性估计的理论基础

### 在线资源

- **DINOv3 GitHub**: https://github.com/facebookresearch/dinov2
- **PyTorch官方教程**: https://pytorch.org/tutorials/
- **主动学习工具库**: https://modal-python.readthedocs.io/

### 学习进度检查表

**第1周：基础理解**
- [ ] 阅读项目README
- [ ] 理解主动学习概念
- [ ] 了解DINOv3架构
- [ ] 运行quick_test实验

**第2周：代码深入**
- [ ] 阅读dataset.py，理解数据流
- [ ] 阅读feature_extractor.py，理解特征提取
- [ ] 阅读memory_bank.py，理解KNN计算
- [ ] 阅读uncertainty.py，理解4种不确定性

**第3周：实验与优化**
- [ ] 运行完整实验
- [ ] 对比不同selector
- [ ] 调整参数观察效果
- [ ] 实现一个新的不确定性度量

**第4周：论文撰写**
- [ ] 整理实验结果
- [ ] 绘制对比图表
- [ ] 撰写方法部分
- [ ] 撰写实验部分

---

## 结语

**学习建议：**

1. **循序渐进**：按照学习路线图一步步来，不要跳跃
2. **动手实践**：每读完一个文件就运行相关代码
3. **理解原理**：不要只记公式，要理解为什么这样设计
4. **修改实验**：尝试修改参数，观察效果，加深理解
5. **记录问题**：遇到不懂的地方记下来，逐个攻克

**推荐学习节奏：**
- **每天2-3小时**，持续3-4周
- **第1周**：理解概念和数据流
- **第2周**：深入核心算法
- **第3周**：实验和调优
- **第4周**：总结和论文撰写

**遇到困难时：**
1. 先查看本学习指南的相关部分
2. 阅读代码注释和docstring
3. 运行测试代码理解行为
4. 在论文中寻找理论依据
5. 与同学或导师讨论

**祝您学习顺利，毕设成功！** 🎓✨

