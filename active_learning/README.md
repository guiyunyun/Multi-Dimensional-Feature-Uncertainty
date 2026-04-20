# 主动学习框架 - DINOv3 + ImageNet-100

基于冻结DINOv3骨干网络的主动学习框架，实现了4种互补的特征不确定性度量 + 预测不确定性融合（闭环模式），用最少的标注数据达到最好的分类性能。

---

## 项目结构

```
active_learning/
├── config.py                   # 配置中心（所有参数集中管理）
├── main.py                     # 实验入口（推荐，零参数运行）
├── feature_extractor.py        # DINOv3多层特征提取器（冻结）
├── memory_bank.py              # Memory Bank（已标注样本特征存储）
├── uncertainty.py              # 4种特征不确定性计算
├── prediction_uncertainty.py   # 预测不确定性 + 混合融合器
├── cascading_selector_v1.py    # V1选择器（综合判断+信号叠加）推荐
├── cascading_selector.py       # V0选择器（短路逻辑）
├── cascading_selector_v2.py    # V2选择器（加权分数）
├── classifier.py               # 分类头（可训练）
├── dataset.py                  # 数据加载和主动学习pool管理
├── active_learner.py           # 主动学习主循环
└── README.md                   # 本文档
```

---

## 核心创新

### 1. 4种特征不确定性

| 不确定性类型    | 公式                                       | 物理意义          |
| --------------- | ------------------------------------------ | ----------------- |
| **Exploration** | `u = 1 - max(sim)`                         | 探索未知区域      |
| **Boundary**    | `u = Entropy(KNN_labels) / log(k)`         | 决策边界样本      |
| **Density**     | `u = Std(KNN_similarities)`                | 稀疏区域/噪声过滤 |
| **Multi-Scale** | `u = mean(u_layers) × (1 - var(u_layers))` | 多层一致性        |

### 2. 闭环模式（预测不确定性融合）

```
                    ┌─────────────────────────────────────────┐
                    │           主动学习闭环                    │
                    └─────────────────────────────────────────┘
                                      │
       ┌──────────────────────────────┴──────────────────────────────┐
       ▼                                                              ▼
  特征不确定性                                                    预测不确定性
  (Feature-based)                                                (Prediction-based)
       │                                                              │
       │  ┌─ Exploration (探索)                                       │  ┌─ Entropy (熵)
       │  ├─ Boundary (边界)                                          │  ├─ Margin (间隔)
       │  ├─ Density (密度)                                           │  ├─ Confidence (置信度)
       │  └─ Multiscale (多尺度)                                      │  └─ Variance (方差)
       │                                                              │
       └──────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
                        动态注意力调制 (Attention Modulation)
                        ─────────────────────────────────────
                        预测不确定性高 → 增强 Exploration/Boundary
                        预测不确定性低 → 均等权重
                                  │
                                  ▼
                           最终不确定性
                                  │
                                  ▼
                        选择样本 → 标注 → 训练 → 循环
```

**融合策略（attention模式）**:
- 预测不确定性高（分类器困惑）→ 放大 Exploration 和 Boundary 权重
- 预测不确定性低（分类器确定）→ 均等权重

### 3. Cascading样本选择

#### V1（推荐）：综合判断 + 信号叠加
- 高价值信号可以"拯救"密度异常样本（类内/类间边界保留）
- 多个不确定性同时high时优先级更高（3个high > 2个high > 1个high）
- 只有在没有高价值信号时，密度高才判为噪声

#### 优先级定义
```
VERY_HIGH (4)  - 3个不确定性high，或 Exploration+Boundary 同时high
HIGH (3)       - 2个不确定性high，或单独 Exploration/Boundary high
MEDIUM (2)     - 只有 Multiscale high
LOW (1)        - 没有high信号，密度正常
NOISE (0)      - 只有密度high，没有其他高价值信号 → 过滤
```

---

## 快速开始

### 1. 环境准备

```bash
# 激活conda环境
conda activate dinov3

# 确保已安装依赖
pip install torch torchvision tqdm
```

### 2. 数据准备

```bash
# 数据集路径（需要修改 config.py 中的 data_root）
# 默认路径: /root/autodl-tmp/dinov3/data/ImageNet100

# 数据集结构:
ImageNet100/
├── train/
│   ├── n01440764/
│   ├── n01443537/
│   └── ... (100个类别)
└── val/
    ├── n01440764/
    └── ...
```

### 3. 运行实验

#### 推荐方式：使用 main.py（零参数）

```bash
cd dinov3-try/active_learning
python main.py
```

所有配置在 `config.py` 中修改，无需命令行参数。

#### 切换实验模式

在 `config.py` 中修改配置，或使用预设函数：

```python
# 在 config.py 末尾添加调用，或在 Python 中动态调用

# 1. 随机采样 baseline（最重要的对比实验）
set_random_sampling()

# 2. 闭环完整版（4种特征不确定性 + 预测不确定性）
set_closed_loop_full()

# 3. 闭环消融：只用 Exploration
set_closed_loop_E_only()

# 4. 闭环消融：E + Boundary
set_closed_loop_E_B()

# 5. 闭环消融：E + B + Multiscale
set_closed_loop_E_B_M()

# 6. 开环模式（纯特征不确定性，使用级联选择器）
set_open_loop()
```

---

## 配置说明 (config.py)

### 实验配置（每次实验前必须检查）

```python
# 实验名称（结果自动加时间戳）
exp_name = 'exp_closed_loop_full'

# 随机采样模式（baseline）
random_sampling = False  # True = 随机采样，False = 主动学习

# 参与融合的特征不确定性（消融实验核心配置）
active_feature_uncertainties = ['exploration', 'boundary', 'density', 'multiscale']
```

### 模型配置

```python
model_size = 'base'             # 'small'(384维) / 'base'(768维) / 'large'(1024维)
feature_dim = 768               # 自动根据 model_size 计算
feature_layers = [3, 6, 9, 11]  # 要提取的中间层
```

### 主动学习配置

```python
initial_samples_per_class = 2   # 初始每类样本数 → 200样本
budget_per_round = 200          # 每轮选择样本数
total_rounds = 20               # 总轮数
# → 最终标注: 200 + 200*20 = 4200样本 (4.14%)
```

### 不确定性阈值（用于级联选择器）

```python
exploration_threshold = 0.5     # 探索性阈值
boundary_threshold = 0.6        # 边界阈值
density_threshold = 0.5         # 密度阈值
multiscale_threshold = 0.5      # 多尺度阈值
```

### 闭环模式配置

```python
use_prediction_uncertainty = True   # True=闭环, False=开环
prediction_uncertainty_mode = 'entropy'  # 预测不确定性模式
fusion_strategy = 'attention'       # 融合策略
noise_threshold = 0.7               # 噪声过滤阈值
```

---

## 实验流程

```
1. 初始化
   ├─ 加载冻结的DINOv3-ViT-B/16（768维特征）
   ├─ 创建可训练的分类头（约77K参数）
   ├─ 创建Memory Bank（存储已标注样本特征）
   └─ 初始labeled pool（每类2个 → 200样本）

2. 主动学习循环（每轮）
   ├─ 训练分类头（50 epochs）
   │   └─ 只更新分类头参数，骨干冻结
   │
   ├─ 提取未标注样本特征
   │   ├─ CLS特征: [N, 768]
   │   └─ 多层特征: layer_3, layer_6, layer_9, layer_11
   │
   ├─ 计算4种特征不确定性
   │   ├─ Exploration: 距离最近已标注样本
   │   ├─ Boundary: KNN标签熵
   │   ├─ Density: KNN相似度标准差
   │   └─ Multi-Scale: 多层一致性
   │
   ├─ 计算预测不确定性（闭环模式）
   │   └─ Entropy（分类头输出的熵）
   │
   ├─ 动态注意力融合
   │   └─ 预测不确定性高 → 增强E/B权重
   │
   ├─ 噪声过滤 + Top-K选择
   │   └─ density > 0.7 的样本被视为噪声
   │
   └─ 更新Memory Bank
       └─ 添加新标注样本的特征和标签

3. 结果保存
   ├─ history.json   # 训练历史
   ├─ checkpoint.pth # 模型权重
   ├─ config.json    # 实验配置
   └─ log.txt        # 运行日志
```

---

## 实验结果

### 闭环完整版 (exp_closed_loop_attention)

| 轮次 | 标注样本数 | 标注比例  | 验证准确率 |
| ---- | ---------- | --------- | ---------- |
| 1    | 400        | 0.39%     | 46.92%     |
| 5    | 1,200      | 1.18%     | 70.82%     |
| 10   | 2,200      | 2.17%     | 77.58%     |
| 15   | 3,200      | 3.16%     | 80.38%     |
| 20   | 4,200      | **4.14%** | **82.83%** |

**关键指标**:
- 使用 **4.14%** 数据达到 **82.83%** 准确率
- 超过预期目标（73-78%）
- 第16-19轮趋于饱和（收益递减）

### 预期对比（待验证）

| 方法                       | 10%标注  | 20%标注  | 30%标注  |
| -------------------------- | -------- | -------- | -------- |
| **Random Sampling**        | ~45%     | ~55%     | ~62%     |
| **开环（纯特征不确定性）** | ~48%     | ~58%     | ~65%     |
| **闭环完整版**             | **~52%** | **~62%** | **~68%** |

---

## 消融实验设计

### 实验列表

| 编号 | 实验名称             | 特征不确定性           | 预测不确定性 | 目的                             |
| ---- | -------------------- | ---------------------- | ------------ | -------------------------------- |
| 0    | Random Sampling      | 无                     | 无           | **最重要基线**，证明主动学习有效 |
| 1    | 闭环 + E only        | Exploration            | entropy      | 最基本的特征不确定性             |
| 2    | 闭环 + E + B         | Exploration + Boundary | entropy      | 加上边界信号                     |
| 3    | 闭环 + E + B + M     | E + B + Multiscale     | entropy      | 加上多尺度                       |
| 4    | 闭环 + E + B + D + M | 全部4种（完整版）      | entropy      | **已完成，82.83%**               |

### 预期结论

| 对比             | 证明什么                                     |
| ---------------- | -------------------------------------------- |
| Random vs E only | 主动学习有效（Exploration能找到有价值样本）  |
| E vs E+B         | Boundary信号互补（找到决策边界附近的难样本） |
| E+B vs E+B+M     | Multiscale提供多层一致性信息                 |
| E+B+M vs E+B+D+M | Density过滤噪声，提高标注效率                |

---

## 测试各组件

```bash
# 测试特征提取器
python feature_extractor.py

# 测试Memory Bank
python memory_bank.py

# 测试不确定性计算
python uncertainty.py

# 测试预测不确定性
python prediction_uncertainty.py

# 测试级联选择器
python cascading_selector_v1.py

# 测试分类头
python classifier.py

# 测试数据加载器
python dataset.py

# 测试主动学习器（不含真实数据）
python -m active_learning.active_learner

# 打印当前配置
python config.py
```

---

## 代码使用示例

### 基础使用

```python
from config import Config
from feature_extractor import MultiLayerDINOv3
from classifier import SimpleClassifier
from memory_bank import MemoryBank
from uncertainty import UncertaintyEstimator
from cascading_selector_v1 import CascadingSelectorV1
from active_learner import ActiveLearner

# 1. 创建组件（所有参数从Config读取）
feature_extractor = MultiLayerDINOv3(pretrained=True)
classifier = SimpleClassifier()
memory_bank = MemoryBank()
uncertainty_estimator = UncertaintyEstimator(memory_bank)
selector = CascadingSelectorV1()

# 2. 创建主动学习器
learner = ActiveLearner(
    feature_extractor=feature_extractor,
    classifier=classifier,
    memory_bank=memory_bank,
    uncertainty_estimator=uncertainty_estimator,
    selector=selector
)

# 3. 运行一轮
metrics = learner.run_one_round(
    round_num=1,
    al_dataset=al_dataset,
    val_loader=val_loader,
    budget=Config.budget_per_round
)
```

### 单独计算不确定性

```python
# 计算4种特征不确定性
uncertainties = uncertainty_estimator.compute_all_uncertainties(
    cls_features=features['cls'],
    multi_layer_features={
        'layer_3': features['layer_3'],
        'layer_6': features['layer_6'],
        'layer_9': features['layer_9'],
        'layer_11': features['layer_11']
    },
    normalize=True
)

print(f"Exploration: {uncertainties['exploration']}")
print(f"Boundary: {uncertainties['boundary']}")
print(f"Density: {uncertainties['density']}")
print(f"Multiscale: {uncertainties['multiscale']}")
```

---

## 文件路径说明

```
dinov3-try/
├── active_learning/           # 核心代码目录
│   ├── main.py               # 新入口（推荐）
│   └── config.py             # 配置中心
├── results/                   # 实验结果保存目录
│   └── {exp_name}_{timestamp}/
│       ├── history.json
│       ├── checkpoint.pth
│       ├── config.json
│       └── log.txt
├── pretrained_models/         # DINOv3预训练权重
│   ├── dinov3_vits16_pretrain_lvd1689m-08c60483.pth
│   ├── dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
│   └── dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
└── scripts/                   # 辅助脚本
    ├── run_active_learning.py # 旧入口（仍可用）
    └── view_results.py        # 查看结果
```

---

## 常见问题

### Q1: 数据集路径错误

```python
# 在 config.py 中修改
data_root = '/your/path/to/ImageNet100'
```

### Q2: GPU显存不足

```python
# 在 config.py 中修改
batch_size = 16  # 减小批大小
model_size = 'small'  # 使用更小的模型
```

### Q3: 训练速度慢

```python
# 在 config.py 中修改
classifier_epochs = 20  # 减少训练轮数
total_rounds = 10       # 减少主动学习轮数
```

### Q4: 如何切换开环/闭环模式

```python
# 闭环模式（默认）
use_prediction_uncertainty = True

# 开环模式
use_prediction_uncertainty = False
```

---

## 关键设计决策

### 1. 为什么冻结DINOv3？
- DINOv3在LVD-1689M上预训练，特征质量极高
- 避免过拟合（只训练约77K分类头参数）
- 计算高效（不需要反向传播到骨干网络）

### 2. 为什么用Std计算Density？
- 更直观：直接衡量邻居距离分散程度
- 更稳定：避免KDE的极值问题
- 无超参数：不需要bandwidth调参

### 3. 为什么用动态注意力融合？
- 利用分类头的"困惑信号"动态调整权重
- 分类器不确定时，更信任探索和边界信号
- 分类器确定时，权重均衡，不偏重某一方面

### 4. 为什么Density要做噪声过滤？
- 在闭环模式中，Density既参与融合，也单独做噪声过滤
- density > 0.7 的样本可能是离群点/噪声
- 防止标注低质量样本浪费预算

---

## TODO

- [x] DINOv3多层特征提取器
- [x] Memory Bank（KNN优化）
- [x] 4种特征不确定性计算
- [x] 预测不确定性 + 混合融合
- [x] 3种Cascading Selector（V0/V1/V2）
- [x] 分类头
- [x] 数据加载器
- [x] 主动学习主循环
- [x] 统一配置中心 (config.py)
- [x] 新入口文件 (main.py)
- [ ] Random Sampling baseline 实验
- [ ] 闭环消融实验（E → E+B → E+B+M → 完整版）
- [ ] 开环 vs 闭环对比实验
- [ ] 可视化分析

---

## 联系

如有问题，请查看代码注释或联系开发者。

---

**祝实验顺利！**
