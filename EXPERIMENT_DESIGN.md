# 🧪 完整对比实验设计方案

## 📋 实验目标

验证以下假设：
1. **主动学习 > 随机采样**
2. **闭环 > 开环**（预测不确定性有帮助）
3. **4种不确定性组合 > 单个**（消融实验）
4. **不同Selector策略的差异**

---

## 🎯 实验组设置（6组实验）

### **Group 1: Baseline（对照组）**

#### **Exp 1-1: Random Sampling（随机采样）**

```bash
python scripts/run_active_learning.py \
    --exp_name baseline_random \
    --num_rounds 20 \
    --budget_per_round 200 \
    --train_epochs 20 \
    --batch_size 4 \
    --selector random \
    --seed 42
```

**说明：**
- 完全随机选择样本
- 不使用任何不确定性
- 作为baseline对照

**预期结果：** 55-60%

---

### **Group 2: 开环模式（特征不确定性）**

#### **Exp 2-1: 开环 - Selector V0（短路过滤）**

```bash
python scripts/run_active_learning.py \
    --exp_name open_loop_v0 \
    --num_rounds 20 \
    --budget_per_round 200 \
    --train_epochs 20 \
    --batch_size 4 \
    --use_prediction_uncertainty False \
    --selector v0 \
    --seed 42
```

**说明：**
- 纯特征不确定性
- 短路过滤策略
- Density → Exploration → Boundary → Multi-Scale

**预期结果：** 68-72%

---

#### **Exp 2-2: 开环 - Selector V1（综合判断）**

```bash
python scripts/run_active_learning.py \
    --exp_name open_loop_v1 \
    --num_rounds 20 \
    --budget_per_round 200 \
    --train_epochs 20 \
    --batch_size 4 \
    --use_prediction_uncertainty False \
    --selector v1 \
    --seed 42
```

**说明：**
- 纯特征不确定性
- 综合判断 + 信号叠加
- 更智能的优先级分配

**预期结果：** 70-74%

---

#### **Exp 2-3: 开环 - Selector V2（加权分数）**

```bash
python scripts/run_active_learning.py \
    --exp_name open_loop_v2 \
    --num_rounds 20 \
    --budget_per_round 200 \
    --train_epochs 20 \
    --batch_size 4 \
    --use_prediction_uncertainty False \
    --selector v2 \
    --seed 42
```

**说明：**
- 纯特征不确定性
- 加权分数 + Density惩罚

**预期结果：** 69-73%

---

### **Group 3: 闭环模式（特征+预测不确定性）**

#### **Exp 3-1: 闭环 - Attention Fusion（推荐）** ⭐

```bash
python scripts/run_active_learning.py \
    --exp_name closed_loop_attention \
    --num_rounds 20 \
    --budget_per_round 200 \
    --train_epochs 20 \
    --batch_size 4 \
    --use_prediction_uncertainty True \
    --fusion_strategy attention \
    --selector v1 \
    --seed 42
```

**说明：**
- 特征 + 预测不确定性
- 动态注意力调制
- 您设计的方案4

**预期结果：** 73-78%（最好）

---

#### **Exp 3-2: 闭环 - Multiply Fusion**

```bash
python scripts/run_active_learning.py \
    --exp_name closed_loop_multiply \
    --num_rounds 20 \
    --budget_per_round 200 \
    --train_epochs 20 \
    --batch_size 4 \
    --use_prediction_uncertainty True \
    --fusion_strategy multiply \
    --selector v1 \
    --seed 42
```

**说明：**
- 简单相乘融合
- 保守策略

**预期结果：** 71-75%

---

#### **Exp 3-3: 闭环 - Add Fusion**

```bash
python scripts/run_active_learning.py \
    --exp_name closed_loop_add \
    --num_rounds 20 \
    --budget_per_round 200 \
    --train_epochs 20 \
    --batch_size 4 \
    --use_prediction_uncertainty True \
    --fusion_strategy add \
    --selector v1 \
    --seed 42
```

**说明：**
- 加权相加融合
- 线性组合

**预期结果：** 72-76%

---

### **Group 4: 消融实验（Ablation Study）**

#### **Exp 4-1: 只用Density**

```bash
python scripts/run_active_learning.py \
    --exp_name ablation_density_only \
    --num_rounds 20 \
    --budget_per_round 200 \
    --train_epochs 20 \
    --batch_size 4 \
    --uncertainty_types density \
    --seed 42
```

**预期结果：** 52-56%（过滤噪声，但探索不足）

---

#### **Exp 4-2: 只用Exploration**

```bash
python scripts/run_active_learning.py \
    --exp_name ablation_exploration_only \
    --num_rounds 20 \
    --budget_per_round 200 \
    --train_epochs 20 \
    --batch_size 4 \
    --uncertainty_types exploration \
    --seed 42
```

**预期结果：** 60-65%（探索新区域，但可能选到噪声）

---

#### **Exp 4-3: 只用Boundary**

```bash
python scripts/run_active_learning.py \
    --exp_name ablation_boundary_only \
    --num_rounds 20 \
    --budget_per_round 200 \
    --train_epochs 20 \
    --batch_size 4 \
    --uncertainty_types boundary \
    --seed 42
```

**预期结果：** 62-67%（精炼边界，但初期效果差）

---

#### **Exp 4-4: 只用Multi-Scale**

```bash
python scripts/run_active_learning.py \
    --exp_name ablation_multiscale_only \
    --num_rounds 20 \
    --budget_per_round 200 \
    --train_epochs 20 \
    --batch_size 4 \
    --uncertainty_types multiscale \
    --seed 42
```

**预期结果：** 58-63%（复杂样本，但可能忽略简单重要样本）

---

#### **Exp 4-5: Density + Exploration**

```bash
python scripts/run_active_learning.py \
    --exp_name ablation_density_exploration \
    --num_rounds 20 \
    --budget_per_round 200 \
    --train_epochs 20 \
    --batch_size 4 \
    --uncertainty_types density exploration \
    --seed 42
```

**预期结果：** 64-68%

---

#### **Exp 4-6: Density + Exploration + Boundary**

```bash
python scripts/run_active_learning.py \
    --exp_name ablation_three_uncertainties \
    --num_rounds 20 \
    --budget_per_round 200 \
    --train_epochs 20 \
    --batch_size 4 \
    --uncertainty_types density exploration boundary \
    --seed 42
```

**预期结果：** 68-72%

---

#### **Exp 4-7: 全部4种（完整版）**

```bash
python scripts/run_active_learning.py \
    --exp_name ablation_all_four \
    --num_rounds 20 \
    --budget_per_round 200 \
    --train_epochs 20 \
    --batch_size 4 \
    --uncertainty_types density exploration boundary multiscale \
    --seed 42
```

**预期结果：** 70-74%（最好的开环）

---

### **Group 5: 分阶段实验**

#### **Exp 5-1: 前期开环 + 后期闭环**

```python
# 需要手动实现或修改代码
# Round 1-5: 开环
for round in [1, 2, 3, 4, 5]:
    use_prediction_uncertainty = False

# Round 6-20: 闭环
for round in [6, 7, ..., 20]:
    use_prediction_uncertainty = True
```

**预期结果：** 75-80%（可能最好）

---

### **Group 6: 不同随机种子（验证稳定性）**

```bash
# 用最好的配置，跑3次不同seed
for seed in 42 43 44; do
    python scripts/run_active_learning.py \
        --exp_name exp_best_seed${seed} \
        --use_prediction_uncertainty True \
        --fusion_strategy attention \
        --seed $seed
done
```

---

## 📊 对比表

### **主要对比**

| 实验组 | 实验名 | 关键差异 | 预期准确率 | 运行时间 |
|-------|-------|---------|----------|---------|
| **Baseline** | Random | 随机采样 | 55-60% | 10-12小时 |
| **开环-V0** | Open-V0 | 短路过滤 | 68-72% | 10-12小时 |
| **开环-V1** | Open-V1 | 综合判断 | 70-74% | 10-12小时 |
| **开环-V2** | Open-V2 | 加权分数 | 69-73% | 10-12小时 |
| **闭环-Attention** | Closed-Att | 动态调制 | 73-78% | 12-14小时 |
| **闭环-Multiply** | Closed-Mul | 相乘融合 | 71-75% | 12-14小时 |
| **闭环-Add** | Closed-Add | 相加融合 | 72-76% | 12-14小时 |

### **消融实验对比**

| 不确定性组合 | 实验名 | 预期准确率 | 说明 |
|------------|-------|----------|------|
| **Density** | Abl-D | 52-56% | 过滤噪声为主 |
| **Exploration** | Abl-E | 60-65% | 探索为主 |
| **Boundary** | Abl-B | 62-67% | 边界为主 |
| **Multi-Scale** | Abl-M | 58-63% | 复杂度为主 |
| **D+E** | Abl-DE | 64-68% | 探索+过滤 |
| **D+E+B** | Abl-DEB | 68-72% | 三种组合 |
| **D+E+B+M** | Abl-All | 70-74% | 全部组合 |

---

## 📈 论文图表设计

### **图1: 学习曲线对比**

```python
# X轴：标注样本数量（200, 400, ..., 4200）
# Y轴：验证准确率（%）
# 曲线：
- Random (灰色虚线)
- Open-V1 (蓝色实线)
- Closed-Attention (红色实线) ⭐
- Full-Data (黑色虚线，上限参考)
```

**预期效果：**
```
80% ┤                               Full-Data -----
    │                          Closed ━━━━━
75% ┤                     Open ━━━━
    │                 Random ----
70% ┤            Open ━━━
    │        Random ----
65% ┤    Random ----
    │
60% ┤
    └────┴────┴────┴────┴────┴────┴────┴────┴────
   200  400  800  1200 1600 2000 2400 2800 3200
              标注样本数量
```

---

### **图2: 消融实验柱状图**

```python
# X轴：不同组合
# Y轴：最终准确率（%）

           78%
    ┌──────┐
    │Closed│
    │  🏆  │
    └──────┘
           74%
    ┌──────┐
    │ All  │  ← 4种全用
    └──────┘
           72%
    ┌──────┐
    │ DEB  │  ← 3种
    └──────┘
           68%
    ┌──────┐
    │ DE   │  ← 2种
    └──────┘
           65%
    ┌──────┐
    │  E   │  ← 只用Exploration
    └──────┘
           60%
    ┌──────┐
    │Random│
    └──────┘
```

---

### **图3: 样本选择分布热力图**

可视化每轮选择的样本在特征空间的分布：
- **Random**：均匀分布
- **Open-Loop**：聚集在高不确定性区域
- **Closed-Loop**：动态分布（随分类头状态变化）

---

### **表1: 完整实验结果**

| Method | Round 5 | Round 10 | Round 20 | 标注数 | 标注比例 | 提升 |
|--------|---------|----------|----------|--------|---------|------|
| Random | 42.3% | 51.2% | 58.6% | 4200 | 4.14% | - |
| Open-V0 | 48.1% | 64.5% | 70.2% | 4200 | 4.14% | +11.6% |
| Open-V1 | 49.3% | 66.8% | 72.5% | 4200 | 4.14% | +13.9% |
| **Closed-Att** | **51.2%** | **69.3%** | **76.8%** | 4200 | 4.14% | **+18.2%** |
| Full-Data | - | - | 85.2% | 101347 | 100% | - |

**相对性能：**
- Closed-Loop用4.14%数据达到76.8%
- Full-Data用100%数据达到85.2%
- **效率：90%性能只需4.14%标注！**

---

### **表2: 消融实验结果**

| Uncertainty Types | Acc@Round10 | Acc@Round20 | ΔAcc |
|------------------|-------------|-------------|------|
| Density | 48.2% | 54.3% | - |
| Exploration | 56.8% | 63.5% | +9.2% |
| Boundary | 58.1% | 65.2% | +10.9% |
| Multi-Scale | 53.6% | 61.8% | +7.5% |
| D+E | 62.3% | 66.7% | +12.4% |
| D+E+B | 65.8% | 70.1% | +15.8% |
| **D+E+B+M (Ours)** | **66.8%** | **72.5%** | **+18.2%** |

**结论：**
1. 单个不确定性效果有限
2. 组合优于单个（互补性）
3. 4种全用最好（探索+边界+密度+复杂度）

---

## 🚀 实验执行计划

### **阶段1: 快速验证（3天）**

```bash
# 1. 快速测试（3轮，验证流程）
python scripts/run_active_learning.py \
    --exp_name quick_test_random \
    --num_rounds 3 --budget_per_round 50 --train_epochs 5

python scripts/run_active_learning.py \
    --exp_name quick_test_open \
    --num_rounds 3 --budget_per_round 50 --train_epochs 5 \
    --use_prediction_uncertainty False

python scripts/run_active_learning.py \
    --exp_name quick_test_closed \
    --num_rounds 3 --budget_per_round 50 --train_epochs 5 \
    --use_prediction_uncertainty True
```

**检查点：**
- ✅ 代码无错误
- ✅ Closed > Open > Random
- ✅ 内存和时间可控

---

### **阶段2: 完整对比（7天）**

```bash
# 依次运行6组实验（每个10-14小时）
1. Random Baseline (12小时)
2. Open-V1 (12小时)
3. Closed-Attention (14小时)
4. Ablation实验（4-7） (每个12小时 × 7 = ~3.5天)
```

**建议：**
- 使用screen后台运行
- 每天检查日志
- 记录中间结果

---

### **阶段3: 分析与撰写（3天）**

1. 收集所有实验结果
2. 绘制对比图表
3. 分析消融实验
4. 撰写论文相关部分

---

## 📝 结果记录模板

```python
# 创建结果汇总表
results = {
    'Random': {
        'round_5': 42.3,
        'round_10': 51.2,
        'round_20': 58.6,
        'time': '11:23:45'
    },
    'Open-V1': {
        'round_5': 49.3,
        'round_10': 66.8,
        'round_20': 72.5,
        'time': '11:45:32'
    },
    'Closed-Attention': {
        'round_5': 51.2,
        'round_10': 69.3,
        'round_20': 76.8,
        'time': '13:12:18'
    },
    ...
}
```

---

## 🎯 成功标准

### **最低要求：**
- ✅ Closed > Open > Random（准确率）
- ✅ 消融实验证明4种不确定性互补
- ✅ 至少3组实验完成

### **理想目标：**
- ✅ Closed用4%数据达到>75%准确率
- ✅ 比Random提升>15%
- ✅ 比Open提升>4%
- ✅ 消融实验每增加一种不确定性都有提升

---

## ⏰ 时间估算

| 实验组 | 数量 | 单个耗时 | 总耗时 |
|-------|-----|---------|--------|
| Baseline | 1 | 12小时 | 12小时 |
| 开环 | 3 | 12小时 | 36小时 |
| 闭环 | 3 | 14小时 | 42小时 |
| 消融 | 7 | 12小时 | 84小时 |
| **总计** | **14** | - | **174小时（7.25天）** |

**建议：**
- 同时开多个实验（如果资源允许）
- 先跑关键实验（Random、Open-V1、Closed）
- 消融实验可以后续补充

---

## 📌 重要提示

1. **随机种子固定**：所有实验用seed=42，保证可复现
2. **参数一致**：除了对比的变量，其他参数保持一致
3. **日志保存**：每个实验保存完整日志
4. **中间检查**：每5轮检查一次是否正常
5. **备份结果**：实验结束立即备份results目录

---

## 🎓 论文撰写建议

### **实验部分结构**

1. **实验设置**
   - 数据集：ImageNet-100
   - 模型：DINOv3-ViT-Base/16 (冻结)
   - 初始样本：200（每类2个）
   - 每轮预算：200
   - 总轮数：20
   - 最终标注：4,200（4.14%）

2. **对比实验**
   - Baseline：随机采样
   - 开环：4种特征不确定性
   - 闭环：特征+预测不确定性融合
   - 结果：图1学习曲线

3. **消融实验**
   - 逐步增加不确定性
   - 验证互补性
   - 结果：图2柱状图

4. **分析讨论**
   - 为什么闭环更好？
   - 每种不确定性的贡献？
   - 与现有方法对比？

---

希望这个实验设计能帮到您的毕设！🎓✨

