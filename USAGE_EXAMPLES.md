# 🚀 使用示例 - 开环/闭环模式切换

## ✅ 方案A已实现

**闭环模式：融合后的不确定性 + Density过滤 + Top-K**

---

## 📝 命令行使用

### **1. 开环模式（默认）**

```bash
# 完整命令
python scripts/run_active_learning.py \
    --exp_name exp_open_loop \
    --num_rounds 20 \
    --budget_per_round 200 \
    --train_epochs 20 \
    --batch_size 4 \
    --num_workers 2 \
    --use_prediction_uncertainty False

# 或者简化（默认就是False）
python scripts/run_active_learning.py \
    --exp_name exp_open_loop \
    --num_rounds 20 \
    --budget_per_round 200 \
    --train_epochs 20 \
    --batch_size 4
```

**流程：**
```
特征提取 → 4种不确定性 → Cascading Selector → 选择样本
```

---

### **2. 闭环模式（启用预测不确定性）**

```bash
# 闭环 - Attention策略（推荐）
python scripts/run_active_learning.py \
    --exp_name exp_closed_loop_attention \
    --num_rounds 20 \
    --budget_per_round 200 \
    --train_epochs 20 \
    --batch_size 4 \
    --num_workers 2 \
    --use_prediction_uncertainty True \
    --fusion_strategy attention

# 闭环 - Multiply策略
python scripts/run_active_learning.py \
    --exp_name exp_closed_loop_multiply \
    --use_prediction_uncertainty True \
    --fusion_strategy multiply \
    --num_rounds 20 \
    --budget_per_round 200

# 闭环 - Add策略
python scripts/run_active_learning.py \
    --exp_name exp_closed_loop_add \
    --use_prediction_uncertainty True \
    --fusion_strategy add \
    --num_rounds 20 \
    --budget_per_round 200
```

**流程：**
```
特征提取 → 4种不确定性 + 分类头预测 
    ↓
动态注意力融合 
    ↓
Density过滤（⭐ 方案A）
    ↓
Top-K选择
```

---

## 🧪 快速测试（验证功能）

### **测试1：开环快速测试**

```bash
python scripts/run_active_learning.py \
    --exp_name test_open \
    --num_rounds 3 \
    --budget_per_round 50 \
    --train_epochs 5 \
    --batch_size 4 \
    --use_prediction_uncertainty False
```

**预期输出：**
```
4. 创建主动学习器...
  模式: 开环（纯特征不确定性）
ℹ️  仅使用特征不确定性 (开环模式)
```

---

### **测试2：闭环快速测试**

```bash
python scripts/run_active_learning.py \
    --exp_name test_closed \
    --num_rounds 3 \
    --budget_per_round 50 \
    --train_epochs 5 \
    --batch_size 4 \
    --use_prediction_uncertainty True \
    --fusion_strategy attention
```

**预期输出：**
```
4. 创建主动学习器...
  模式: 闭环（特征 + 预测不确定性）
  融合策略: attention
✓ 预测不确定性已启用 (闭环模式)
  融合策略: attention

...

3. 选择新样本 (budget=50)...
  提取未标注样本特征...
  计算特征不确定性...
  计算预测不确定性...  ← 新增
  融合不确定性（动态注意力调制）...  ← 新增
  应用Density过滤（噪声保护）...  ← 新增
    过滤前: 101147 样本
    过滤后: 95234 样本 (过滤掉 5913 个噪声)
    ✓ 从 95234 个非噪声样本中选择了 50 个
```

---

## 📊 对比实验命令

### **实验组1：Baseline**

```bash
# TODO: 需要实现random selector
python scripts/run_active_learning.py \
    --exp_name baseline_random \
    --num_rounds 20 \
    --budget_per_round 200 \
    --selector random
```

---

### **实验组2：开环**

```bash
python scripts/run_active_learning.py \
    --exp_name exp_open_loop \
    --num_rounds 20 \
    --budget_per_round 200 \
    --train_epochs 20 \
    --batch_size 4 \
    --num_workers 2 \
    --use_prediction_uncertainty False
```

---

### **实验组3：闭环（您的核心贡献）**

```bash
python scripts/run_active_learning.py \
    --exp_name exp_closed_loop \
    --num_rounds 20 \
    --budget_per_round 200 \
    --train_epochs 20 \
    --batch_size 4 \
    --num_workers 2 \
    --use_prediction_uncertainty True \
    --fusion_strategy attention
```

---

## 🔍 区别对比

| 特性 | 开环模式 | 闭环模式 |
|-----|---------|---------|
| **命令行参数** | `--use_prediction_uncertainty False` | `--use_prediction_uncertainty True` |
| **特征不确定性** | ✅ 4种 | ✅ 4种 |
| **预测不确定性** | ❌ | ✅ 4种（熵/间隔/置信度/方差）|
| **融合策略** | - | attention/multiply/add |
| **样本选择** | Cascading Selector（级联规则）| 融合分数 + Density过滤 + Top-K |
| **噪声过滤** | ✅ Cascading内置 | ✅ Density阈值过滤 |
| **速度** | 快（~20min/轮）| 稍慢（~23min/轮，+15%）|
| **内存** | 低 | 稍高（分类头前向传播）|

---

## ⚙️ 参数说明

### **新增参数**

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `--use_prediction_uncertainty` | bool | False | 是否启用闭环模式 |
| `--fusion_strategy` | str | attention | 融合策略（attention/multiply/add）|

### **使用示例**

```bash
# 开环（默认）
--use_prediction_uncertainty False

# 闭环 - attention
--use_prediction_uncertainty True --fusion_strategy attention

# 闭环 - multiply
--use_prediction_uncertainty True --fusion_strategy multiply

# 闭环 - add
--use_prediction_uncertainty True --fusion_strategy add
```

---

## 🎯 推荐配置

### **快速验证**

```bash
python scripts/run_active_learning.py \
    --exp_name quick_test \
    --num_rounds 3 \
    --budget_per_round 50 \
    --train_epochs 5 \
    --batch_size 4 \
    --use_prediction_uncertainty True
```

**时间：** ~20-30分钟

---

### **完整实验（论文用）**

```bash
# 开环
python scripts/run_active_learning.py \
    --exp_name paper_open_loop \
    --num_rounds 20 \
    --budget_per_round 200 \
    --train_epochs 20 \
    --batch_size 4 \
    --use_prediction_uncertainty False

# 闭环（⭐ 您的创新）
python scripts/run_active_learning.py \
    --exp_name paper_closed_loop \
    --num_rounds 20 \
    --budget_per_round 200 \
    --train_epochs 20 \
    --batch_size 4 \
    --use_prediction_uncertainty True \
    --fusion_strategy attention
```

**时间：** 每个12-14小时

---

## ✅ 验证功能

### **检查1：参数解析**

```bash
python scripts/run_active_learning.py --help | grep prediction
```

**预期输出：**
```
--use_prediction_uncertainty
--fusion_strategy {attention,multiply,add}
```

---

### **检查2：日志输出**

**开环模式日志：**
```
4. 创建主动学习器...
  模式: 开环（纯特征不确定性）
ℹ️  仅使用特征不确定性 (开环模式)
```

**闭环模式日志：**
```
4. 创建主动学习器...
  模式: 闭环（特征 + 预测不确定性）
  融合策略: attention
✓ 预测不确定性已启用 (闭环模式)
  融合策略: attention
```

---

## 🚀 现在开始！

```bash
# 激活环境
source /root/miniconda3/bin/activate dinov3
cd /root/autodl-tmp/dinov3

# 快速测试闭环模式
python scripts/run_active_learning.py \
    --exp_name test_closed_loop \
    --num_rounds 3 \
    --budget_per_round 50 \
    --train_epochs 5 \
    --batch_size 4 \
    --use_prediction_uncertainty True
```

**检查要点：**
1. ✅ 看到"闭环模式"提示
2. ✅ 看到"计算预测不确定性"
3. ✅ 看到"应用Density过滤"
4. ✅ 没有错误，正常运行

**如果成功 → 可以跑完整实验！** 🎉

