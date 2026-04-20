# 🚀 实验脚本使用指南

## 📁 文件说明

```
scripts/
├── run_active_learning.py   # 主实验脚本（带完整日志）
├── view_results.py           # 查看结果工具
├── prepare_imagenet100.py    # 数据集准备
├── quick_start.sh            # 一键启动（本地用）
└── README.md                 # 本文档
```

---

## 🎯 运行实验

### **1. 使用screen运行（推荐SSH环境）**

```bash
# 创建screen会话
screen -S active_learning

# 激活环境
source /root/miniconda3/bin/activate dinov3

# 切换目录
cd /root/autodl-tmp/dinov3

# 开始训练
python scripts/run_active_learning.py \
    --exp_name my_first_exp \
    --num_rounds 10 \
    --budget_per_round 100

# 退出screen（训练继续）
# 按键：Ctrl + A，然后 D

# 重新进入查看
screen -r active_learning
```

### **2. 快速测试（少轮数）**

```bash
python scripts/run_active_learning.py \
    --exp_name quick_test \
    --num_rounds 3 \
    --budget_per_round 50 \
    --train_epochs 5
```

---

## 📊 查看结果

### **方法1：使用工具脚本**

```bash
# 列出所有实验
python scripts/view_results.py list

# 查看某个实验的历史
python scripts/view_results.py history --exp my_first_exp

# 查看某个实验的日志（最后50行）
python scripts/view_results.py log --exp my_first_exp --tail 50

# 对比多个实验
python scripts/view_results.py compare --exps exp1 exp2 exp3
```

### **方法2：直接查看文件**

```bash
# 查看历史（JSON格式）
cat results/my_first_exp/history.json

# 或者格式化查看
python -m json.tool results/my_first_exp/history.json

# 查看日志（最后50行）
tail -n 50 results/my_first_exp/log_*.txt

# 查看完整日志
cat results/my_first_exp/log_*.txt

# 实时监控日志（如果正在训练）
tail -f results/my_first_exp/log_*.txt
```

---

## 📂 结果文件说明

训练后，结果会保存在 `results/{exp_name}/` 目录：

```
results/my_first_exp/
├── history.json              # 📊 训练历史（JSON格式）
├── checkpoint.pth            # 💾 模型检查点
└── log_20241120_143025.txt   # 📄 完整训练日志
```

### **history.json 内容**

```json
{
  "rounds": [1, 2, 3, ...],
  "train_accuracy": [45.2, 52.3, ...],
  "val_accuracy": [43.1, 50.8, ...],
  "labeled_samples": [200, 300, 400, ...],
  "selected_priorities": [...]
}
```

### **日志文件内容**

包含所有终端输出：
- 实验参数
- 数据加载信息
- 模型创建信息
- 每轮训练过程
- 不确定性计算
- 样本选择结果
- 准确率变化
- 最终评估结果

---

## 🎛️ 常用参数

```bash
# 数据参数
--data_root              # 数据集路径（默认：imagenet100_split）
--batch_size 32          # 批大小
--num_workers 4          # 数据加载线程数

# 模型参数
--model_size base        # DINOv3模型（small/base/large）
--layers 3 6 9 11        # 提取特征的层
--dropout 0.1            # 分类头dropout

# 主动学习参数
--num_rounds 10          # 主动学习轮数
--budget_per_round 100   # 每轮选择样本数
--init_samples_per_class 2  # 初始每类样本数
--k_neighbors 10         # KNN的K值

# 不确定性阈值
--density_threshold 0.5
--exploration_threshold 0.5
--boundary_threshold 0.6
--multiscale_threshold 0.5

# 训练参数
--train_epochs 10        # 每轮训练epoch数
--learning_rate 0.001    # 学习率

# 其他
--seed 42                # 随机种子
--exp_name my_exp        # 实验名称
```

---

## 🔧 常用命令

### **训练相关**

```bash
# 标准实验
python scripts/run_active_learning.py --exp_name standard

# 快速测试
python scripts/run_active_learning.py --exp_name test --num_rounds 3

# 完整实验（更多轮数）
python scripts/run_active_learning.py --exp_name full --num_rounds 20 --budget_per_round 200

# 小batch（显存不足）
python scripts/run_active_learning.py --exp_name small_batch --batch_size 16
```

### **查看相关**

```bash
# 查看所有实验
python scripts/view_results.py list

# 查看历史
python scripts/view_results.py history --exp my_exp

# 查看最后100行日志
python scripts/view_results.py log --exp my_exp --tail 100

# 实时查看日志
tail -f results/my_exp/log_*.txt

# 对比实验
python scripts/view_results.py compare --exps exp1 exp2 exp3
```

### **screen管理**

```bash
# 列出所有screen
screen -ls

# 进入screen
screen -r active_learning

# 创建新screen
screen -S new_exp

# 在screen内退出（训练继续）
# Ctrl + A，然后 D

# 终止screen
# 在screen内：Ctrl + A，然后 K，确认 Y
```

---

## 📈 典型实验流程

### **1. 快速测试（5分钟）**

验证代码是否正常运行：

```bash
python scripts/run_active_learning.py \
    --exp_name quick_test \
    --num_rounds 2 \
    --budget_per_round 20 \
    --train_epochs 3
```

### **2. 标准实验（2-3小时）**

正式实验：

```bash
screen -S exp1
source /root/miniconda3/bin/activate dinov3
cd /root/autodl-tmp/dinov3

python scripts/run_active_learning.py \
    --exp_name exp_v1_standard \
    --num_rounds 10 \
    --budget_per_round 100 \
    --train_epochs 10

# Ctrl+A+D 退出
```

### **3. 查看进度**

```bash
# 进入screen查看实时输出
screen -r exp1

# 或者查看日志最后几行
python scripts/view_results.py log --exp exp_v1_standard --tail 30

# 或者实时监控
tail -f results/exp_v1_standard/log_*.txt
```

### **4. 查看结果**

```bash
# 查看完整历史
python scripts/view_results.py history --exp exp_v1_standard

# 查看JSON
cat results/exp_v1_standard/history.json | python -m json.tool
```

---

## ⚠️ 常见问题

### **Q1: 日志文件在哪里？**

```bash
# 在实验目录下，文件名带时间戳
ls -lh results/my_exp/

# 输出示例：
# log_20241120_143025.txt  ← 这个就是日志
# history.json
# checkpoint.pth
```

### **Q2: 如何实时查看训练进度？**

```bash
# 方法1：进入screen
screen -r active_learning

# 方法2：tail实时监控日志
tail -f results/my_exp/log_*.txt

# 方法3：查看最后N行
tail -n 50 results/my_exp/log_*.txt
```

### **Q3: 训练中断了怎么办？**

```bash
# 查看日志看到哪一步了
python scripts/view_results.py log --exp my_exp --tail 100

# 目前不支持断点恢复，需要重新开始
# TODO: 未来添加断点恢复功能
```

### **Q4: 如何对比多个实验？**

```bash
# 使用对比工具
python scripts/view_results.py compare --exps exp1 exp2 exp3

# 或者分别查看
python scripts/view_results.py history --exp exp1
python scripts/view_results.py history --exp exp2
```

---

## 💡 提示

1. **使用screen**：SSH环境下务必使用screen，避免连接断开导致训练中止
2. **命名规范**：给实验起有意义的名字，如 `exp_v1_budget100_rounds10`
3. **日志查看**：训练时可以另开终端用 `tail -f` 实时查看日志
4. **资源监控**：另开终端运行 `watch -n 1 nvidia-smi` 监控GPU使用
5. **备份结果**：重要实验结果及时备份

---

## 📧 帮助

如有问题，请查看：
1. 日志文件：`results/{exp_name}/log_*.txt`
2. 代码注释：`active_learning/README.md`
3. 脚本源码：`scripts/run_active_learning.py`

