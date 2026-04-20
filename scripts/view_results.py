"""
查看实验结果的工具脚本
"""

import json
import sys
from pathlib import Path
import argparse

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def view_history(exp_name: str):
    """查看实验历史"""
    history_file = project_root / 'results' / exp_name / 'history.json'
    
    if not history_file.exists():
        print(f"❌ 历史文件不存在: {history_file}")
        return
    
    with open(history_file) as f:
        history = json.load(f)
    
    print("=" * 70)
    print(f"实验: {exp_name}")
    print("=" * 70)
    print()
    
    print("📊 验证准确率变化:")
    print("Round  | Labeled | Val Acc")
    print("-------|---------|--------")
    for i, (round_num, labeled, val_acc) in enumerate(zip(
        history['rounds'],
        history['labeled_samples'],
        history['val_accuracy']
    )):
        print(f"{round_num:6d} | {labeled:7d} | {val_acc:6.2f}%")
    
    print()
    print("📈 最终结果:")
    print(f"  最高验证准确率: {max(history['val_accuracy']):.2f}%")
    print(f"  最终验证准确率: {history['val_accuracy'][-1]:.2f}%")
    print(f"  总标注样本: {history['labeled_samples'][-1]}")
    print()


def view_logs(exp_name: str, tail: int = 50):
    """查看日志（最后N行）"""
    log_dir = project_root / 'results' / exp_name
    
    if not log_dir.exists():
        print(f"❌ 实验目录不存在: {log_dir}")
        return
    
    # 找到最新的日志文件
    log_files = sorted(log_dir.glob('log_*.txt'))
    
    if not log_files:
        print(f"❌ 没有找到日志文件")
        return
    
    latest_log = log_files[-1]
    
    print("=" * 70)
    print(f"日志文件: {latest_log}")
    print("=" * 70)
    print()
    
    # 读取并打印最后N行
    with open(latest_log, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"显示最后 {tail} 行:")
    print("-" * 70)
    for line in lines[-tail:]:
        print(line.rstrip())


def list_experiments():
    """列出所有实验"""
    results_dir = project_root / 'results'
    
    if not results_dir.exists():
        print("❌ results目录不存在")
        return
    
    experiments = sorted([d.name for d in results_dir.iterdir() if d.is_dir()])
    
    if not experiments:
        print("❌ 没有找到实验")
        return
    
    print("=" * 70)
    print("所有实验:")
    print("=" * 70)
    print()
    
    for i, exp in enumerate(experiments, 1):
        exp_dir = results_dir / exp
        history_file = exp_dir / 'history.json'
        log_files = list(exp_dir.glob('log_*.txt'))
        
        print(f"{i}. {exp}")
        if history_file.exists():
            with open(history_file) as f:
                history = json.load(f)
            print(f"   - 轮数: {len(history['rounds'])}")
            print(f"   - 最终准确率: {history['val_accuracy'][-1]:.2f}%")
        print(f"   - 日志文件: {len(log_files)} 个")
        print()


def compare_experiments(exp_names: list):
    """对比多个实验"""
    print("=" * 70)
    print("实验对比")
    print("=" * 70)
    print()
    
    histories = {}
    for exp_name in exp_names:
        history_file = project_root / 'results' / exp_name / 'history.json'
        if history_file.exists():
            with open(history_file) as f:
                histories[exp_name] = json.load(f)
        else:
            print(f"⚠️  {exp_name}: 历史文件不存在")
    
    if not histories:
        return
    
    # 打印对比表
    max_rounds = max(len(h['rounds']) for h in histories.values())
    
    print("Round", end="")
    for exp_name in histories.keys():
        print(f" | {exp_name:12s}", end="")
    print()
    print("-" * (7 + 15 * len(histories)))
    
    for i in range(max_rounds):
        print(f"{i+1:5d}", end="")
        for exp_name, history in histories.items():
            if i < len(history['val_accuracy']):
                acc = history['val_accuracy'][i]
                print(f" | {acc:12.2f}%", end="")
            else:
                print(f" | {'':12s}", end="")
        print()
    
    print()
    print("最终结果:")
    for exp_name, history in histories.items():
        final_acc = history['val_accuracy'][-1]
        max_acc = max(history['val_accuracy'])
        print(f"  {exp_name:20s}: 最终={final_acc:.2f}%, 最高={max_acc:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='查看实验结果')
    parser.add_argument('command', choices=['history', 'log', 'list', 'compare'],
                        help='命令: history(历史), log(日志), list(列表), compare(对比)')
    parser.add_argument('--exp', type=str, help='实验名称')
    parser.add_argument('--exps', nargs='+', help='多个实验名称（用于对比）')
    parser.add_argument('--tail', type=int, default=50, help='显示日志的最后N行')
    
    args = parser.parse_args()
    
    if args.command == 'history':
        if not args.exp:
            print("❌ 请指定实验名称: --exp EXP_NAME")
            return
        view_history(args.exp)
    
    elif args.command == 'log':
        if not args.exp:
            print("❌ 请指定实验名称: --exp EXP_NAME")
            return
        view_logs(args.exp, tail=args.tail)
    
    elif args.command == 'list':
        list_experiments()
    
    elif args.command == 'compare':
        if not args.exps or len(args.exps) < 2:
            print("❌ 请指定至少2个实验名称: --exps EXP1 EXP2 ...")
            return
        compare_experiments(args.exps)


if __name__ == "__main__":
    main()

