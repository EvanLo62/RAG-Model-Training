"""
繪製訓練曲線腳本

從 trainer_state.json 讀取訓練歷史並繪製學習曲線
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

# 設定中文字體支援
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

parser = argparse.ArgumentParser(description='繪製訓練曲線')
parser.add_argument('--trainer_state_path', type=str, default='./models/retriever/trainer_state.json',
                    help='trainer_state.json 的路徑')
parser.add_argument('--output_dir', type=str, default='./plots',
                    help='圖片輸出目錄')
parser.add_argument('--output_name', type=str, default='training_curves',
                    help='輸出檔案名稱 (不含副檔名)')
parser.add_argument('--dpi', type=int, default=300,
                    help='圖片解析度')
parser.add_argument('--figsize', type=str, default='12,8',
                    help='圖片大小 (寬,高)')
args = parser.parse_args()

# 載入訓練狀態
print("=" * 70)
print("載入訓練歷史")
print("=" * 70)

with open(args.trainer_state_path, 'r', encoding='utf-8') as f:
    trainer_state = json.load(f)

log_history = trainer_state['log_history']
print(f"載入 {len(log_history)} 筆記錄")
print(f"總訓練 epochs: {trainer_state['epoch']}")
print(f"總訓練 steps: {trainer_state['global_step']}")

# 提取資料
steps = []
epochs = []
losses = []
learning_rates = []
grad_norms = []

for entry in log_history:
    if 'loss' in entry:  # 只處理有 loss 的記錄
        steps.append(entry['step'])
        epochs.append(entry['epoch'])
        losses.append(entry['loss'])
        learning_rates.append(entry['learning_rate'])
        grad_norms.append(entry['grad_norm'])

print(f"\n提取資料:")
print(f"  - Training loss 資料點: {len(losses)}")
print(f"  - Learning rate 資料點: {len(learning_rates)}")
print(f"  - Gradient norm 資料點: {len(grad_norms)}")

# 創建輸出目錄
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# 解析圖片大小
figsize = tuple(map(float, args.figsize.split(',')))

# ============================================================================
# 繪製完整的訓練曲線 (4 子圖)
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=figsize)
fig.suptitle('Training Curves - Retriever', fontsize=16, fontweight='bold')

# 子圖 1: Training Loss vs Steps
ax1 = axes[0, 0]
ax1.plot(steps, losses, 'b-', linewidth=2, label='Training Loss')
ax1.scatter(steps, losses, c='blue', s=30, alpha=0.6, zorder=5)
ax1.set_xlabel('Steps', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)

# 標註最終 loss
final_loss = losses[-1]
ax1.annotate(f'Final: {final_loss:.4f}', 
             xy=(steps[-1], final_loss),
             xytext=(10, 10), textcoords='offset points',
             fontsize=10, color='blue',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3),
             arrowprops=dict(arrowstyle='->', color='blue'))

# 子圖 2: Training Loss vs Epochs
ax2 = axes[0, 1]
ax2.plot(epochs, losses, 'g-', linewidth=2, label='Training Loss')
ax2.scatter(epochs, losses, c='green', s=30, alpha=0.6, zorder=5)
ax2.set_xlabel('Epochs', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.set_title('Training Loss (by Epoch)', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

# 標註初始和最終 loss
initial_loss = losses[0]
ax2.annotate(f'Initial: {initial_loss:.4f}', 
             xy=(epochs[0], initial_loss),
             xytext=(10, -15), textcoords='offset points',
             fontsize=9, color='green')
ax2.annotate(f'Final: {final_loss:.4f}', 
             xy=(epochs[-1], final_loss),
             xytext=(-60, 10), textcoords='offset points',
             fontsize=9, color='green')

# 子圖 3: Learning Rate Schedule
ax3 = axes[1, 0]
ax3.plot(steps, learning_rates, 'r-', linewidth=2, label='Learning Rate')
ax3.scatter(steps, learning_rates, c='red', s=30, alpha=0.6, zorder=5)
ax3.set_xlabel('Steps', fontsize=12)
ax3.set_ylabel('Learning Rate', fontsize=12)
ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=10)
ax3.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

# 子圖 4: Gradient Norm
ax4 = axes[1, 1]
ax4.plot(steps, grad_norms, 'm-', linewidth=2, label='Gradient Norm')
ax4.scatter(steps, grad_norms, c='magenta', s=30, alpha=0.6, zorder=5)
ax4.set_xlabel('Steps', fontsize=12)
ax4.set_ylabel('Gradient Norm', fontsize=12)
ax4.set_title('Gradient Norm', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=10)

# 標註最大 gradient norm
max_grad_idx = np.argmax(grad_norms)
max_grad = grad_norms[max_grad_idx]
ax4.annotate(f'Max: {max_grad:.2f}', 
             xy=(steps[max_grad_idx], max_grad),
             xytext=(10, -15), textcoords='offset points',
             fontsize=9, color='magenta',
             arrowprops=dict(arrowstyle='->', color='magenta'))

plt.tight_layout()

# 儲存圖片
output_path = output_dir / f"{args.output_name}_full.png"
plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
print(f"\n完整訓練曲線已儲存至: {output_path}")

# ============================================================================
# 繪製單獨的 Loss 曲線 (用於報告)
# ============================================================================

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# 繪製 loss 曲線
ax.plot(steps, losses, 'b-', linewidth=2.5, label='Training Loss', marker='o', markersize=6)
ax.set_xlabel('Training Steps', fontsize=14, fontweight='bold')
ax.set_ylabel('Loss', fontsize=14, fontweight='bold')
ax.set_title('Retriever Training Loss Curve', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=12, loc='upper right')

# 添加統計資訊
textstr = '\n'.join([
    f'Initial Loss: {initial_loss:.4f}',
    f'Final Loss: {final_loss:.4f}',
    f'Loss Reduction: {((initial_loss - final_loss) / initial_loss * 100):.1f}%',
    f'Total Epochs: {trainer_state["epoch"]:.1f}',
    f'Total Steps: {trainer_state["global_step"]}'
])
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()

# 儲存圖片
output_path = output_dir / f"{args.output_name}_loss_only.png"
plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
print(f"Loss 曲線已儲存至: {output_path}")

# ============================================================================
# 輸出訓練統計
# ============================================================================

print("\n" + "=" * 70)
print("訓練統計")
print("=" * 70)
print(f"初始 Loss:    {initial_loss:.4f}")
print(f"最終 Loss:    {final_loss:.4f}")
print(f"Loss 降幅:    {initial_loss - final_loss:.4f} ({(initial_loss - final_loss) / initial_loss * 100:.1f}%)")
print(f"最小 Loss:    {min(losses):.4f} (step {steps[np.argmin(losses)]})")
print(f"最大 Grad Norm: {max(grad_norms):.2f} (step {steps[np.argmax(grad_norms)]})")
print(f"最終 LR:      {learning_rates[-1]:.2e}")
print("=" * 70)

# ============================================================================
# 儲存數據到 CSV (方便後續分析)
# ============================================================================

import csv

csv_path = output_dir / f"{args.output_name}_data.csv"
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['step', 'epoch', 'loss', 'learning_rate', 'grad_norm'])
    for i in range(len(steps)):
        writer.writerow([steps[i], epochs[i], losses[i], learning_rates[i], grad_norms[i]])

print(f"\n訓練數據已儲存至: {csv_path}")

print("\n所有圖表生成完成!")
print("=" * 70)
