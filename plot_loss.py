"""Plot training loss from latest CSV data.
Usage: python plot_loss.py [N_LAST_POINTS]
Default: plot last 1000 points (best for current run)
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

df = pd.read_csv('training_metrics.csv', header=None)
df.columns = ['curr_action_l1_loss', 'loss_value', 'next_actions_l1_loss']

# Take last N points only (current training run)
n_last = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
n_last = min(n_last, len(df))
df = df.iloc[-n_last:].reset_index(drop=True)

# Compute stats
loss = df['loss_value']
loss_min, loss_max = loss.min(), loss.max()
loss_start, loss_end = loss.iloc[0], loss.iloc[-1]
loss_avg50 = loss.iloc[-50:].mean()

print(f"V3 Training — Pure Linear SceneProjector")
print(f"Showing last {n_last}/{len(df)} points")
print(f"Loss: {loss_start:.4f} -> {loss_end:.4f} (50-step avg: {loss_avg50:.4f})")
print(f"Range: {loss_min:.4f} - {loss_max:.4f}")
print(f"Min at step {loss.idxmin()}: {loss_min:.4f}")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Main loss curve
w = max(3, n_last // 40)
ax = axes[0]
ax.plot(loss.values, alpha=0.15, color='#2196F3', linewidth=0.5)
ax.plot(loss.rolling(w, min_periods=1).mean(), color='#1565C0', linewidth=2, label=f'Smoothed (w={w})')
ax.axhline(y=loss_avg50, color='red', linestyle='--', alpha=0.5, label=f'Avg last 50: {loss_avg50:.4f}')
ax.set_xlabel('Batch Iteration')
ax.set_ylabel('L1 Loss')
ax.set_title('Training Loss (Pure Linear SceneProjector)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Zoom: last 200
d200 = df.iloc[-200:].reset_index(drop=True)
w2 = 10
ax2 = axes[1]
ax2.plot(d200['loss_value'].values, alpha=0.2, color='#4CAF50', linewidth=0.5)
ax2.plot(d200['loss_value'].rolling(w2, min_periods=1).mean(), color='#2E7D32', linewidth=2, label=f'Smoothed (w={w2})')
ax2.set_xlabel('Batch Iteration')
ax2.set_ylabel('L1 Loss')
ax2.set_title('Last 200 Steps')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Action breakdown
w3 = max(5, n_last // 20)
ax3 = axes[2]
ax3.plot(df['curr_action_l1_loss'].rolling(w3, min_periods=1).mean(), label='Curr Action L1', linewidth=1.5, color='#FF9800')
ax3.plot(df['next_actions_l1_loss'].rolling(w3, min_periods=1).mean(), label='Next Actions L1', linewidth=1.5, color='#9C27B0')
ax3.set_xlabel('Batch Iteration')
ax3.set_ylabel('L1 Loss')
ax3.set_title('Action L1 Breakdown (Smoothed)')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_loss_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Plot saved: training_loss_curve.png")
