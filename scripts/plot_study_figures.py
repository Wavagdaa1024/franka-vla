import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Set clean scientific style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

DATA_FILE = Path(r"C:\Users\74727\Desktop\project\VLA_franka\outputs\reports\comparison_lora_vs_unfreeze.json")
FIG_DIR = Path(r"C:\Users\74727\Desktop\project\VLA_franka\outputs\reports\figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

with open(DATA_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Colors
colors = ['#7f7f7f', '#1f77b4', '#d62728', '#2ca02c', '#9467bd']
tasks = ['pick and place the blue cube', 'pick and place the carrot', 'pick and place the orange cube', 'pick and place the pepper', 'pick and place the red cube']
task_short = ['Blue Cube', 'Carrot (OOD)', 'Orange Cube', 'Pepper (OOD)', 'Red Cube (ID)']

# -------------------------------------------------------------------------------------------------
# FIGURE 1: 5-Task Mean MAE and Task-by-task Breakdown (Bar Chart)
# -------------------------------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={'width_ratios': [1, 2]})

# Left: Overall Mean MAE
names_short = ['Zero-Shot', 'Expert-Unfreeze\n(Round 1)', 'Vision+Expert\n(Round 2)', 'Multi-LoRA\n(Step 3000)', 'Multi-LoRA\n(Step 5000)']
mean_maes = [d['overall_mean_mae_deg'] for d in data]
bars = ax1.bar(names_short, mean_maes, color=colors, width=0.55, edgecolor='black', linewidth=1)
ax1.set_ylabel('Mean Joint MAE (degrees)', fontsize=12, fontweight='bold')
ax1.set_title('(a) 5-Task Overall Mean MAE', fontsize=13, fontweight='bold', pad=12)
ax1.axhline(1.0, color='red', linestyle='--', linewidth=1.2, label='1.0 deg Threshold')
ax1.grid(axis='y', linestyle=':', alpha=0.6)
ax1.set_ylim(0, 6.5)
ax1.legend(loc='upper right', frameon=True)
for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.12, f'{yval:.2f}°', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Right: Per-Task Breakdown (Grouped Bar Chart)
x = np.arange(len(task_short))
bar_w = 0.16
offsets = [-2*bar_w, -bar_w, 0, bar_w, 2*bar_w]
model_labels = ['Base Zero-Shot', 'Round 1 (Expert)', 'Round 2 (Vision+Expert)', 'LoRA (Step 3000)', 'LoRA (Step 5000)']

for i, (d, col, off, lbl) in enumerate(zip(data, colors, offsets, model_labels)):
    task_vals = [d['tasks'][t]['joint_mae_deg'] for t in tasks]
    rects = ax2.bar(x + off, task_vals, bar_w, label=lbl, color=col, edgecolor='black', linewidth=0.8)

ax2.set_xticks(x)
ax2.set_xticklabels(task_short, fontsize=11, fontweight='bold')
ax2.set_ylabel('Per-Task Joint MAE (degrees)', fontsize=12, fontweight='bold')
ax2.set_title('(b) Per-Task Joint MAE Across Manipulation Scenarios', fontsize=13, fontweight='bold', pad=12)
ax2.grid(axis='y', linestyle=':', alpha=0.6)
ax2.axhline(1.0, color='red', linestyle='--', linewidth=1.0)
ax2.legend(loc='upper right', fontsize=9, frameon=True, ncol=2)

plt.tight_layout()
fig1_path = FIG_DIR / "fig1_ablation_5task_mae.png"
plt.savefig(fig1_path, dpi=300)
plt.close()
print(f"Saved: {fig1_path}")

# -------------------------------------------------------------------------------------------------
# FIGURE 2: Trainable Parameters vs Performance (Bubble Chart)
# -------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.5))
params_m = [d['trainable_params_m'] for d in data]
sizes_mb = [d['ckpt_size_mb'] for d in data]
maes = [d['overall_mean_mae_deg'] for d in data]

# Adjust zero-shot display coordinates
disp_params = [max(p, 0.5) for p in params_m]

for i, (p, m, s, col, lbl) in enumerate(zip(disp_params, maes, sizes_mb, colors, model_labels)):
    bubble_size = max(np.sqrt(s) * 25, 80) if s > 0 else 80
    ax.scatter(p, m, s=bubble_size, color=col, alpha=0.8, edgecolors='black', linewidth=1.5, label=f"{lbl} ({s:.1f}MB)")
    ax.annotate(f"{lbl}\n(MAE: {m:.2f}°, {s:.1f}MB)", (p, m), xytext=(0, 15), textcoords='offset points', ha='center', fontsize=9, fontweight='bold')

ax.set_xscale('log')
ax.set_xlabel('Trainable Parameters (Millions, Log Scale)', fontsize=12, fontweight='bold')
ax.set_ylabel('5-Task Mean Joint MAE (degrees, lower is better)', fontsize=12, fontweight='bold')
ax.set_title('Figure 2: Parameter & Storage Efficiency vs. Policy Precision\n(Bubble size proportional to Checkpoint File Size)', fontsize=13, fontweight='bold', pad=12)
ax.axhline(1.0, color='red', linestyle='--', linewidth=1.2, label='1.0° Threshold')
ax.grid(True, linestyle=':', alpha=0.6)
ax.set_ylim(0.5, 6.5)

plt.tight_layout()
fig2_path = FIG_DIR / "fig2_parameter_storage_efficiency.png"
plt.savefig(fig2_path, dpi=300)
plt.close()
print(f"Saved: {fig2_path}")

# -------------------------------------------------------------------------------------------------
# FIGURE 3: Reaching vs Grasping Error Comparison
# -------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
names = ['Zero-Shot', 'Round 1 (Expert)', 'Round 2 (Vision+Expert)', 'LoRA (Step 3000)', 'LoRA (Step 5000)']
reach_maes = [d['reach_mae_deg'] for d in data]
grasp_maes = [d['grasp_mae_deg'] for d in data]

x = np.arange(len(names))
width = 0.35

rects1 = ax.bar(x - width/2, reach_maes, width, label='Reaching Phase MAE (Approach)', color='#3498db', edgecolor='black')
rects2 = ax.bar(x + width/2, grasp_maes, width, label='Grasping Phase MAE (Contact)', color='#e74c3c', edgecolor='black')

ax.set_ylabel('Joint MAE (degrees)', fontsize=12, fontweight='bold')
ax.set_title('Figure 3: Reaching vs. Grasping Phase Precision Across Model Architectures', fontsize=13, fontweight='bold', pad=12)
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=10, fontweight='bold')
ax.legend(fontsize=11, frameon=True)
ax.grid(axis='y', linestyle=':', alpha=0.6)

for rects in [rects1, rects2]:
    for bar in rects:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, h + 0.1, f'{h:.2f}°', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
fig3_path = FIG_DIR / "fig3_reaching_vs_grasping.png"
plt.savefig(fig3_path, dpi=300)
plt.close()
print(f"Saved: {fig3_path}")

# -------------------------------------------------------------------------------------------------
# FIGURE 4: LoRA Checkpoint Ladder Convergence (Steps 1000 to 5000)
# -------------------------------------------------------------------------------------------------
# Data from evaluation_matrix.json
ladder_steps = [1000, 2000, 3000, 4000, 5000]
ladder_means = [1.37, 1.31, 0.99, 1.12, 1.03]
ladder_blue = [1.43, 1.11, 0.87, 1.36, 1.62]
ladder_carrot = [1.32, 0.91, 0.63, 0.89, 1.14]
ladder_orange = [2.09, 1.74, 1.16, 1.08, 0.80]
ladder_pepper = [0.63, 1.41, 0.60, 0.74, 0.62]
ladder_red = [1.38, 1.35, 1.67, 1.54, 0.96]

fig, ax = plt.subplots(figsize=(8.5, 5))
ax.plot(ladder_steps, ladder_means, 'k-o', linewidth=3, markersize=8, label='Overall Mean MAE', zorder=5)
ax.plot(ladder_steps, ladder_red, '--s', color='#e74c3c', linewidth=2, label='Red Cube (ID)', alpha=0.9)
ax.plot(ladder_steps, ladder_orange, '--^', color='#e67e22', linewidth=2, label='Orange Cube', alpha=0.9)
ax.plot(ladder_steps, ladder_blue, '--d', color='#2980b9', linewidth=2, label='Blue Cube', alpha=0.9)
ax.plot(ladder_steps, ladder_carrot, '--v', color='#27ae60', linewidth=2, label='Carrot', alpha=0.9)
ax.plot(ladder_steps, ladder_pepper, '--p', color='#8e44ad', linewidth=2, label='Pepper', alpha=0.9)

ax.axhline(1.0, color='gray', linestyle=':', linewidth=1.2, label='1.0° Threshold')
ax.axvline(3000, color='green', linestyle='-.', linewidth=1.5, alpha=0.7, label='Optimal Checkpoint (Step 3000)')

ax.set_xlabel('LoRA Fine-Tuning Steps', fontsize=12, fontweight='bold')
ax.set_ylabel('Joint MAE (degrees, lower is better)', fontsize=12, fontweight='bold')
ax.set_title('Figure 4: Multi-Task LoRA Checkpoint Ladder Evaluation Across Training Steps', fontsize=13, fontweight='bold', pad=12)
ax.set_xticks(ladder_steps)
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='upper right', fontsize=9, frameon=True, ncol=2)
ax.set_ylim(0.4, 2.3)

plt.tight_layout()
fig4_path = FIG_DIR / "fig4_lora_ladder_convergence.png"
plt.savefig(fig4_path, dpi=300)
plt.close()
print(f"Saved: {fig4_path}")

print("All 4 publication figures generated successfully.")
