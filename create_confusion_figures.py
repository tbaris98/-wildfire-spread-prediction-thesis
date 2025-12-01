#!/usr/bin/env python3
"""
Create confusion matrix heatmap and error-breakdown plot for thesis.
Saves `fig_confusion_matrix.png` and `fig_error_breakdown.png` in repository root.
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Load confusion matrix counts from CSV for a selected configuration and algorithm
import pandas as pd

# Path to CSV file
csv_path = os.path.join(os.path.dirname(__file__), 'corrected_spatial_cv_results', 'corrected_spatial_cv_confusion_matrices.csv')


# Select configuration to compare all algorithms
SELECTED_CONFIG = 'tree_based_random_negative'  # change as needed

# Load CSV
df = pd.read_csv(csv_path)
algorithms = ['RandomForest', 'GradientBoosting', 'LogisticRegression', 'MLP']

# Prepare confusion matrices for each algorithm
cm_dict = {}

for algo in algorithms:
    filtered = df[(df['configuration'] == SELECTED_CONFIG) & (df['algorithm'] == algo)]
    if filtered.empty:
        continue
    # Calculate mean values
    TN = filtered['tn'].mean()
    FP = filtered['fp'].mean()
    FN = filtered['fn'].mean()
    TP = filtered['tp'].mean()
    # Rescale so each actual class sums to 100,000
    actual_nonfire_sum = TN + FP
    actual_fire_sum = FN + TP
    TN_rescaled = TN * 100000 / actual_nonfire_sum
    FP_rescaled = FP * 100000 / actual_nonfire_sum
    FN_rescaled = FN * 100000 / actual_fire_sum
    TP_rescaled = TP * 100000 / actual_fire_sum
    cm_dict[algo] = np.array([[int(round(TN_rescaled)), int(round(FP_rescaled))],
                              [int(round(FN_rescaled)), int(round(TP_rescaled))]])

# Create output directory (repo root)
figures_dir = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(figures_dir, exist_ok=True)
comparison_path = os.path.join(figures_dir, 'fig_confusion_matrix_comparison.png')

# Plot 2x2 grid of confusion matrices
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for i, algo in enumerate(algorithms):
    if algo not in cm_dict:
        continue
    ax = axes[i//2, i%2]
    cm = cm_dict[algo]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues', cbar=False,
                xticklabels=['Predicted Non-Fire', 'Predicted Fire'],
                yticklabels=['Actual Non-Fire', 'Actual Fire'], ax=ax)
    # annotate normalized percentages
    for ii in range(2):
        for jj in range(2):
            val = cm_norm[ii, jj]
            ax.text(jj+0.5, ii+0.5 - 0.25, f"{val:.1%}", ha='center', color='black', fontsize=9)
    ax.set_title(algo)

# Ensure the figure title always matches SELECTED_CONFIG
title_config = SELECTED_CONFIG.replace('_', ' ').title()
plt.suptitle(f'Confusion Matrix Comparison\n{title_config} (Sampling)', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(comparison_path, dpi=300)
plt.close()
print('Saved:', comparison_path)

# Create output directory (repo root)
confusion_path = os.path.join(figures_dir, 'fig_confusion_matrix.png')
breakdown_path = os.path.join(figures_dir, 'fig_error_breakdown.png')

# Confusion matrix array
cm = np.array([[TN, FP], [FN, TP]])
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

# Plot heatmap (counts with normalized percentages)
plt.figure(figsize=(6,5))
ax = sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues', cbar=False,
                 xticklabels=['Predicted Non-Fire', 'Predicted Fire'],
                 yticklabels=['Actual Non-Fire', 'Actual Fire'])
# annotate normalized percentages
for i in range(2):
    for j in range(2):
        val = cm_norm[i, j]
        ax.text(j+0.5, i+0.5 - 0.25, f"{val:.1%}", ha='center', color='black', fontsize=9)

plt.title('Confusion Matrix (counts)\nRandom Forest — Tree-based features, Random Sampling')
plt.tight_layout()
plt.savefig(confusion_path, dpi=300)
plt.close()

# Error breakdown bar chart: show TP, TN, FP, FN counts and percentages
labels = ['True Positive (TP)', 'True Negative (TN)', 'False Positive (FP)', 'False Negative (FN)']
counts = [TP, TN, FP, FN]
percent = [c / sum(counts) for c in counts]

plt.figure(figsize=(8,4))
sns.barplot(x=counts, y=labels, palette=['#2ca02c','#1f77b4','#ff7f0e','#d62728'])
# annotate counts and percentages
for i, (c, p) in enumerate(zip(counts, percent)):
    plt.text(c + max(counts)*0.005, i, f"{c:,} ({p:.1%})", va='center')

plt.xlabel('Count')
plt.title('Error Breakdown: TP/TN/FP/FN counts (Random Forest, Selected Config)')
ax = plt.gca()
legend = ax.get_legend()
if legend:
    legend.set_bbox_to_anchor((1.01, 1))
    legend.set_loc('upper left')
plt.tight_layout(pad=0.5)
plt.savefig(breakdown_path, dpi=300, bbox_inches='tight')
plt.close()

print('Saved:', confusion_path)
print('Saved:', breakdown_path)
print("Saved", comparison_path)