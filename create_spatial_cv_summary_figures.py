import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load results
csv_path = "corrected_spatial_cv_results/corrected_spatial_cv_results.csv"
df = pd.read_csv(csv_path)

# --- Algorithm Performance Bar Chart ---
algos = ['RandomForest', 'GradientBoosting', 'LogisticRegression', 'MLP']
perf = df.groupby('algorithm').agg({'spatial_auc_mean':'mean','spatial_auc_std':'mean','spatial_f1_mean':'mean','spatial_f1_std':'mean'}).loc[algos]

plt.figure(figsize=(8,5))
bar_width = 0.35
x = range(len(algos))
plt.bar(x, perf['spatial_auc_mean'], bar_width, yerr=perf['spatial_auc_std'], label='AUC', color='#1f77b4')
plt.bar([i+bar_width for i in x], perf['spatial_f1_mean'], bar_width, yerr=perf['spatial_f1_std'], label='F1', color='#ff7f0e')
plt.xticks([i+bar_width/2 for i in x], algos)
plt.ylabel('Score')
plt.title('Spatial CV Algorithm Performance (Mean ± Std)')
plt.legend()
plt.tight_layout()
plt.savefig('spatial_cv_algorithm_performance.png', dpi=300)
plt.close()

# --- Top Configurations Summary Chart ---
top_configs = df.sort_values('spatial_auc_mean', ascending=False).head(10)
plt.figure(figsize=(10,6))
ax = sns.barplot(y=top_configs['configuration'] + ' + ' + top_configs['algorithm'], x=top_configs['spatial_auc_mean'], palette='Blues_d')
for i, (auc, f1) in enumerate(zip(top_configs['spatial_auc_mean'], top_configs['spatial_f1_mean'])):
    ax.text(auc+0.001, i, f"AUC={auc:.4f}\nF1={f1:.4f}", va='center', fontsize=9)
plt.xlabel('Mean AUC')
plt.title('Top 10 Spatial CV Configurations (AUC, F1)')
plt.tight_layout()
plt.savefig('spatial_cv_analysis_summary.png', dpi=300)
plt.close()

print('Saved: spatial_cv_algorithm_performance.png')
print('Saved: spatial_cv_analysis_summary.png')
