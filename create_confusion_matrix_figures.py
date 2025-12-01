import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Path to your CSV file
csv_path = "corrected_spatial_cv_results/corrected_spatial_cv_confusion_matrices.csv"
output_dir = "corrected_spatial_cv_results/confusion_matrix_figures"
os.makedirs(output_dir, exist_ok=True)

# Load the CSV
df = pd.read_csv(csv_path)


# Aggregate by configuration and algorithm

# For each configuration, plot all algorithms in a 2x2 grid
grouped = df.groupby(["configuration", "algorithm"])
configs = df['configuration'].unique()
algos = df['algorithm'].unique()

for config in configs:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f"Confusion Matrices for {config}", fontsize=16)
    algo_list = list(algos)
    for i, algo in enumerate(algo_list):
        group = df[(df['configuration'] == config) & (df['algorithm'] == algo)]
        if group.empty:
            continue
        tn = group['tn'].mean()
        fp = group['fp'].mean()
        fn = group['fn'].mean()
        tp = group['tp'].mean()
        matrix = [[tn, fp], [fn, tp]]
        auc = group['auc'].mean()
        f1 = group['f1'].mean()
        ax = axes[i//2, i%2]
        sns.heatmap(matrix, annot=True, fmt=".0f", cmap="Blues", cbar=False,
                    xticklabels=["Pred 0", "Pred 1"], yticklabels=["Actual 0", "Actual 1"], ax=ax)
        ax.set_title(f"{algo}\nMean AUC: {auc:.3f}, Mean F1: {f1:.3f}")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_name = f"{config}_all_algorithms_comparison.png".replace(" ", "_")
    plt.savefig(os.path.join(output_dir, fig_name))
    plt.close()

print(f"Comparison confusion matrix figures saved to {output_dir}")
