import pandas as pd

# Read the CSV
df = pd.read_csv('/Volumes/Extreme SSD/DSS_Thesis/corrected_spatial_cv_results/corrected_spatial_cv_confusion_matrices.csv')

# Group by configuration and algorithm
grouped = df.groupby(['configuration', 'algorithm'])

print("# Confusion Matrices Summary\n")

for (config, algo), group in grouped:
    # We only care about RandomForest for now as it was the best
    if algo != 'RandomForest':
        continue
        
    print(f"## {config} - {algo}")
    
    total_tn = group['tn'].sum()
    total_fp = group['fp'].sum()
    total_fn = group['fn'].sum()
    total_tp = group['tp'].sum()
    
    # Calculate rates
    tpr = total_tp / (total_tp + total_fn) # Recall / Sensitivity
    tnr = total_tn / (total_tn + total_fp) # Specificity
    ppv = total_tp / (total_tp + total_fp) # Precision
    npv = total_tn / (total_tn + total_fn)
    
    print(f"**Aggregated across folds:**")
    print(f"- True Negatives: {total_tn}")
    print(f"- False Positives: {total_fp}")
    print(f"- False Negatives: {total_fn}")
    print(f"- True Positives: {total_tp}")
    print(f"\n**Metrics:**")
    print(f"- Sensitivity (Recall): {tpr:.4f}")
    print(f"- Specificity: {tnr:.4f}")
    print(f"- Precision: {ppv:.4f}")
    print(f"- NPV: {npv:.4f}")
    
    print("\n**Confusion Matrix:**")
    print("| | Predicted Negative | Predicted Positive |")
    print("|---|---|---|")
    print(f"| **Actual Negative** | {total_tn} | {total_fp} |")
    print(f"| **Actual Positive** | {total_fn} | {total_tp} |")
    print("\n---\n")
