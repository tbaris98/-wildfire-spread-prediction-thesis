"""
Thesis Figures Generation Script
Generates all main figures for the thesis, including feature selection visualizations.
Excludes leakage-prone features: acquisition time, FRP, brightness, confidence, scan, track.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# --- Centralized Logging Setup ---
import logging
from pathlib import Path
LOG_DIR = Path('wildfire_results')
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f'thesis_all_figures_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    filename=LOG_FILE,
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# --- Pipeline Configuration Logging ---
CONFIG_LOG = LOG_DIR / 'thesis_all_figures_config.txt'
with open(CONFIG_LOG, 'w') as f:
    f.write("# Thesis All Figures Configuration\n")
    f.write(f"Run timestamp: {pd.Timestamp.now().isoformat()}\n")
    f.write("Input: corrected_spatial_cv_results/corrected_spatial_cv_results.csv\n")
    f.write("Output: figures/ (all figures, logs)\n")
    f.write("Key Steps: figure generation, feature selection visualization, algorithm comparison\n")
    f.write("Hyperparameters: see code for plotting settings\n")

# --- Error Handling Decorator ---
def log_exceptions(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Exception in {func.__name__}: {e}", exc_info=True)
            raise
    return wrapper

# Ensure 'figures' directory exists for all saves
os.makedirs('figures', exist_ok=True)

# Leakage-prone feature patterns (case-insensitive)
LEAKAGE_PATTERNS = [
    'acq_time', 'neighbour_acq_time', 'frp', 'brightness', 'confidence', 'scan', 'track'
]

def exclude_leakage_features(features):
    """Exclude features containing leakage-prone patterns"""
    return [f for f in features if not any(pat in f.lower() for pat in LEAKAGE_PATTERNS)]

# --- Feature Selection Visualizations ---

# --- Algorithm Comparison (Original Style) ---
cv_results_path = 'corrected_spatial_cv_results/corrected_spatial_cv_results.csv'
if os.path.exists(cv_results_path):
    df = pd.read_csv(cv_results_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    algo_stats = df.groupby('algorithm').agg({
        'spatial_auc_mean': ['mean', 'std'],
        'spatial_f1_mean': ['mean', 'std']
    }).reset_index()
    # AUC comparison
    ax1 = axes[0]
    algorithms = algo_stats['algorithm']
    auc_means = algo_stats[('spatial_auc_mean', 'mean')]
    auc_stds = algo_stats[('spatial_auc_mean', 'std')]
    bars1 = ax1.bar(range(len(algorithms)), auc_means, yerr=auc_stds, capsize=5, alpha=0.8, edgecolor='black')
    ax1.set_xticks(range(len(algorithms)))
    ax1.set_xticklabels(algorithms, rotation=45, ha='right')
    ax1.set_ylabel('Mean AUC Score')
    ax1.set_title('(a) Algorithm Performance - AUC')
    ax1.set_ylim([0.0, 1.0])
    ax1.grid(axis='y', alpha=0.3)
    for i, (bar, val, std) in enumerate(zip(bars1, auc_means, auc_stds)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=8)
    # F1 comparison
    ax2 = axes[1]
    f1_means = algo_stats[('spatial_f1_mean', 'mean')]
    f1_stds = algo_stats[('spatial_f1_mean', 'std')]
    bars2 = ax2.bar(range(len(algorithms)), f1_means, yerr=f1_stds, capsize=5, alpha=0.8, edgecolor='black', color='coral')
    ax2.set_xticks(range(len(algorithms)))
    ax2.set_xticklabels(algorithms, rotation=45, ha='right')
    ax2.set_ylabel('Mean F1 Score')
    ax2.set_title('(b) Algorithm Performance - F1')
    ax2.set_ylim([0.0, 1.0])
    ax2.grid(axis='y', alpha=0.3)
    for i, (bar, val, std) in enumerate(zip(bars2, f1_means, f1_stds)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig('figures/algorithm_comparison.png', bbox_inches='tight')
    plt.close()

# --- Feature Method Comparison (Original Style) ---
if os.path.exists(cv_results_path):
    df = pd.read_csv(cv_results_path)
    rf_data = df[df['algorithm'] == 'RandomForest'].copy()
    rf_data['feature_method'] = rf_data['configuration'].apply(lambda x: x.split('_hard_')[0].split('_random_')[0])
    method_order = rf_data.groupby('feature_method')['spatial_auc_mean'].mean().sort_values(ascending=False).index
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    # AUC by feature method
    ax1 = axes[0]
    auc_data = rf_data.groupby('feature_method')['spatial_auc_mean'].mean().reindex(method_order)
    auc_std = rf_data.groupby('feature_method')['spatial_auc_std'].mean().reindex(method_order)
    bars1 = ax1.barh(range(len(method_order)), auc_data, xerr=auc_std, capsize=8, alpha=0.95, edgecolor='navy', color=sns.color_palette('crest', len(method_order)))
    ax1.set_yticks(range(len(method_order)))
    ax1.set_yticklabels(method_order, fontsize=13)
    ax1.set_xlabel('Mean AUC Score', fontsize=14)
    ax1.set_title('(a) Feature Selection Method Performance - AUC (RandomForest)', fontsize=16, fontweight='bold')
    ax1.set_xlim([0.80, 1.02])
    ax1.grid(axis='x', alpha=0.25, linestyle='--')
    for i, (bar, val) in enumerate(zip(bars1, auc_data)):
        ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center', fontsize=12, fontweight='bold', color='black')
    # F1 by feature method
    ax2 = axes[1]
    f1_data = rf_data.groupby('feature_method')['spatial_f1_mean'].mean().reindex(method_order)
    f1_std = rf_data.groupby('feature_method')['spatial_f1_std'].mean().reindex(method_order)
    bars2 = ax2.barh(range(len(method_order)), f1_data, xerr=f1_std, capsize=8, alpha=0.95, edgecolor='darkred', color=sns.color_palette('flare', len(method_order)))
    ax2.set_yticks(range(len(method_order)))
    ax2.set_yticklabels(method_order, fontsize=13)
    ax2.set_xlabel('Mean F1 Score', fontsize=14)
    ax2.set_title('(b) Feature Selection Method Performance - F1 (RandomForest)', fontsize=16, fontweight='bold')
    ax2.set_xlim([0.0, 1.05])
    ax2.grid(axis='x', alpha=0.25, linestyle='--')
    for i, (bar, val) in enumerate(zip(bars2, f1_data)):
        ax2.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center', fontsize=12, fontweight='bold', color='black')
    plt.tight_layout(pad=2)
    plt.savefig('figures/auc_by_method.png', bbox_inches='tight')
    plt.close()

# --- Sampling Strategy Comparison (Original Style) ---
    # --- Performance Heatmap (Original Style) ---
    rf_data = df[df['algorithm'].isin(['RandomForest', 'GradientBoosting', 'LogisticRegression', 'MLP'])].copy()
    rf_data['feature_method'] = rf_data['configuration'].apply(lambda x: x.split('_hard_')[0].split('_random_')[0])
    key_methods = ['tree_based', 'ensemble', 'correlation', 'variance', 'rfe', 'mutual_info']
    rf_data = rf_data[rf_data['feature_method'].isin(key_methods)]
    auc_pivot = rf_data.pivot_table(values='spatial_auc_mean', index='feature_method', columns='algorithm', aggfunc='mean')
    f1_pivot = rf_data.pivot_table(values='spatial_f1_mean', index='feature_method', columns='algorithm', aggfunc='mean')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # AUC heatmap
    ax1 = axes[0]
    sns.heatmap(auc_pivot, annot=True, fmt='.3f', cmap='YlGnBu', vmin=0.75, vmax=1.0, ax=ax1, cbar_kws={'label': 'AUC Score'})
    ax1.set_title('(a) AUC Performance Heatmap')
    ax1.set_xlabel('Algorithm')
    ax1.set_ylabel('Feature Selection Method')
    # F1 heatmap
    ax2 = axes[1]
    sns.heatmap(f1_pivot, annot=True, fmt='.3f', cmap='YlOrRd', vmin=0.0, vmax=1.0, ax=ax2, cbar_kws={'label': 'F1 Score'})
    ax2.set_title('(b) F1 Performance Heatmap')
    ax2.set_xlabel('Algorithm')
    ax2.set_ylabel('Feature Selection Method')
    plt.tight_layout()
    plt.savefig('figures/heatmap_performance.png', bbox_inches='tight')
    plt.close()
if os.path.exists(cv_results_path):
    df = pd.read_csv(cv_results_path)
    balanced_configs = ['ensemble', 'correlation', 'variance', 'tree_based', 'baseline']
    sampling_data = []
    for config in balanced_configs:
        hard_neg = df[df['configuration'] == f'{config}_hard_negative']
        random_neg = df[df['configuration'] == f'{config}_random_negative']
        if len(hard_neg) > 0 and len(random_neg) > 0:
            for algo in ['RandomForest', 'GradientBoosting', 'LogisticRegression', 'MLP']:
                hard = hard_neg[hard_neg['algorithm'] == algo]
                rand = random_neg[random_neg['algorithm'] == algo]
                if len(hard) > 0 and len(rand) > 0:
                    sampling_data.append({
                        'config': config,
                        'algorithm': algo,
                        'hard_auc': hard['spatial_auc_mean'].values[0],
                        'random_auc': rand['spatial_auc_mean'].values[0],
                        'hard_f1': hard['spatial_f1_mean'].values[0],
                        'random_f1': rand['spatial_f1_mean'].values[0]
                    })
    sampling_df = pd.DataFrame(sampling_data)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # AUC comparison
    ax1 = axes[0]
    x = np.arange(len(sampling_df))
    width = 0.35
    bars1 = ax1.bar(x - width/2, sampling_df['hard_auc'], width, label='Hard Negative', alpha=0.8, edgecolor='black')
    bars2 = ax1.bar(x + width/2, sampling_df['random_auc'], width, label='Random Negative', alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('AUC Score')
    ax1.set_title('(a) Sampling Strategy Comparison - AUC')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{row['config']}\n{row['algorithm']}" for _, row in sampling_df.iterrows()], rotation=45, ha='right', fontsize=7)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0.75, 1.05])
    # F1 comparison
    ax2 = axes[1]
    bars3 = ax2.bar(x - width/2, sampling_df['hard_f1'], width, label='Hard Negative', alpha=0.8, edgecolor='black', color='coral')
    bars4 = ax2.bar(x + width/2, sampling_df['random_f1'], width, label='Random Negative', alpha=0.8, edgecolor='black', color='lightcoral')
    ax2.set_xlabel('Configuration')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('(b) Sampling Strategy Comparison - F1')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{row['config']}\n{row['algorithm']}" for _, row in sampling_df.iterrows()], rotation=45, ha='right', fontsize=7)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0.65, 1.05])
    plt.tight_layout()
    plt.savefig('figures/sampling_strategy_comparison.png', bbox_inches='tight')
    plt.close()

# --- Sampling Strategy Impact: Hard Negative vs Random ---
cv_results_path = 'corrected_spatial_cv_results/corrected_spatial_cv_results.csv'
if os.path.exists(cv_results_path):
    cv_df = pd.read_csv(cv_results_path)
    # Extract sampling_strategy from configuration
    cv_df['sampling_strategy'] = cv_df['configuration'].apply(lambda x: 'hard_negative' if 'hard_negative' in x else ('random_negative' if 'random_negative' in x else 'other'))
    plt.figure(figsize=(13,8))
    ax = sns.barplot(x='sampling_strategy', y='spatial_auc_mean', hue='algorithm', data=cv_df, errorbar='sd', palette='Set2')
    plt.title('AUC by Sampling Strategy and Algorithm', fontsize=16, fontweight='bold')
    plt.ylabel('Spatial AUC (mean)', fontsize=14)
    plt.xlabel('Sampling Strategy', fontsize=14)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    ax.legend(title='Algorithm', fontsize=12, title_fontsize=13, loc='best', frameon=True)
    ax.grid(axis='y', alpha=0.25, linestyle='--')
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.3f}',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=12, fontweight='bold', color='black', xytext=(0, 5), textcoords='offset points')
    plt.tight_layout(pad=2)
    plt.savefig('figures/auc_by_sampling_strategy.png', bbox_inches='tight')
    plt.close()



# --- Feature Category Breakdown for Each Method ---
for method in ['ensemble', 'tree_based', 'rfe', 'mutual_info', 'l1_regularization', 'correlation']:
    path = f'feature_selection_results/selected_features_{method}.csv'
    if os.path.exists(path):
        df = pd.read_csv(path)
        if 'category' in df.columns:
            cat_counts = df['category'].value_counts()
            plt.figure(figsize=(8,5))
            sns.barplot(x=cat_counts.index, y=cat_counts.values, palette='mako')
            plt.title(f'Selected Features by Category: {method.replace("_", " ").title()}')
            plt.xlabel('Category')
            plt.ylabel('Number of Features')
            plt.xticks(rotation=45)
            # Move legend outside plot if present
            ax = plt.gca()
            legend = ax.get_legend()
            if legend:
                legend.set_bbox_to_anchor((1.01, 1))
                legend.set_loc('upper left')
            plt.tight_layout(pad=0.5)
            plt.savefig(f'figures/features_by_category_{method}.png', bbox_inches='tight')
            plt.close()

# --- Model Stability: Std Dev/Error Bars for AUC/F1 ---
if os.path.exists(cv_results_path):
    plt.figure(figsize=(14,8))
    ax = sns.barplot(x='algorithm', y='spatial_auc_mean', hue='configuration', data=cv_df, errorbar='sd', palette='Set2')
    plt.title('Model Stability: AUC by Algorithm and Configuration')
    plt.ylabel('Spatial AUC (mean)')
    plt.xlabel('Algorithm')
    plt.xticks(rotation=60, ha='right', va='top', fontsize=12)
    # Move legend to right bottom corner and make it smaller
    legend = ax.legend(loc='lower right', fontsize=10, frameon=True)
    plt.tight_layout()
    plt.savefig('figures/model_stability_auc.png', bbox_inches='tight')
    plt.close()

# --- Performance vs. Number of Features ---
if os.path.exists(cv_results_path):
    fig, ax = plt.subplots(figsize=(14,8))
    sns.scatterplot(x='n_features', y='spatial_auc_mean', hue='algorithm', style='configuration', data=cv_df, palette='Set1', ax=ax, s=120)
    ax.set_title('Performance vs. Number of Features')
    ax.set_xlabel('Number of Features')
    ax.set_ylabel('Spatial AUC (mean)')
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticks(), rotation=60, ha='right', va='top', fontsize=12)
    # Move legend outside plot area in white space and make it smaller
    handles, labels = ax.get_legend_handles_labels()
    ax.get_legend().remove()  # Remove the legend from inside the plot
    legend = fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.18, 0.5), fontsize=10, frameon=True)
    # Move description outside plot area
    fig.text(0.5, 0.01, 'Algorithm: RandomForest, XGBoost, CatBoost, LightGBM; Configurations: hard_negative, random_negative', ha='center', va='bottom', fontsize=11, wrap=True)
    plt.tight_layout(rect=[0,0.03,1,1])
    plt.savefig('figures/performance_vs_n_features.png', bbox_inches='tight')
    plt.close()
# --- Tree-based vs Ensemble Feature Comparison ---
TREE_SELECTED_PATH = 'feature_selection_results/selected_features_tree_based.csv'
ENSEMBLE_SELECTED_PATH = 'feature_selection_results/selected_features_ensemble.csv'
TREE_IMPORTANCE_PATH = 'feature_selection_results/feature_importance_tree_based.csv'
ENSEMBLE_IMPORTANCE_PATH = 'feature_selection_results/feature_importance_ensemble.csv'
ENSEMBLE_PATH = ENSEMBLE_SELECTED_PATH
if all(os.path.exists(p) for p in [TREE_SELECTED_PATH, ENSEMBLE_SELECTED_PATH, TREE_IMPORTANCE_PATH, ENSEMBLE_IMPORTANCE_PATH]):
    tree_sel = pd.read_csv(TREE_SELECTED_PATH)
    ensemble_sel = pd.read_csv(ENSEMBLE_SELECTED_PATH)
    tree_imp = pd.read_csv(TREE_IMPORTANCE_PATH)
    ensemble_imp = pd.read_csv(ENSEMBLE_IMPORTANCE_PATH)
    # Rename method_count to importance for ensemble
    if 'method_count' in ensemble_imp.columns:
        ensemble_imp = ensemble_imp.rename(columns={'method_count': 'importance'})
    # Exclude leakage-prone features
    tree_sel = tree_sel[tree_sel['feature'].apply(lambda x: not any(pat in str(x).lower() for pat in LEAKAGE_PATTERNS))]
    ensemble_sel = ensemble_sel[ensemble_sel['feature'].apply(lambda x: not any(pat in str(x).lower() for pat in LEAKAGE_PATTERNS))]
    tree_imp = tree_imp[tree_imp['feature'].apply(lambda x: not any(pat in str(x).lower() for pat in LEAKAGE_PATTERNS))]
    ensemble_imp = ensemble_imp[ensemble_imp['feature'].apply(lambda x: not any(pat in str(x).lower() for pat in LEAKAGE_PATTERNS))]
    # Merge selected features with importances
    tree_df = pd.merge(tree_sel, tree_imp[['feature', 'importance']], on='feature', how='left')
    ensemble_df = pd.merge(ensemble_sel, ensemble_imp[['feature', 'importance']], on='feature', how='left')
    # Only compare features present in both methods
    common_features = set(tree_df['feature']).intersection(set(ensemble_df['feature']))
    tree_common = tree_df[tree_df['feature'].isin(common_features)].set_index('feature')
    ensemble_common = ensemble_df[ensemble_df['feature'].isin(common_features)].set_index('feature')
    # Drop features with missing importance
    comp_df = pd.DataFrame({
        'Tree Importance': tree_common['importance'],
        'Ensemble Importance': ensemble_common['importance']
    }).dropna()
    comp_df['Mean Importance'] = comp_df[['Tree Importance', 'Ensemble Importance']].mean(axis=1)
    comp_df = comp_df.sort_values('Mean Importance', ascending=False).head(20)
    if not comp_df.empty:
        # Normalize both importances to [0, 1]
        tree_norm = (comp_df['Tree Importance'] - comp_df['Tree Importance'].min()) / (comp_df['Tree Importance'].max() - comp_df['Tree Importance'].min())
        ensemble_norm = (comp_df['Ensemble Importance'] - comp_df['Ensemble Importance'].min()) / (comp_df['Ensemble Importance'].max() - comp_df['Ensemble Importance'].min())
        fig, ax = plt.subplots(figsize=(14, 10))
        feature_names = comp_df.index.tolist()
        y_pos = np.arange(len(feature_names))
        ax.barh(y_pos - 0.2, tree_norm, height=0.4, color="#1f77b4", label='Tree-based (normalized)')
        ax.barh(y_pos + 0.2, ensemble_norm, height=0.4, color="#9467bd", label='Ensemble (normalized)', alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.set_xlabel('Normalized Importance', fontsize=14)
        ax.set_ylabel('Feature', fontsize=14)
        ax.set_title('Top 20 Common Features: Tree-based vs Ensemble Importance (Normalized)', fontsize=16, fontweight='bold')
        ax.legend(loc='best', fontsize=13)
        plt.tight_layout()
        plt.savefig('figures/top20_tree_vs_ensemble.png', bbox_inches='tight')
        plt.close()
    else:
        print('No common features with valid importance for comparison. Skipping top20_tree_vs_ensemble.png.')



# Ensure 'figures' directory exists before saving any figures
os.makedirs('figures', exist_ok=True)

if os.path.exists(ENSEMBLE_SELECTED_PATH):
    df = pd.read_csv(ENSEMBLE_SELECTED_PATH)
    # If category column missing, merge from tree_based
    if 'category' not in df.columns:
        if os.path.exists(TREE_SELECTED_PATH):
            tree_df = pd.read_csv(TREE_SELECTED_PATH)
            # If no 'category' column, infer from feature names (example logic)
            if 'category' not in tree_df.columns:
                def infer_category(feat):
                    if 'TEMP' in feat: return 'Temperature'
                    if 'PRES' in feat: return 'Pressure'
                    if 'WSPD' in feat: return 'Wind Speed'
                    if 'CBD' in feat: return 'Canopy Bulk Density'
                    if 'EVH' in feat: return 'Vegetation Height'
                    if 'CH' in feat: return 'Canopy Height'
                    if 'CC' in feat: return 'Canopy Cover'
                    if 'CBH' in feat: return 'Canopy Base Height'
                    if 'ELEV' in feat: return 'Elevation'
                    if 'SLP' in feat: return 'Slope'
                    if 'EVT' in feat: return 'Vegetation Type'
                    if 'EVC' in feat: return 'Vegetation Cover'
                    return 'Other'
                tree_df['category'] = tree_df['feature'].apply(infer_category)
            tree_cat_map = tree_df[['feature', 'category']]
            df = df.merge(tree_cat_map, on='feature', how='left')
            df['category'] = df['category'].fillna('Unknown')
        else:
            df['category'] = 'Unknown'
    # Exclude leakage-prone features
    df = df[df['feature'].apply(lambda x: not any(pat in str(x).lower() for pat in LEAKAGE_PATTERNS))]
    # Top 20 features by importance (if available)
    if 'importance' in df.columns:
        top20 = df.sort_values('importance', ascending=True).tail(20)
        plt.figure(figsize=(14, 10))
        y_pos = np.arange(len(top20['feature']))
        plt.barh(y_pos, top20['importance'], color='mediumseagreen', height=0.6)
        plt.yticks(y_pos, top20['feature'])
        method = 'Ensemble Feature Selection' if 'ensemble' in ENSEMBLE_PATH else 'Unknown Method'
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Top 20 Selected Features (Leakage Excluded)\nMethod: {method}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('figures/top20_features.png', bbox_inches='tight')
        plt.close()
    else:
        top20 = df.head(20)
        plt.figure(figsize=(10, 7))
        ax = sns.barplot(y='feature', x=range(1, len(top20)+1), data=top20, hue='feature', palette='viridis', legend=False)
        plt.title('Top 20 Selected Features (Leakage Excluded)')
        plt.xlabel('Feature Rank')
        plt.ylabel('Feature')
        # No text annotation for clarity
        plt.tight_layout(pad=1.5)
        plt.savefig('figures/top20_features.png', bbox_inches='tight')
        plt.close()
    # Features by category
    if 'category' in df.columns and df['category'].notna().any():
        category_counts = df['category'].value_counts()
        plt.figure(figsize=(8, 5))
        sns.barplot(x=category_counts.index, y=category_counts.values, hue=category_counts.index, palette='mako', legend=False)
        plt.title('Selected Features by Category (Leakage Excluded)')
        plt.xlabel('Category')
        plt.ylabel('Number of Features')
        plt.xticks(rotation=45)
        ax = plt.gca()
        legend = ax.get_legend()
        if legend:
            legend.set_bbox_to_anchor((1.01, 1))
            legend.set_loc('upper left')
        plt.tight_layout(pad=0.5)
        plt.savefig('figures/features_by_category.png', bbox_inches='tight')
        plt.close()
    else:
        print('No category information available in ensemble features. Skipping features_by_category.png.')
else:
    print(f"Ensemble feature file not found: {ENSEMBLE_PATH}")

# --- Add other figure generation code below ---
# --- Subgroup and Confusion Matrix Analysis (Framework Logic) ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# Load best configuration results (example path, adjust as needed)
conf_matrix_path = 'corrected_spatial_cv_results/corrected_spatial_cv_confusion_matrices.csv'
if os.path.exists(conf_matrix_path):
    df_cm = pd.read_csv(conf_matrix_path)
    # Example: Use best config and algorithm (can be parameterized)
    best_config = df_cm['configuration'].value_counts().idxmax()
    best_algo = df_cm[df_cm['configuration'] == best_config]['algorithm'].value_counts().idxmax()
    filtered = df_cm[(df_cm['configuration'] == best_config) & (df_cm['algorithm'] == best_algo)]
    TN = filtered['tn'].mean()
    FP = filtered['fp'].mean()
    FN = filtered['fn'].mean()
    TP = filtered['tp'].mean()
    cm = np.array([[TN, FP], [FN, TP]])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(6,5))
    ax = sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues', cbar=False,
                     xticklabels=['Predicted Non-Fire', 'Predicted Fire'],
                     yticklabels=['Actual Non-Fire', 'Actual Fire'])
    for i in range(2):
        for j in range(2):
            val = cm_norm[i, j]
            ax.text(j+0.5, i+0.5 - 0.25, f"{val:.1%}", ha='center', color='black', fontsize=9)
    plt.title(f'Confusion Matrix (counts)\n{best_algo} — {best_config}')
    plt.tight_layout()
    plt.savefig('figures/confusion_matrix_framework.png', dpi=300)
    plt.close()

# Subgroup analysis (example, adjust bins and factors as needed)
subgroup_path = 'corrected_spatial_cv_results/subgroup_analysis/subgroup_error_metrics.csv'
if os.path.exists(subgroup_path):
    df_sub = pd.read_csv(subgroup_path)
    for factor in ['Temperature', 'Vegetation', 'Region']:
        factor_data = df_sub[df_sub['Factor'] == factor].sort_values('Bin')
        # FPR
        plt.figure(figsize=(10, 6))
        sns.barplot(data=factor_data, x='Bin', y='FPR', hue='Algorithm')
        plt.title(f'False Positive Rate by {factor}')
        plt.ylabel('False Positive Rate')
        plt.xlabel(factor)
        plt.tight_layout()
        plt.savefig(f'figures/fpr_by_{factor.lower()}_framework.png')
        plt.close()
        # FNR
        plt.figure(figsize=(10, 6))
        sns.barplot(data=factor_data, x='Bin', y='FNR', hue='Algorithm')
        plt.title(f'False Negative Rate by {factor}')
        plt.ylabel('False Negative Rate (Missed Fires)')
        plt.xlabel(factor)
        plt.tight_layout()
        plt.savefig(f'figures/fnr_by_{factor.lower()}_framework.png')
        plt.close()
# --- Integrated Subgroup Analysis using run_subgroup_analysis.py ---
import sys
sys.path.append('.')
import run_subgroup_analysis as subgroup

# Override output directory for subgroup plots to 'figures'
from pathlib import Path
subgroup.OUTPUT_DIR = Path('figures')

df_subgroup = subgroup.load_data()
full_results = subgroup.run_analysis(df_subgroup)
metrics_df = subgroup.analyze_subgroups(full_results)
subgroup.plot_results(metrics_df)
# --- Integrated Subgroup Analysis ---
import sys
sys.path.append('.')
import run_subgroup_analysis as subgroup

# Override output directory for subgroup plots to 'figures'
from pathlib import Path
subgroup.OUTPUT_DIR = Path('figures')

df_subgroup = subgroup.load_data()
full_results = subgroup.run_analysis(df_subgroup)
metrics_df = subgroup.analyze_subgroups(full_results)
subgroup.plot_results(metrics_df)

# --- Hard Negative Sampling Figure ---
import sys
sys.path.append('github')
from create_hard_negative_sampling_figure import create_hard_negative_comparison_figure
create_hard_negative_comparison_figure()

# --- Subgroup Analysis Figures ---

# Update DATASET_PATH for subgroup analysis to correct location
import sys
sys.path.append('github')
import run_subgroup_analysis as subgroup
subgroup.DATASET_PATH = '/home/u427312/wildfire_project/github/corrected_hard_negative_results/tree_based_random_negative_dataset.csv'
df_subgroup = subgroup.load_data()
full_results = subgroup.run_analysis(df_subgroup)
metrics_df = subgroup.analyze_subgroups(full_results)
subgroup.plot_results(metrics_df)

# --- Confusion Matrix and Error Breakdown Figures ---
import sys
sys.path.append('github')
import create_confusion_figures
# This script runs on import and saves figures automatically

# --- Spatial CV Summary Figures (Top 10 Configurations) ---
import sys
sys.path.append('github')
import create_spatial_cv_summary_figures as cv_summary
# This generates the Top 10 Configurations bar chart
# Note: Algorithm comparison already generated above as algorithm_comparison.png
