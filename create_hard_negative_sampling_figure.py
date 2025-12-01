"""
Create Figure for Hard Negative Sampling Section
================================================
Compares hard negative vs random negative sampling strategies

Author: Tuna Baris Unal
Date: November 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10

def create_hard_negative_comparison_figure():
    """Create comparison figure for hard negative vs random negative sampling"""
    
    # Load data
    print("Loading hard negative sampling results...")
    results_file = 'corrected_hard_negative_results/corrected_hard_negative_spatial_cv_results.csv'
    df = pd.read_csv(results_file)
    print(f"Loaded {len(df)} configurations")
    
    # Extract only CLEAN feature methods (exclude tree and baseline - contain data leakage)
    # Use ensemble, correlation, and variance which are leakage-free
    clean_methods = ['ensemble', 'correlation', 'variance']
    clean_df = df[df['feature_method'].isin(clean_methods)].copy()
    
    # Calculate average performance across clean methods
    avg_performance = clean_df.groupby(['algorithm', 'sampling_strategy']).agg({
        'spatial_auc_mean': 'mean',
        'spatial_auc_std': 'mean',
        'spatial_f1_mean': 'mean',
        'spatial_f1_std': 'mean'
    }).reset_index()
    
    print(f"\nUsing {len(clean_methods)} clean feature methods: {', '.join(clean_methods)}")
    print(f"Total configurations: {len(avg_performance)}")
    
    # Create figure with 2 subplots (AUC and F1 comparison)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Define colors for strategies
    colors = {
        'hard_negative': '#E74C3C',  # Red
        'random_negative': '#3498DB'  # Blue
    }
    
    # Panel (a): AUC Comparison by Algorithm
    ax1 = axes[0]
    
    # Prepare data for grouped bar plot
    strategies = avg_performance['sampling_strategy'].unique()
    algorithms = avg_performance['algorithm'].unique()
    x = np.arange(len(algorithms))
    width = 0.35
    
    for i, strategy in enumerate(strategies):
        strategy_data = avg_performance[avg_performance['sampling_strategy'] == strategy]
        means = [strategy_data[strategy_data['algorithm'] == alg]['spatial_auc_mean'].values[0] 
                for alg in algorithms]
        stds = [strategy_data[strategy_data['algorithm'] == alg]['spatial_auc_std'].values[0] 
               for alg in algorithms]
        
        offset = width * (i - 0.5)
        bars = ax1.bar(x + offset, means, width, 
                      label=strategy.replace('_', ' ').title(),
                      color=colors[strategy], alpha=0.8, yerr=stds, capsize=4)
        
        # Add value labels on bars
        for j, (bar, val) in enumerate(zip(bars, means)):
            ax1.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax1.set_xlabel('Algorithm', fontweight='bold')
    ax1.set_ylabel('Spatial AUC (mean)', fontweight='bold')
    ax1.set_title('(a) AUC Performance: Hard Negative vs Random Sampling', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(algorithms, rotation=15, ha='right')
    ax1.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=True)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim([0.70, 0.90])
    ax1.set_axisbelow(True)
    
    # Panel (b): F1 Comparison by Algorithm
    ax2 = axes[1]
    
    for i, strategy in enumerate(strategies):
        strategy_data = avg_performance[avg_performance['sampling_strategy'] == strategy]
        means = [strategy_data[strategy_data['algorithm'] == alg]['spatial_f1_mean'].values[0] 
                for alg in algorithms]
        stds = [strategy_data[strategy_data['algorithm'] == alg]['spatial_f1_std'].values[0] 
               for alg in algorithms]
        
        offset = width * (i - 0.5)
        bars = ax2.bar(x + offset, means, width, 
                      label=strategy.replace('_', ' ').title(),
                      color=colors[strategy], alpha=0.8, yerr=stds, capsize=4)
        
        # Add value labels on bars
        for j, (bar, val) in enumerate(zip(bars, means)):
            ax2.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax2.set_xlabel('Algorithm', fontweight='bold')
    ax2.set_ylabel('Spatial F1 Score (mean)', fontweight='bold')
    ax2.set_title('(b) F1 Performance: Hard Negative vs Random Sampling', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(algorithms, rotation=15, ha='right')
    ax2.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=True)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim([0.65, 0.85])
    ax2.set_axisbelow(True)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path('thesis_visualizations')
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / 'figure_hard_negative_comparison.png'
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"\n✅ Saved: {output_file}")
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("HARD NEGATIVE SAMPLING COMPARISON (Averaged Across Clean Methods)")
    print("="*60)
    
    for strategy in strategies:
        print(f"\n{strategy.replace('_', ' ').title()}:")
        strategy_data = avg_performance[avg_performance['sampling_strategy'] == strategy]
        for alg in algorithms:
            alg_data = strategy_data[strategy_data['algorithm'] == alg]
            if len(alg_data) > 0:
                auc = alg_data['spatial_auc_mean'].values[0]
                f1 = alg_data['spatial_f1_mean'].values[0]
                print(f"  {alg:20s}: AUC={auc:.4f}, F1={f1:.4f}")
    
    # Calculate average improvement
    print("\n" + "="*60)
    print("AVERAGE PERFORMANCE COMPARISON:")
    print("="*60)
    for strategy in strategies:
        strategy_data = avg_performance[avg_performance['sampling_strategy'] == strategy]
        mean_auc = strategy_data['spatial_auc_mean'].mean()
        mean_f1 = strategy_data['spatial_f1_mean'].mean()
        print(f"{strategy.replace('_', ' ').title():20s}: Avg AUC={mean_auc:.4f}, Avg F1={mean_f1:.4f}")
    
    # Performance difference
    hard_auc = avg_performance[avg_performance['sampling_strategy'] == 'hard_negative']['spatial_auc_mean'].mean()
    rand_auc = avg_performance[avg_performance['sampling_strategy'] == 'random_negative']['spatial_auc_mean'].mean()
    hard_f1 = avg_performance[avg_performance['sampling_strategy'] == 'hard_negative']['spatial_f1_mean'].mean()
    rand_f1 = avg_performance[avg_performance['sampling_strategy'] == 'random_negative']['spatial_f1_mean'].mean()
    
    auc_diff = hard_auc - rand_auc
    f1_diff = hard_f1 - rand_f1
    
    print(f"\nPerformance Difference (Hard - Random):")
    print(f"  AUC: {auc_diff:+.4f} ({auc_diff/rand_auc*100:+.2f}%)")
    print(f"  F1:  {f1_diff:+.4f} ({f1_diff/rand_f1*100:+.2f}%)")
    print("="*60)

if __name__ == "__main__":
    create_hard_negative_comparison_figure()
