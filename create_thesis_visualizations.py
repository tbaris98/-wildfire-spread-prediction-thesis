"""
Create Publication-Quality Visualizations for Thesis
====================================================
Generates all figures for the Results section based on spatial CV results.

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
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.figsize'] = (10, 6)

class ThesisVisualizations:
    def __init__(self, results_file='corrected_spatial_cv_results/corrected_spatial_cv_results.csv',
                 output_dir='thesis_visualizations'):
        self.results_file = results_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.df = None
        
    def load_data(self):
        """Load results data"""
        print(f"Loading data from {self.results_file}...")
        self.df = pd.read_csv(self.results_file)
        print(f"Loaded {len(self.df)} configurations")
        
    def figure1_algorithm_comparison(self):
        """Figure 1: Overall Algorithm Performance Comparison"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Calculate mean performance by algorithm
        algo_stats = self.df.groupby('algorithm').agg({
            'spatial_auc_mean': ['mean', 'std'],
            'spatial_f1_mean': ['mean', 'std']
        }).reset_index()
        
        # AUC comparison
        ax1 = axes[0]
        algorithms = algo_stats['algorithm']
        auc_means = algo_stats[('spatial_auc_mean', 'mean')]
        auc_stds = algo_stats[('spatial_auc_mean', 'std')]
        
        bars1 = ax1.bar(range(len(algorithms)), auc_means, yerr=auc_stds, 
                       capsize=5, alpha=0.8, edgecolor='black')
        ax1.set_xticks(range(len(algorithms)))
        ax1.set_xticklabels(algorithms, rotation=45, ha='right')
        ax1.set_ylabel('Mean AUC Score')
        ax1.set_title('(a) Algorithm Performance - AUC')
        ax1.set_ylim([0.75, 0.95])
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, val, std) in enumerate(zip(bars1, auc_means, auc_stds)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=8)
        
        # F1 comparison
        ax2 = axes[1]
        f1_means = algo_stats[('spatial_f1_mean', 'mean')]
        f1_stds = algo_stats[('spatial_f1_mean', 'std')]
        
        bars2 = ax2.bar(range(len(algorithms)), f1_means, yerr=f1_stds,
                       capsize=5, alpha=0.8, edgecolor='black', color='coral')
        ax2.set_xticks(range(len(algorithms)))
        ax2.set_xticklabels(algorithms, rotation=45, ha='right')
        ax2.set_ylabel('Mean F1 Score')
        ax2.set_title('(b) Algorithm Performance - F1')
        ax2.set_ylim([0.0, 0.7])
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, (bar, val, std) in enumerate(zip(bars2, f1_means, f1_stds)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        output_file = self.output_dir / 'figure1_algorithm_comparison.png'
        plt.savefig(output_file, bbox_inches='tight')
        print(f"✅ Saved: {output_file}")
        plt.close()
        
    def figure2_feature_method_comparison(self):
        """Figure 2: Feature Selection Method Performance (RandomForest only)"""
        # Filter for RandomForest only
        rf_data = self.df[self.df['algorithm'] == 'RandomForest'].copy()
        
        # Extract feature method from configuration name
        rf_data['feature_method'] = rf_data['configuration'].apply(
            lambda x: x.split('_hard_')[0].split('_random_')[0]
        )
        
        # Sort by AUC
        method_order = rf_data.groupby('feature_method')['spatial_auc_mean'].mean().sort_values(ascending=False).index
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # AUC by feature method
        ax1 = axes[0]
        auc_data = rf_data.groupby('feature_method')['spatial_auc_mean'].mean().reindex(method_order)
        auc_std = rf_data.groupby('feature_method')['spatial_auc_std'].mean().reindex(method_order)
        
        bars1 = ax1.barh(range(len(method_order)), auc_data, xerr=auc_std,
                        capsize=5, alpha=0.8, edgecolor='black')
        ax1.set_yticks(range(len(method_order)))
        ax1.set_yticklabels(method_order)
        ax1.set_xlabel('Mean AUC Score')
        ax1.set_title('(a) Feature Selection Method Performance - AUC (RandomForest)')
        ax1.set_xlim([0.80, 1.02])
        ax1.grid(axis='x', alpha=0.3)
        
        # Color-code perfect scores
        for i, (bar, val) in enumerate(zip(bars1, auc_data)):
            if val >= 0.999:
                bar.set_color('green')
                bar.set_alpha(0.9)
            ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=8)
        
        # F1 by feature method
        ax2 = axes[1]
        f1_data = rf_data.groupby('feature_method')['spatial_f1_mean'].mean().reindex(method_order)
        f1_std = rf_data.groupby('feature_method')['spatial_f1_std'].mean().reindex(method_order)
        
        bars2 = ax2.barh(range(len(method_order)), f1_data, xerr=f1_std,
                        capsize=5, alpha=0.8, edgecolor='black', color='coral')
        ax2.set_yticks(range(len(method_order)))
        ax2.set_yticklabels(method_order)
        ax2.set_xlabel('Mean F1 Score')
        ax2.set_title('(b) Feature Selection Method Performance - F1 (RandomForest)')
        ax2.set_xlim([0.0, 1.05])
        ax2.grid(axis='x', alpha=0.3)
        
        # Color-code perfect scores
        for i, (bar, val) in enumerate(zip(bars2, f1_data)):
            if val >= 0.999:
                bar.set_color('green')
                bar.set_alpha(0.9)
            ax2.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=8)
        
        plt.tight_layout()
        output_file = self.output_dir / 'figure2_feature_method_comparison.png'
        plt.savefig(output_file, bbox_inches='tight')
        print(f"✅ Saved: {output_file}")
        plt.close()
        
    def figure3_sampling_strategy_comparison(self):
        """Figure 3: Hard Negative vs Random Negative Sampling"""
        # Filter for balanced datasets only
        balanced_configs = ['ensemble', 'correlation', 'variance', 'tree_based', 'baseline']
        
        sampling_data = []
        for config in balanced_configs:
            hard_neg = self.df[self.df['configuration'] == f'{config}_hard_negative']
            random_neg = self.df[self.df['configuration'] == f'{config}_random_negative']
            
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
        
        bars1 = ax1.bar(x - width/2, sampling_df['hard_auc'], width, label='Hard Negative',
                       alpha=0.8, edgecolor='black')
        bars2 = ax1.bar(x + width/2, sampling_df['random_auc'], width, label='Random Negative',
                       alpha=0.8, edgecolor='black')
        
        ax1.set_xlabel('Configuration')
        ax1.set_ylabel('AUC Score')
        ax1.set_title('(a) Sampling Strategy Comparison - AUC')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"{row['config']}\n{row['algorithm']}" 
                            for _, row in sampling_df.iterrows()], 
                           rotation=45, ha='right', fontsize=7)
        ax1.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=True)
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim([0.75, 1.05])
        
        # F1 comparison
        ax2 = axes[1]
        bars3 = ax2.bar(x - width/2, sampling_df['hard_f1'], width, label='Hard Negative',
                       alpha=0.8, edgecolor='black', color='coral')
        bars4 = ax2.bar(x + width/2, sampling_df['random_f1'], width, label='Random Negative',
                       alpha=0.8, edgecolor='black', color='lightcoral')
        
        ax2.set_xlabel('Configuration')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('(b) Sampling Strategy Comparison - F1')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"{row['config']}\n{row['algorithm']}" 
                            for _, row in sampling_df.iterrows()], 
                           rotation=45, ha='right', fontsize=7)
        ax2.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=True)
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim([0.65, 1.05])
        
        plt.tight_layout()
        output_file = self.output_dir / 'figure3_sampling_strategy_comparison.png'
        plt.savefig(output_file, bbox_inches='tight')
        print(f"✅ Saved: {output_file}")
        plt.close()
        
    def figure4_balanced_vs_imbalanced(self):
        """Figure 4: Impact of Class Imbalance"""
        # Compare balanced vs imbalanced for key configurations
        comparisons = []
        
        for config in ['tree_based', 'ensemble', 'correlation']:
            # Imbalanced (original)
            imb = self.df[(self.df['configuration'] == config) & 
                         (self.df['algorithm'] == 'RandomForest')]
            # Balanced (hard negative)
            bal = self.df[(self.df['configuration'] == f'{config}_hard_negative') & 
                         (self.df['algorithm'] == 'RandomForest')]
            
            if len(imb) > 0 and len(bal) > 0:
                comparisons.append({
                    'config': config,
                    'imbalanced_auc': imb['spatial_auc_mean'].values[0],
                    'balanced_auc': bal['spatial_auc_mean'].values[0],
                    'imbalanced_f1': imb['spatial_f1_mean'].values[0],
                    'balanced_f1': bal['spatial_f1_mean'].values[0],
                    'imbalanced_samples': imb['n_samples'].values[0],
                    'balanced_samples': bal['n_samples'].values[0]
                })
        
        comp_df = pd.DataFrame(comparisons)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # AUC comparison
        ax1 = axes[0]
        x = np.arange(len(comp_df))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, comp_df['imbalanced_auc'], width,
                       label='Imbalanced (17.8M samples)', alpha=0.8, edgecolor='black')
        bars2 = ax1.bar(x + width/2, comp_df['balanced_auc'], width,
                       label='Balanced (200K samples)', alpha=0.8, edgecolor='black', color='green')
        
        ax1.set_ylabel('AUC Score')
        ax1.set_title('(a) Impact of Class Balance on AUC (RandomForest)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(comp_df['config'], rotation=45, ha='right')
        ax1.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=True)
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim([0.80, 1.05])
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # F1 comparison (dramatic difference)
        ax2 = axes[1]
        bars3 = ax2.bar(x - width/2, comp_df['imbalanced_f1'], width,
                       label='Imbalanced (17.8M samples)', alpha=0.8, edgecolor='black', color='red')
        bars4 = ax2.bar(x + width/2, comp_df['balanced_f1'], width,
                       label='Balanced (200K samples)', alpha=0.8, edgecolor='black', color='green')
        
        ax2.set_ylabel('F1 Score')
        ax2.set_title('(b) Impact of Class Balance on F1 (RandomForest)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(comp_df['config'], rotation=45, ha='right')
        ax2.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=True)
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim([0.0, 1.1])
        
        # Add value labels
        for bars in [bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        output_file = self.output_dir / 'figure4_balanced_vs_imbalanced.png'
        plt.savefig(output_file, bbox_inches='tight')
        print(f"✅ Saved: {output_file}")
        plt.close()
        
    def figure5_heatmap_performance(self):
        """Figure 5: Performance Heatmap (Algorithm × Feature Method)"""
        # Create pivot tables for heatmap
        rf_data = self.df[self.df['algorithm'].isin(['RandomForest', 'GradientBoosting', 
                                                      'LogisticRegression', 'MLP'])].copy()
        
        # Extract feature method
        rf_data['feature_method'] = rf_data['configuration'].apply(
            lambda x: x.split('_hard_')[0].split('_random_')[0]
        )
        
        # Filter to key feature methods
        key_methods = ['tree_based', 'ensemble', 'correlation', 'variance', 'rfe', 'mutual_info']
        rf_data = rf_data[rf_data['feature_method'].isin(key_methods)]
        
        # Create pivot for AUC
        auc_pivot = rf_data.pivot_table(values='spatial_auc_mean', 
                                        index='feature_method',
                                        columns='algorithm',
                                        aggfunc='mean')
        
        # Create pivot for F1
        f1_pivot = rf_data.pivot_table(values='spatial_f1_mean',
                                       index='feature_method',
                                       columns='algorithm',
                                       aggfunc='mean')
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # AUC heatmap
        ax1 = axes[0]
        sns.heatmap(auc_pivot, annot=True, fmt='.3f', cmap='YlGnBu', 
                   vmin=0.75, vmax=1.0, ax=ax1, cbar_kws={'label': 'AUC Score'})
        ax1.set_title('(a) AUC Performance Heatmap')
        ax1.set_xlabel('Algorithm')
        ax1.set_ylabel('Feature Selection Method')
        
        # F1 heatmap
        ax2 = axes[1]
        sns.heatmap(f1_pivot, annot=True, fmt='.3f', cmap='YlOrRd',
                   vmin=0.0, vmax=1.0, ax=ax2, cbar_kws={'label': 'F1 Score'})
        ax2.set_title('(b) F1 Performance Heatmap')
        ax2.set_xlabel('Algorithm')
        ax2.set_ylabel('Feature Selection Method')
        
        plt.tight_layout()
        output_file = self.output_dir / 'figure5_heatmap_performance.png'
        plt.savefig(output_file, bbox_inches='tight')
        print(f"✅ Saved: {output_file}")
        plt.close()
        
    def figure6_feature_efficiency(self):
        """Figure 6: Feature Efficiency Analysis"""
        # Get RandomForest results for different feature methods
        rf_data = self.df[self.df['algorithm'] == 'RandomForest'].copy()
        rf_data['feature_method'] = rf_data['configuration'].apply(
            lambda x: x.split('_hard_')[0].split('_random_')[0]
        )
        
        # Aggregate by feature method
        efficiency_data = rf_data.groupby('feature_method').agg({
            'n_features': 'first',
            'spatial_auc_mean': 'mean',
            'spatial_f1_mean': 'mean'
        }).reset_index()
        
        # Calculate efficiency metric
        efficiency_data['efficiency_score'] = (efficiency_data['spatial_auc_mean'] + 
                                               efficiency_data['spatial_f1_mean']) / 2
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Scatter plot
        scatter = ax.scatter(efficiency_data['n_features'], 
                           efficiency_data['efficiency_score'],
                           s=200, alpha=0.6, edgecolors='black', linewidth=2)
        
        # Annotate points
        for _, row in efficiency_data.iterrows():
            ax.annotate(row['feature_method'], 
                       (row['n_features'], row['efficiency_score']),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax.set_xlabel('Number of Features')
        ax.set_ylabel('Efficiency Score (Mean of AUC and F1)')
        ax.set_title('Feature Efficiency: Performance vs Complexity')
        ax.grid(True, alpha=0.3)
        
        # Add efficiency zones
        ax.axhline(y=0.85, color='green', linestyle='--', alpha=0.3, label='High Performance (>0.85)')
        ax.axvline(x=30, color='blue', linestyle='--', alpha=0.3, label='Low Complexity (<30 features)')
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=True)
        
        plt.tight_layout()
        output_file = self.output_dir / 'figure6_feature_efficiency.png'
        plt.savefig(output_file, bbox_inches='tight')
        print(f"✅ Saved: {output_file}")
        plt.close()
        
    def generate_all_figures(self):
        """Generate all thesis figures"""
        print("=" * 60)
        print("GENERATING THESIS VISUALIZATIONS")
        print("=" * 60)
        
        self.load_data()
        
        print("\nGenerating Figure 1: Algorithm Comparison...")
        self.figure1_algorithm_comparison()
        
        print("Generating Figure 2: Feature Method Comparison...")
        self.figure2_feature_method_comparison()
        
        print("Generating Figure 3: Sampling Strategy Comparison...")
        self.figure3_sampling_strategy_comparison()
        
        print("Generating Figure 4: Balanced vs Imbalanced...")
        self.figure4_balanced_vs_imbalanced()
        
        print("Generating Figure 5: Performance Heatmap...")
        self.figure5_heatmap_performance()
        
        print("Generating Figure 6: Feature Efficiency...")
        self.figure6_feature_efficiency()
        
        print("\n" + "=" * 60)
        print("✅ ALL FIGURES GENERATED SUCCESSFULLY")
        print(f"📁 Output directory: {self.output_dir.absolute()}")
        print("=" * 60)

if __name__ == "__main__":
    visualizer = ThesisVisualizations()
    visualizer.generate_all_figures()
