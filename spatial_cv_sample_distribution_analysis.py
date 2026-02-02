"""
Spatial Cross-Validation Sample Distribution Analysis
=====================================================

Analyzes probability distributions of hard negative vs random negative samples
using spatial cross-validation to maintain methodological consistency with thesis.

Addresses professor's feedback:
"It would be interesting to see the distribution of samples categorised by 
positive, hard-negative, negative and perhaps the distribution of sample 
'difficulty' (uncertainty). Is there a clear difference between hard-negative 
and negative and could this explain the similar results?"

Author: DSS Thesis
Date: 2026-02-01
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class SpatialCVSampleDistributionAnalyzer:
    """
    Analyzes sample probability distributions using spatial cross-validation.
    
    Loads balanced datasets from corrected_hard_negative_results/ and uses
    StratifiedGroupKFold to generate out-of-sample predicted probabilities,
    maintaining consistency with thesis spatial CV methodology.
    """
    
    def __init__(self, results_dir='corrected_hard_negative_results', 
                 output_dir='sample_distribution_results',
                 n_folds=3, random_state=42):
        """
        Initialize analyzer.
        
        Parameters:
        -----------
        results_dir : str
            Directory containing balanced datasets
        output_dir : str
            Directory to save results
        n_folds : int
            Number of spatial CV folds (default: 3, matches thesis)
        random_state : int
            Random seed for reproducibility
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.n_folds = n_folds
        self.random_state = random_state
        
        # Feature selection methods to analyze
        self.methods = ['ensemble', 'tree_based', 'correlation', 'variance', 'baseline']
        
        print(f"Initialized SpatialCVSampleDistributionAnalyzer")
        print(f"Results directory: {self.results_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Spatial CV folds: {n_folds}")
    
    def load_balanced_datasets(self, method):
        """
        Load hard negative and random negative balanced datasets.
        
        Parameters:
        -----------
        method : str
            Feature selection method (ensemble, tree_based, etc.)
        
        Returns:
        --------
        tuple: (hard_neg_df, random_df)
        """
        hard_neg_file = self.results_dir / f"{method}_hard_negative_dataset.csv"
        random_file = self.results_dir / f"{method}_random_negative_dataset.csv"
        
        print(f"\nLoading datasets for {method}...")
        
        if not hard_neg_file.exists():
            raise FileNotFoundError(f"Hard negative file not found: {hard_neg_file}")
        if not random_file.exists():
            raise FileNotFoundError(f"Random file not found: {random_file}")
        
        hard_neg_df = pd.read_csv(hard_neg_file)
        random_df = pd.read_csv(random_file)
        
        print(f"  Hard negative: {hard_neg_df.shape}")
        print(f"  Random: {random_df.shape}")
        print(f"  Hard neg class dist: {hard_neg_df['target'].value_counts().to_dict()}")
        print(f"  Random class dist: {random_df['target'].value_counts().to_dict()}")
        
        return hard_neg_df, random_df
    
    def create_geographic_groups(self, df, n_groups=10):
        """
        Create geographic groups based on latitude bands for spatial CV.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset with 'c_latitude' column
        n_groups : int
            Number of geographic groups (latitude bands)
        
        Returns:
        --------
        np.ndarray: Group labels for each sample
        """
        # Try different latitude column names
        lat_col = None
        for col in ['c_latitude', 'latitude', 'lat']:
            if col in df.columns:
                lat_col = col
                break
        
        if lat_col is None:
            # If latitude not available, use random groups
            print("  Warning: 'latitude' column not found, using random groups")
            return np.random.randint(0, n_groups, size=len(df))
        
        # Create latitude bands
        latitude = df[lat_col].values
        lat_bins = np.linspace(latitude.min(), latitude.max(), n_groups + 1)
        groups = np.digitize(latitude, lat_bins) - 1
        
        # Ensure groups are in valid range
        groups = np.clip(groups, 0, n_groups - 1)
        
        print(f"  Created {n_groups} geographic groups using '{lat_col}' (latitude bands)")
        print(f"  Group distribution: {np.bincount(groups)}")
        
        return groups
    
    def spatial_cv_probabilities(self, df, method_name, sampling_strategy):
        """
        Generate out-of-sample predicted probabilities using spatial CV.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset with features and target
        method_name : str
            Feature selection method name (for logging)
        sampling_strategy : str
            'hard_neg' or 'random' (for logging)
        
        Returns:
        --------
        dict: {
            'probabilities': array of out-of-sample probabilities,
            'true_labels': array of true labels,
            'indices': array of original indices,
            'fold_ids': array indicating which fold each sample was in test set,
            'auc_per_fold': list of AUC scores per fold
        }
        """
        print(f"\n  Generating spatial CV probabilities for {method_name}_{sampling_strategy}...")
        
        # Separate features and target
        target_col = 'target'
        feature_cols = [col for col in df.columns if col not in [
            target_col, 'latitude', 'longitude', 'c_latitude', 'date', 'fire_id', 
            'Unnamed: 0', 'index', 'level_0'
        ]]
        
        X = df[feature_cols].copy()
        y = df[target_col].values
        
        print(f"    Features: {len(feature_cols)}")
        print(f"    Samples: {len(X)}")
        
        # Check for NaN values and handle them
        if X.isnull().any().any():
            print(f"    Warning: Found NaN values, filling with column means")
            X = X.fillna(X.mean())
        
        # Check for infinite values
        if np.isinf(X.values).any():
            print(f"    Warning: Found infinite values, replacing with column max/min")
            X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())
        
        # Create geographic groups for spatial CV
        groups = self.create_geographic_groups(df, n_groups=10)
        
        # Validate that we have enough groups for CV
        n_unique_groups = len(np.unique(groups))
        if n_unique_groups < self.n_folds:
            print(f"    Warning: Only {n_unique_groups} groups, reducing folds from {self.n_folds} to {n_unique_groups}")
            n_folds = n_unique_groups
        else:
            n_folds = self.n_folds
        
        # Initialize storage for out-of-sample predictions
        all_probs = np.zeros(len(X))
        all_indices = np.arange(len(X))
        fold_ids = np.zeros(len(X), dtype=int)
        auc_per_fold = []
        
        # Spatial cross-validation
        skf = StratifiedGroupKFold(
            n_splits=n_folds, 
            shuffle=True, 
            random_state=self.random_state
        )
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y, groups)):
            print(f"    Fold {fold + 1}/{n_folds}: Train={len(train_idx)}, Test={len(test_idx)}")
            
            # Train Random Forest on training fold
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                n_jobs=-1,
                random_state=self.random_state + fold,
                class_weight='balanced'
            )
            
            rf.fit(X.iloc[train_idx], y[train_idx])
            
            # Predict on held-out test fold (out-of-sample)
            fold_probs = rf.predict_proba(X.iloc[test_idx])[:, 1]
            
            # Calculate fold AUC
            fold_auc = roc_auc_score(y[test_idx], fold_probs)
            auc_per_fold.append(fold_auc)
            print(f"      Fold AUC: {fold_auc:.4f}")
            
            # Store out-of-sample predictions
            all_probs[test_idx] = fold_probs
            fold_ids[test_idx] = fold + 1
        
        mean_auc = np.mean(auc_per_fold)
        std_auc = np.std(auc_per_fold)
        print(f"    Overall AUC: {mean_auc:.4f} ± {std_auc:.4f}")
        
        return {
            'probabilities': all_probs,
            'true_labels': y,
            'indices': all_indices,
            'fold_ids': fold_ids,
            'auc_per_fold': auc_per_fold,
            'mean_auc': mean_auc,
            'std_auc': std_auc
        }
    
    def analyze_method(self, method):
        """
        Analyze probability distributions for a specific feature selection method.
        
        Parameters:
        -----------
        method : str
            Feature selection method name
        
        Returns:
        --------
        dict: Analysis results for hard negative vs random negative
        """
        print(f"\n{'='*60}")
        print(f"Analyzing: {method}")
        print(f"{'='*60}")
        
        # Load balanced datasets
        hard_neg_df, random_df = self.load_balanced_datasets(method)
        
        # Generate spatial CV probabilities
        hard_results = self.spatial_cv_probabilities(hard_neg_df, method, 'hard_neg')
        random_results = self.spatial_cv_probabilities(random_df, method, 'random')
        
        # Separate by true class
        hard_pos_probs = hard_results['probabilities'][hard_results['true_labels'] == 1]
        hard_neg_probs = hard_results['probabilities'][hard_results['true_labels'] == 0]
        random_pos_probs = random_results['probabilities'][random_results['true_labels'] == 1]
        random_neg_probs = random_results['probabilities'][random_results['true_labels'] == 0]
        
        # Validate we have samples in each category
        if len(hard_pos_probs) == 0 or len(hard_neg_probs) == 0:
            raise ValueError(f"Empty probability arrays for {method}: hard_pos={len(hard_pos_probs)}, hard_neg={len(hard_neg_probs)}")
        if len(random_pos_probs) == 0 or len(random_neg_probs) == 0:
            raise ValueError(f"Empty probability arrays for {method}: random_pos={len(random_pos_probs)}, random_neg={len(random_neg_probs)}")
        
        print(f"\n  Probability summary:")
        print(f"    Hard neg positives: n={len(hard_pos_probs)}, mean={hard_pos_probs.mean():.4f}")
        print(f"    Hard negatives: n={len(hard_neg_probs)}, mean={hard_neg_probs.mean():.4f}")
        print(f"    Random positives: n={len(random_pos_probs)}, mean={random_pos_probs.mean():.4f}")
        print(f"    Random negatives: n={len(random_neg_probs)}, mean={random_neg_probs.mean():.4f}")
        
        return {
            'method': method,
            'hard_results': hard_results,
            'random_results': random_results,
            'hard_pos_probs': hard_pos_probs,
            'hard_neg_probs': hard_neg_probs,
            'random_pos_probs': random_pos_probs,
            'random_neg_probs': random_neg_probs
        }
    
    def calculate_statistics(self, probs, label):
        """
        Calculate comprehensive statistics for probability distribution.
        
        Parameters:
        -----------
        probs : np.ndarray
            Array of predicted probabilities
        label : str
            Label for the sample type
        
        Returns:
        --------
        dict: Statistics
        """
        # Validate input
        if len(probs) == 0:
            raise ValueError(f"Cannot calculate statistics for empty array: {label}")
        
        # Basic statistics
        stats_dict = {
            'label': label,
            'n': len(probs),
            'mean': np.mean(probs),
            'median': np.median(probs),
            'std': np.std(probs),
            'q25': np.percentile(probs, 25),
            'q75': np.percentile(probs, 75),
            'iqr': np.percentile(probs, 75) - np.percentile(probs, 25),
            'min': np.min(probs),
            'max': np.max(probs),
        }
        
        # Uncertainty metrics (distance from decision boundary 0.5)
        uncertainty = np.abs(probs - 0.5)
        stats_dict['mean_uncertainty'] = np.mean(uncertainty)
        stats_dict['median_uncertainty'] = np.median(uncertainty)
        
        # Percentage near decision boundary (within 0.1 of 0.5)
        near_boundary = np.sum((probs >= 0.4) & (probs <= 0.6))
        stats_dict['pct_near_boundary'] = (near_boundary / len(probs)) * 100
        
        return stats_dict
    
    def test_distribution_overlap(self, probs1, probs2, label1, label2):
        """
        Test statistical overlap between two probability distributions.
        
        Parameters:
        -----------
        probs1, probs2 : np.ndarray
            Arrays of predicted probabilities
        label1, label2 : str
            Labels for the two distributions
        
        Returns:
        --------
        dict: Statistical test results
        """
        # Kolmogorov-Smirnov test
        ks_stat, ks_pval = stats.ks_2samp(probs1, probs2)
        
        # Mann-Whitney U test
        mw_stat, mw_pval = stats.mannwhitneyu(probs1, probs2, alternative='two-sided')
        
        # Cohen's d effect size
        pooled_std = np.sqrt((np.std(probs1)**2 + np.std(probs2)**2) / 2)
        cohens_d = (np.mean(probs1) - np.mean(probs2)) / pooled_std if pooled_std > 0 else 0
        
        # Effect size interpretation
        if abs(cohens_d) < 0.2:
            effect_interp = "negligible"
        elif abs(cohens_d) < 0.5:
            effect_interp = "small"
        elif abs(cohens_d) < 0.8:
            effect_interp = "medium"
        else:
            effect_interp = "large"
        
        return {
            'label1': label1,
            'label2': label2,
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pval,
            'mw_statistic': mw_stat,
            'mw_pvalue': mw_pval,
            'cohens_d': cohens_d,
            'effect_size': effect_interp
        }
    
    def create_visualizations(self, analysis_results, method):
        """
        Create comprehensive visualizations comparing distributions.
        
        Parameters:
        -----------
        analysis_results : dict
            Results from analyze_method()
        method : str
            Feature selection method name
        """
        hard_pos = analysis_results['hard_pos_probs']
        hard_neg = analysis_results['hard_neg_probs']
        random_pos = analysis_results['random_pos_probs']
        random_neg = analysis_results['random_neg_probs']
        
        # Create 4-panel figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Sample Probability Distributions: {method.upper()}', 
                     fontsize=16, fontweight='bold')
        
        # Panel 1: All distributions overlaid
        ax1 = axes[0, 0]
        ax1.hist(hard_pos, bins=50, alpha=0.6, label='Hard Neg: Positives', 
                 color='red', density=True)
        ax1.hist(hard_neg, bins=50, alpha=0.6, label='Hard Neg: Negatives', 
                 color='blue', density=True)
        ax1.hist(random_pos, bins=50, alpha=0.4, label='Random: Positives', 
                 color='orange', density=True, histtype='step', linewidth=2)
        ax1.hist(random_neg, bins=50, alpha=0.4, label='Random: Negatives', 
                 color='cyan', density=True, histtype='step', linewidth=2)
        ax1.axvline(0.5, color='black', linestyle='--', linewidth=1, label='Decision Boundary')
        ax1.set_xlabel('Predicted Probability', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.set_title('All Sample Types Overlaid', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Hard negative vs Random negative (negatives only)
        ax2 = axes[0, 1]
        ax2.hist(hard_neg, bins=50, alpha=0.6, label='Hard Negatives', 
                 color='blue', density=True)
        ax2.hist(random_neg, bins=50, alpha=0.6, label='Random Negatives', 
                 color='cyan', density=True)
        ax2.axvline(0.5, color='black', linestyle='--', linewidth=1)
        ax2.set_xlabel('Predicted Probability', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.set_title('Negative Samples: Hard vs Random', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f"Hard: μ={hard_neg.mean():.3f}, σ={hard_neg.std():.3f}\n"
        stats_text += f"Random: μ={random_neg.mean():.3f}, σ={random_neg.std():.3f}"
        ax2.text(0.98, 0.97, stats_text, transform=ax2.transAxes, 
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Panel 3: Box plots
        ax3 = axes[1, 0]
        box_data = [hard_pos, hard_neg, random_pos, random_neg]
        box_labels = ['Hard\nPositives', 'Hard\nNegatives', 'Random\nPositives', 'Random\nNegatives']
        bp = ax3.boxplot(box_data, labels=box_labels, patch_artist=True,
                         showmeans=True, meanline=True)
        colors = ['red', 'blue', 'orange', 'cyan']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax3.axhline(0.5, color='black', linestyle='--', linewidth=1)
        ax3.set_ylabel('Predicted Probability', fontsize=12)
        ax3.set_title('Distribution Comparison (Box Plots)', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Panel 4: Uncertainty distribution (distance from 0.5)
        ax4 = axes[1, 1]
        hard_uncertainty = np.abs(hard_neg - 0.5)
        random_uncertainty = np.abs(random_neg - 0.5)
        ax4.hist(hard_uncertainty, bins=50, alpha=0.6, label='Hard Negatives', 
                 color='blue', density=True)
        ax4.hist(random_uncertainty, bins=50, alpha=0.6, label='Random Negatives', 
                 color='cyan', density=True)
        ax4.set_xlabel('Uncertainty (Distance from 0.5)', fontsize=12)
        ax4.set_ylabel('Density', fontsize=12)
        ax4.set_title('Sample Uncertainty Distribution', fontsize=13, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        # Add mean uncertainty text
        unc_text = f"Hard: μ={hard_uncertainty.mean():.3f}\n"
        unc_text += f"Random: μ={random_uncertainty.mean():.3f}"
        ax4.text(0.98, 0.97, unc_text, transform=ax4.transAxes, 
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save figure
        output_file = self.output_dir / f'{method}_distribution_4panel.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n  Saved 4-panel plot: {output_file}")
        plt.close()
        
        # Create CDF comparison
        self._create_cdf_plot(hard_neg, random_neg, method)
    
    def _create_cdf_plot(self, hard_neg, random_neg, method):
        """Create cumulative distribution function comparison."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate CDFs
        hard_sorted = np.sort(hard_neg)
        random_sorted = np.sort(random_neg)
        hard_cdf = np.arange(1, len(hard_sorted) + 1) / len(hard_sorted)
        random_cdf = np.arange(1, len(random_sorted) + 1) / len(random_sorted)
        
        # Plot CDFs
        ax.plot(hard_sorted, hard_cdf, label='Hard Negatives', linewidth=2, color='blue')
        ax.plot(random_sorted, random_cdf, label='Random Negatives', linewidth=2, color='cyan')
        ax.axvline(0.5, color='black', linestyle='--', linewidth=1, label='Decision Boundary')
        
        ax.set_xlabel('Predicted Probability', fontsize=12)
        ax.set_ylabel('Cumulative Probability', fontsize=12)
        ax.set_title(f'Cumulative Distribution Function: {method.upper()}', 
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / f'{method}_cdf.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved CDF plot: {output_file}")
        plt.close()
    
    def generate_latex_table(self, all_results):
        """
        Generate LaTeX table summarizing all methods.
        
        Parameters:
        -----------
        all_results : dict
            Results from all methods
        """
        print("\n" + "="*60)
        print("Generating LaTeX table...")
        print("="*60)
        
        if not all_results:
            print("  Warning: No results to generate LaTeX table")
            return
        
        latex_lines = []
        latex_lines.append("\\begin{table}[htbp]")
        latex_lines.append("\\centering")
        latex_lines.append("\\caption{Probability Distribution Statistics by Feature Selection Method and Sampling Strategy}")
        latex_lines.append("\\label{tab:sample-probability-distributions}")
        latex_lines.append("\\begin{tabular}{lcccccc}")
        latex_lines.append("\\toprule")
        latex_lines.append("\\textbf{Method} & \\textbf{Sampling} & \\textbf{Mean} & \\textbf{Std} & \\textbf{Median} & \\textbf{Near Boundary} & \\textbf{AUC} \\\\")
        latex_lines.append("\\midrule")
        
        for method in self.methods:
            if method not in all_results:
                continue
            
            results = all_results[method]
            
            # Hard negative negatives
            hard_stats = self.calculate_statistics(results['hard_neg_probs'], 'hard_neg')
            hard_auc = results['hard_results']['mean_auc']
            hard_auc_std = results['hard_results']['std_auc']
            
            latex_lines.append(
                f"{method.replace('_', ' ').title()} & Hard Neg & "
                f"{hard_stats['mean']:.3f} & {hard_stats['std']:.3f} & "
                f"{hard_stats['median']:.3f} & {hard_stats['pct_near_boundary']:.1f}\\% & "
                f"{hard_auc:.3f}$\\pm${hard_auc_std:.3f} \\\\"
            )
            
            # Random negatives
            random_stats = self.calculate_statistics(results['random_neg_probs'], 'random')
            random_auc = results['random_results']['mean_auc']
            random_auc_std = results['random_results']['std_auc']
            
            latex_lines.append(
                f" & Random & "
                f"{random_stats['mean']:.3f} & {random_stats['std']:.3f} & "
                f"{random_stats['median']:.3f} & {random_stats['pct_near_boundary']:.1f}\\% & "
                f"{random_auc:.3f}$\\pm${random_auc_std:.3f} \\\\"
            )
            
            # Overlap test
            overlap = self.test_distribution_overlap(
                results['hard_neg_probs'], results['random_neg_probs'],
                'hard', 'random'
            )
            
            latex_lines.append(
                f"\\multicolumn{{7}}{{l}}{{\\textit{{Cohen's d: {overlap['cohens_d']:.3f} "
                f"({overlap['effect_size']}), KS p-value: {overlap['ks_pvalue']:.4f}}}}} \\\\"
            )
            
            if method != self.methods[-1]:
                latex_lines.append("\\midrule")
        
        latex_lines.append("\\bottomrule")
        latex_lines.append("\\end{tabular}")
        latex_lines.append("\\end{table}")
        
        # Save LaTeX table
        latex_file = self.output_dir / 'probability_distributions_table.tex'
        with open(latex_file, 'w') as f:
            f.write('\n'.join(latex_lines))
        
        print(f"Saved LaTeX table: {latex_file}")
        
        # Also print to console
        print("\n" + "="*60)
        print("LaTeX Table:")
        print("="*60)
        print('\n'.join(latex_lines))
    
    def run_full_analysis(self):
        """
        Run complete analysis for all feature selection methods.
        """
        print("\n" + "="*80)
        print("SPATIAL CV SAMPLE DISTRIBUTION ANALYSIS")
        print("="*80)
        
        all_results = {}
        
        # Analyze each method
        for method in self.methods:
            try:
                results = self.analyze_method(method)
                all_results[method] = results
                
                # Create visualizations
                self.create_visualizations(results, method)
                
                # Print overlap test
                overlap = self.test_distribution_overlap(
                    results['hard_neg_probs'], results['random_neg_probs'],
                    f'{method}_hard', f'{method}_random'
                )
                print(f"\n  Overlap test (negative samples only):")
                print(f"    Cohen's d: {overlap['cohens_d']:.4f} ({overlap['effect_size']})")
                print(f"    KS test: stat={overlap['ks_statistic']:.4f}, p={overlap['ks_pvalue']:.4f}")
                print(f"    Mann-Whitney U: stat={overlap['mw_statistic']:.1f}, p={overlap['mw_pvalue']:.4f}")
                
            except Exception as e:
                print(f"\n  ERROR analyzing {method}: {e}")
                import traceback
                traceback.print_exc()
        
        # Generate summary LaTeX table
        if all_results:
            self.generate_latex_table(all_results)
        
        # Save summary statistics CSV
        self._save_summary_csv(all_results)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"Results saved to: {self.output_dir}/")
        print(f"  - {len(all_results)} method comparisons")
        print(f"  - {len(all_results) * 2} visualizations (4-panel + CDF per method)")
        print(f"  - LaTeX table: probability_distributions_table.tex")
        print(f"  - Summary CSV: summary_statistics.csv")
    
    def _save_summary_csv(self, all_results):
        """Save summary statistics to CSV."""
        if not all_results:
            print("\n  Warning: No results to save to CSV")
            return
        
        rows = []
        
        for method, results in all_results.items():
            # Hard negative stats
            hard_stats = self.calculate_statistics(results['hard_neg_probs'], 'hard_neg')
            hard_stats['method'] = method
            hard_stats['sampling'] = 'hard_neg'
            hard_stats['auc'] = results['hard_results']['mean_auc']
            hard_stats['auc_std'] = results['hard_results']['std_auc']
            rows.append(hard_stats)
            
            # Random stats
            random_stats = self.calculate_statistics(results['random_neg_probs'], 'random')
            random_stats['method'] = method
            random_stats['sampling'] = 'random'
            random_stats['auc'] = results['random_results']['mean_auc']
            random_stats['auc_std'] = results['random_results']['std_auc']
            rows.append(random_stats)
            
            # Overlap test
            overlap = self.test_distribution_overlap(
                results['hard_neg_probs'], results['random_neg_probs'],
                'hard', 'random'
            )
            overlap_row = {
                'method': method,
                'sampling': 'overlap_test',
                'cohens_d': overlap['cohens_d'],
                'effect_size': overlap['effect_size'],
                'ks_statistic': overlap['ks_statistic'],
                'ks_pvalue': overlap['ks_pvalue'],
                'mw_pvalue': overlap['mw_pvalue']
            }
            rows.append(overlap_row)
        
        df = pd.DataFrame(rows)
        csv_file = self.output_dir / 'summary_statistics.csv'
        df.to_csv(csv_file, index=False)
        print(f"\nSaved summary CSV: {csv_file}")


def main():
    """Main execution function."""
    # Initialize analyzer
    analyzer = SpatialCVSampleDistributionAnalyzer(
        results_dir='corrected_hard_negative_results',
        output_dir='sample_distribution_results',
        n_folds=3,  # Match thesis spatial CV
        random_state=42
    )
    
    # Run full analysis
    analyzer.run_full_analysis()


if __name__ == '__main__':
    main()
