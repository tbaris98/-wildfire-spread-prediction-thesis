"""
Corrected Spatial CV Framework - Class Imbalance Addressed
====================================================

This script addresses the following critical class imbalance issues found in spatial CV results:
- Implements balanced sampling strategies
- Adds class distribution validation
- Provides algorithm-specific parameter tuning
- Ensures robust spatial cross-validation

Author: Tuna Baris Unal
File: DSS Thesis - Wildfire Prediction Framework
Date: October 2024
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless plotting
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import os
from datetime import datetime

# Machine Learning
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BalancedSpatialCV:
    """Spatial Cross-Validation with class imbalance handling"""
    
    def __init__(self, output_dir='corrected_spatial_cv_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        self.error_log = []
        self.confusion_matrices = []  # Store confusion matrices for error analysis

    
    def load_datasets(self):
        """Load all experimental datasets"""
        datasets = {}
        
        # Load hard negative sampling datasets (balanced versions)
        # Load processed hard negative and random negative datasets from corrected_hard_negative_results
        processed_dir = Path('corrected_hard_negative_results')
        for method in ['tree_based', 'ensemble', 'correlation', 'variance', 'baseline']:
            for sampling in ['hard_negative', 'random_negative']:
                dataset_file = processed_dir / f'{method}_{sampling}_dataset.csv'
                if dataset_file.exists():
                    try:
                        data = pd.read_csv(dataset_file)
                        X = data.drop('target', axis=1).values
                        y = data['target'].values
                        datasets[f'{method}_{sampling}'] = (X, y)
                        logger.info(f"Loaded {method} {sampling} dataset: {X.shape}")
                    except Exception as e:
                        logger.warning(f"Could not load {method} {sampling} dataset: {e}")
        
        return datasets
        
    def validate_class_distribution(self, y, dataset_name):
        """Validate class distribution and detect issues"""
        unique_classes, counts = np.unique(y, return_counts=True)
        
        logger.info(f"\n{dataset_name} - Class Distribution:")
        for cls, count in zip(unique_classes, counts):
            percentage = (count / len(y)) * 100
            logger.info(f"  Class {cls}: {count:,} ({percentage:.2f}%)")
        
        # Check for critical issues
        if len(unique_classes) < 2:
            logger.error(f"CRITICAL: Only {len(unique_classes)} class in {dataset_name}")
            return False, "single_class"
        
        minority_ratio = min(counts) / max(counts)
        if minority_ratio < 0.001:  # Less than 0.1%
            logger.warning(f"Extreme imbalance in {dataset_name}: {minority_ratio:.6f}")
            return False, "extreme_imbalance"
        
        return True, "balanced"
    
    def get_balanced_indices(self, y, target_ratio=0.3):
        """Get indices for balanced random sampling"""
        unique_classes, counts = np.unique(y, return_counts=True)
        majority_class = unique_classes[np.argmax(counts)]
        minority_class = unique_classes[np.argmin(counts)]
        
        majority_mask = (y == majority_class)
        minority_mask = (y == minority_class)
        
        n_minority = np.sum(minority_mask)
        n_majority_target = int(n_minority / target_ratio)
        
        # Don't undersample too aggressively
        n_majority_available = np.sum(majority_mask)
        n_majority_target = min(n_majority_target, n_majority_available)
        
        # Sample indices
        majority_indices = np.where(majority_mask)[0]
        selected_majority = np.random.choice(
            majority_indices, n_majority_target, replace=False
        )
        minority_indices = np.where(minority_mask)[0]
        
        # Combine samples
        selected_indices = np.concatenate([minority_indices, selected_majority])
        np.random.shuffle(selected_indices)
        
        return selected_indices

    def create_balanced_sample(self, X, y, strategy='balanced_random', target_ratio=0.3):
        """Create balanced sample from imbalanced dataset"""
        
        is_valid, issue = self.validate_class_distribution(y, "Original Dataset")
        if not is_valid and issue == "single_class":
            raise ValueError("Cannot balance single-class dataset")
        
        if strategy == 'balanced_random':
            # Random undersampling of majority class
            selected_indices = self.get_balanced_indices(y, target_ratio)
            return X[selected_indices], y[selected_indices]
        
        elif strategy == 'smote':
            # Use SMOTE for oversampling minority class
            try:
                smote = SMOTE(random_state=42)
                X_balanced, y_balanced = smote.fit_resample(X, y)
                return X_balanced, y_balanced
            except Exception as e:
                logger.warning(f"SMOTE failed: {e}, falling back to random sampling")
                return self.create_balanced_sample(X, y, 'balanced_random', target_ratio)
        
        else:
            raise ValueError(f"Unknown balancing strategy: {strategy}")
    
    def get_algorithm_config(self, algorithm_name, is_imbalanced=False):
        """Get algorithm-specific configuration for imbalanced datasets"""
        
        configs = {
            'RandomForest': {
                'balanced': RandomForestClassifier(
                    n_estimators=100,
                    class_weight='balanced',
                    max_depth=10,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42
                ),
                'regular': RandomForestClassifier(
                    n_estimators=100,
                    random_state=42
                )
            },
            'GradientBoosting': {
                'balanced': GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    subsample=0.8,
                    random_state=42
                ),
                'regular': GradientBoostingClassifier(
                    n_estimators=100,
                    random_state=42
                )
            },
            'LogisticRegression': {
                'balanced': LogisticRegression(
                    class_weight='balanced',
                    max_iter=1000,
                    C=1.0,
                    solver='liblinear',
                    random_state=42
                ),
                'regular': LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            },
            'MLP': {
                'balanced': MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    max_iter=500,
                    learning_rate='adaptive',
                    early_stopping=True,
                    validation_fraction=0.1,
                    random_state=42
                ),
                'regular': MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    max_iter=300,
                    random_state=42
                )
            }
        }
        
        config_type = 'balanced' if is_imbalanced else 'regular'
        return configs[algorithm_name][config_type]
    
    def create_spatial_coordinates(self, X):
        """Create synthetic spatial coordinates"""
        n_samples = len(X)
        # Create grid-like coordinates
        grid_size = int(np.sqrt(n_samples)) + 1
        coordinates = np.array([
            [i % grid_size, i // grid_size] for i in range(n_samples)
        ])
        return coordinates
    
    def create_spatial_groups(self, coordinates, n_groups=10):
        """Create spatial groups using K-means clustering"""
        kmeans = KMeans(n_clusters=n_groups, random_state=42)
        return kmeans.fit_predict(coordinates)
    
    def evaluate_configuration(self, config_name, X, y, algorithm_name):
        """Evaluate a single configuration with spatial CV"""
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {config_name} + {algorithm_name}")
        logger.info(f"{'='*60}")
        
        try:
            # Validate class distribution
            is_valid, issue = self.validate_class_distribution(y, config_name)
            
            # Handle class imbalance
            if not is_valid:
                if issue == "single_class":
                    error_msg = f"Single class dataset: {config_name}"
                    logger.error(error_msg)
                    self.error_log.append({
                        'config': config_name,
                        'algorithm': algorithm_name,
                        'error': error_msg,
                        'type': 'single_class'
                    })
                    return None
                
                elif issue == "extreme_imbalance":
                    logger.info("Applying balanced sampling to extreme imbalance...")
                    try:
                        X, y = self.create_balanced_sample(X, y, 'balanced_random', 0.2)
                        is_valid, _ = self.validate_class_distribution(y, f"{config_name} (Balanced)")
                    except Exception as e:
                        error_msg = f"Balancing failed: {e}"
                        logger.error(error_msg)
                        self.error_log.append({
                            'config': config_name,
                            'algorithm': algorithm_name,
                            'error': error_msg,
                            'type': 'balancing_failed'
                        })
                        return None
            
            if not is_valid:
                return None
            
            # Create spatial coordinates and groups
            coordinates = self.create_spatial_coordinates(X)
            spatial_groups = self.create_spatial_groups(coordinates, n_groups=5)
            
            # Get algorithm with appropriate configuration
            is_imbalanced = issue == "extreme_imbalance"
            model = self.get_algorithm_config(algorithm_name, is_imbalanced)
            
            # Spatial Cross-Validation
            spatial_cv = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=42)
            
            auc_scores = []
            f1_scores = []
            
            fold_num = 0
            for train_idx, test_idx in spatial_cv.split(X, y, spatial_groups):
                fold_num += 1
                logger.info(f"  Processing fold {fold_num}/3...")
                
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Validate fold class distribution
                train_valid, _ = self.validate_class_distribution(y_train, f"Train Fold {fold_num}")
                test_valid, _ = self.validate_class_distribution(y_test, f"Test Fold {fold_num}")
                
                if not train_valid or not test_valid:
                    logger.warning(f"Skipping fold {fold_num} due to class imbalance")
                    continue
                
                # Scale features for algorithms that need it
                if algorithm_name in ['LogisticRegression', 'MLP']:
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)
                
                # Train and evaluate
                model.fit(X_train, y_train)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                auc = roc_auc_score(y_test, y_pred_proba)
                f1 = f1_score(y_test, y_pred)
                
                # Store confusion matrix for error analysis
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                self.confusion_matrices.append({
                    'configuration': config_name,
                    'algorithm': algorithm_name,
                    'fold': fold_num,
                    'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
                    'auc': float(auc), 'f1': float(f1)
                })
                
                auc_scores.append(auc)
                f1_scores.append(f1)
                
                logger.info(f"    Fold {fold_num}: AUC={auc:.4f}, F1={f1:.4f}")
            
            if len(auc_scores) == 0:
                error_msg = "No valid folds completed"
                logger.error(error_msg)
                self.error_log.append({
                    'config': config_name,
                    'algorithm': algorithm_name,
                    'error': error_msg,
                    'type': 'no_valid_folds'
                })
                return None
            
            # Calculate final metrics
            mean_auc = np.mean(auc_scores)
            std_auc = np.std(auc_scores)
            mean_f1 = np.mean(f1_scores)
            std_f1 = np.std(f1_scores)
            
            result = {
                'configuration': config_name,
                'algorithm': algorithm_name,
                'spatial_auc_mean': mean_auc,
                'spatial_auc_std': std_auc,
                'spatial_f1_mean': mean_f1,
                'spatial_f1_std': std_f1,
                'n_features': X.shape[1],
                'n_samples': len(y),
                'n_folds_completed': len(auc_scores),
                'was_balanced': is_imbalanced
            }
            
            logger.info(f"✅ COMPLETED: {config_name} + {algorithm_name}")
            logger.info(f"   Final AUC: {mean_auc:.4f} ± {std_auc:.4f}")
            logger.info(f"   Final F1:  {mean_f1:.4f} ± {std_f1:.4f}")
            
            return result
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(f"❌ FAILED: {config_name} + {algorithm_name} - {error_msg}")
            self.error_log.append({
                'config': config_name,
                'algorithm': algorithm_name,
                'error': error_msg,
                'type': 'unexpected_error'
            })
            return None
    
    def load_results_from_csv(self, csv_path='corrected_hard_negative_results/corrected_hard_negative_spatial_cv_results.csv'):
        """Load results directly from the corrected CSV file"""
        if not os.path.exists(csv_path):
            logger.error(f"Results CSV not found: {csv_path}")
            return None
        try:
            results_df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(results_df)} results from {csv_path}")
            return results_df
        except Exception as e:
            logger.error(f"Failed to load results CSV: {e}")
            return None
    
    def run_comprehensive_evaluation(self):
        """Run comprehensive spatial CV evaluation (restored training logic)"""
        logger.info("Starting Corrected Spatial CV Evaluation...")
        logger.info(f"Output directory: {self.output_dir.absolute()}")

        # Load datasets
        datasets = self.load_datasets()
        if not datasets:
            logger.error("No datasets loaded!")
            return

        logger.info(f"Loaded {len(datasets)} datasets")

        # Algorithms to test
        algorithms = ['RandomForest', 'GradientBoosting', 'LogisticRegression', 'MLP']

        # Run evaluations
        total_configs = len(datasets) * len(algorithms)
        current_config = 0

        for dataset_name, (X, y) in datasets.items():
            # Skip baseline_all_features to focus on selected feature sets
            if dataset_name == 'baseline_all_features':
                logger.info(f"Skipping baseline_all_features dataset for evaluation.")
                continue
            for algorithm in algorithms:
                current_config += 1
                logger.info(f"\nProgress: {current_config}/{total_configs}")
                result = self.evaluate_configuration(dataset_name, X, y, algorithm)
                if result:
                    self.results.append(result)

        # Save results
        self.save_results()
        self.create_analysis_report()
        
        # Perform Subgroup Analysis on the Best Model
        if self.results:
            best_result = max(self.results, key=lambda x: x['spatial_auc_mean'])
            best_config = best_result['configuration']
            best_algo = best_result['algorithm']
            
            logger.info(f"\nBest Model Found: {best_config} + {best_algo}")
            self.perform_subgroup_analysis(best_config, best_algo)
            
            # Also perform detailed confusion matrix analysis on the best model
            self.analyze_confusion_matrices(best_config, best_algo)

    def analyze_confusion_matrices(self, config_name, algorithm_name):
        """Detailed analysis of confusion matrices for the best model"""
        logger.info(f"Analyzing confusion matrices for {config_name} + {algorithm_name}...")
        
        # Filter confusion matrices for the best model
        relevant_cms = [cm for cm in self.confusion_matrices 
                       if cm['configuration'] == config_name and cm['algorithm'] == algorithm_name]
        
        if not relevant_cms:
            logger.warning("No confusion matrices found for best model analysis")
            return
            
        # Aggregate CM
        total_tn = sum(cm['tn'] for cm in relevant_cms)
        total_fp = sum(cm['fp'] for cm in relevant_cms)
        total_fn = sum(cm['fn'] for cm in relevant_cms)
        total_tp = sum(cm['tp'] for cm in relevant_cms)
        
        # Calculate aggregate metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0
        
        analysis_lines = []
        analysis_lines.append(f"\nDETAILED CONFUSION MATRIX ANALYSIS: {config_name} + {algorithm_name}")
        analysis_lines.append("-" * 60)
        analysis_lines.append(f"Total Samples: {total_tn + total_fp + total_fn + total_tp}")
        analysis_lines.append(f"True Negatives: {total_tn}")
        analysis_lines.append(f"False Positives: {total_fp}")
        analysis_lines.append(f"False Negatives: {total_fn}")
        analysis_lines.append(f"True Positives: {total_tp}")
        analysis_lines.append("-" * 30)
        analysis_lines.append(f"Precision: {precision:.4f}")
        analysis_lines.append(f"Recall (Sensitivity): {recall:.4f}")
        analysis_lines.append(f"Specificity: {specificity:.4f}")
        analysis_lines.append(f"F1 Score: {f1:.4f}")
        
        # Save to file
        analysis_file = self.output_dir / 'best_model_confusion_analysis.txt'
        with open(analysis_file, 'w') as f:
            f.write('\n'.join(analysis_lines))
            
        logger.info(f"Confusion matrix analysis saved to {analysis_file}")
    
    def save_results(self):
        """Save results to files"""
        
        # Save successful results
        if self.results:
            results_df = pd.DataFrame(self.results)
            results_file = self.output_dir / 'corrected_spatial_cv_results.csv'
            results_df.to_csv(results_file, index=False)
            logger.info(f"Saved {len(self.results)} successful results to {results_file}")
            
        # Save confusion matrices for error analysis
        if self.confusion_matrices:
            cm_df = pd.DataFrame(self.confusion_matrices)
            cm_file = self.output_dir / 'corrected_spatial_cv_confusion_matrices.csv'
            cm_df.to_csv(cm_file, index=False)
            logger.info(f"Saved {len(self.confusion_matrices)} confusion matrices to {cm_file}")
        
        # Save error log
        if self.error_log:
            errors_df = pd.DataFrame(self.error_log)
            errors_file = self.output_dir / 'corrected_spatial_cv_errors.csv'
            errors_df.to_csv(errors_file, index=False)
            logger.info(f"Saved {len(self.error_log)} errors to {errors_file}")
    
    def create_analysis_report(self):
        """Create comprehensive analysis report"""
        
        report_lines = []
        report_lines.append("CORRECTED SPATIAL CROSS-VALIDATION RESULTS")
        report_lines.append("=" * 60)
        report_lines.append(f"Execution Time: {datetime.now()}")
        report_lines.append(f"Total Successful Configurations: {len(self.results)}")
        report_lines.append(f"Total Failed Configurations: {len(self.error_log)}")
        
        if self.results:
            results_df = pd.DataFrame(self.results)
            
            # Best performing configurations
            report_lines.append("\nTOP 10 PERFORMING CONFIGURATIONS:")
            report_lines.append("-" * 40)
            top_10 = results_df.nlargest(10, 'spatial_auc_mean')
            for _, row in top_10.iterrows():
                report_lines.append(
                    f"{row['configuration']} + {row['algorithm']}: "
                    f"AUC={row['spatial_auc_mean']:.4f}±{row['spatial_auc_std']:.4f}, "
                    f"F1={row['spatial_f1_mean']:.4f}±{row['spatial_f1_std']:.4f}"
                )
            
            # Algorithm comparison
            report_lines.append("\nALGORITHM PERFORMANCE COMPARISON:")
            report_lines.append("-" * 40)
            for algorithm in results_df['algorithm'].unique():
                alg_results = results_df[results_df['algorithm'] == algorithm]
                mean_auc = alg_results['spatial_auc_mean'].mean()
                std_auc = alg_results['spatial_auc_mean'].std()
                report_lines.append(f"{algorithm}: Mean AUC = {mean_auc:.4f} ± {std_auc:.4f}")
        
        if self.error_log:
            # Error analysis
            errors_df = pd.DataFrame(self.error_log)
            report_lines.append("\nERROR ANALYSIS:")
            report_lines.append("-" * 40)
            error_counts = errors_df['type'].value_counts()
            for error_type, count in error_counts.items():
                report_lines.append(f"{error_type}: {count} occurrences")
        
        # Save report
        report_file = self.output_dir / 'corrected_spatial_cv_analysis.txt'
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Analysis report saved to {report_file}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("CORRECTED SPATIAL CV EVALUATION COMPLETED")
        print("=" * 60)
        print(f"✅ Successful configurations: {len(self.results)}")
        print(f"❌ Failed configurations: {len(self.error_log)}")
        if self.results:
            best_result = max(self.results, key=lambda x: x['spatial_auc_mean'])
            print(f"🏆 Best configuration: {best_result['configuration']} + {best_result['algorithm']}")
            print(f"   Best AUC: {best_result['spatial_auc_mean']:.4f} ± {best_result['spatial_auc_std']:.4f}")
        
    def perform_subgroup_analysis(self, dataset_name, algorithm_name):
        """Perform detailed subgroup analysis on the best model"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Performing Subgroup Analysis (RQ3) for: {dataset_name} + {algorithm_name}")
        logger.info(f"{'='*60}")
        
        # 1. Reload dataset with metadata
        # We need to find the file path again. 
        # This logic duplicates load_datasets slightly but is safer than refactoring the whole class.
        processed_dir = Path('corrected_hard_negative_results')
        dataset_file = processed_dir / f'{dataset_name}.csv'
        
        # Handle the case where dataset_name might not match filename exactly if logic changes
        # But based on load_datasets, keys are 'method_sampling' which matches 'method_sampling_dataset.csv'
        if not dataset_file.exists():
             # Try appending _dataset.csv if not present (load_datasets keys are like 'tree_based_hard_negative')
             dataset_file = processed_dir / f'{dataset_name}_dataset.csv'
        
        if not dataset_file.exists():
            logger.error(f"Could not find dataset file for {dataset_name}")
            return

        df = pd.read_csv(dataset_file)
        
        # 2. Prepare Data (Metadata + Features)
        target_col = 'target'
        y_raw = df[target_col].values
        
        # Handle Class Imbalance (Same logic as evaluate_configuration)
        is_valid, issue = self.validate_class_distribution(y_raw, f"{dataset_name} (Analysis)")
        was_balanced = False
        
        if not is_valid and issue == "extreme_imbalance":
            logger.info("Applying balanced sampling for analysis...")
            # Use shared balancing logic
            selected_indices = self.get_balanced_indices(y_raw, target_ratio=0.2)
            
            df = df.iloc[selected_indices].reset_index(drop=True)
            was_balanced = True
            logger.info(f"Balanced dataset shape: {df.shape}")

        # Prepare X and y
        X = df.drop(columns=[target_col]).values
        y = df[target_col].values
        
        # Metadata columns to track
        metadata_cols = ['TEMP_ave', 'Neighbour_EVC_mean', 'c_latitude']
        available_metadata = [c for c in metadata_cols if c in df.columns]
        
        # Create spatial groups
        coordinates = self.create_spatial_coordinates(X)
        groups = self.create_spatial_groups(coordinates, n_groups=5)
        
        # Get model
        model = self.get_algorithm_config(algorithm_name, is_imbalanced=was_balanced)
        
        # Run CV and collect predictions
        cv = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=42)
        results = []
        
        fold = 1
        for train_idx, test_idx in cv.split(X, y, groups):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            if algorithm_name in ['LogisticRegression', 'MLP']:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Store predictions
            fold_results = pd.DataFrame({
                'Algorithm': algorithm_name,
                'Fold': fold,
                'True_Label': y_test,
                'Predicted_Label': y_pred
            })
            
            # Add metadata
            for col in available_metadata:
                fold_results[col] = df.iloc[test_idx][col].values
            
            results.append(fold_results)
            fold += 1
            
        full_results = pd.concat(results, ignore_index=True)
        
        # Save raw predictions
        full_results.to_csv(self.output_dir / 'best_model_predictions_with_metadata.csv', index=False)
        
        # Analyze Subgroups
        self.analyze_subgroups(full_results)

    def analyze_subgroups(self, full_results):
        """Calculate error metrics for subgroups and plot"""
        logger.info("Calculating subgroup error metrics...")
        
        # Binning
        if 'TEMP_ave' in full_results.columns:
            full_results['Temp_Bin'] = pd.cut(full_results['TEMP_ave'], 
                                            bins=[-np.inf, 10, 20, 30, np.inf],
                                            labels=['<10°C', '10-20°C', '20-30°C', '>30°C'])
        
        if 'Neighbour_EVC_mean' in full_results.columns:
            full_results['Veg_Bin'] = pd.qcut(full_results['Neighbour_EVC_mean'], q=4, 
                                            labels=['Low', 'Medium', 'High', 'Very High'], duplicates='drop')
            
        if 'c_latitude' in full_results.columns:
            full_results['Lat_Bin'] = pd.qcut(full_results['c_latitude'], q=4, 
                                            labels=['South', 'South-Mid', 'North-Mid', 'North'])
            
        metrics = []
        algo = full_results['Algorithm'].iloc[0]
        
        factors = []
        if 'Temp_Bin' in full_results.columns: factors.append(('Temperature', 'Temp_Bin'))
        if 'Veg_Bin' in full_results.columns: factors.append(('Vegetation', 'Veg_Bin'))
        if 'Lat_Bin' in full_results.columns: factors.append(('Region', 'Lat_Bin'))
        
        for factor, bin_col in factors:
            for bin_val in full_results[bin_col].unique():
                subset = full_results[full_results[bin_col] == bin_val]
                if len(subset) == 0: continue
                
                tn, fp, fn, tp = confusion_matrix(subset['True_Label'], subset['Predicted_Label'], labels=[0,1]).ravel()
                
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
                accuracy = (tp + tn) / len(subset)
                
                metrics.append({
                    'Algorithm': algo,
                    'Factor': factor,
                    'Bin': bin_val,
                    'FPR': fpr,
                    'FNR': fnr,
                    'Accuracy': accuracy,
                    'Count': len(subset)
                })
        
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(self.output_dir / 'subgroup_error_metrics.csv', index=False)
        
        # Plotting
        self.plot_subgroup_results(metrics_df)

    def plot_subgroup_results(self, metrics_df):
        """Generate plots for subgroup analysis"""
        logger.info("Generating subgroup plots...")
        sns.set_style("whitegrid")
        
        subgroup_dir = self.output_dir / 'subgroup_plots'
        subgroup_dir.mkdir(exist_ok=True)
        
        factors = metrics_df['Factor'].unique()
        
        for factor in factors:
            factor_data = metrics_df[metrics_df['Factor'] == factor].sort_values('Bin')
            
            # Plot FPR
            plt.figure(figsize=(10, 6))
            sns.barplot(data=factor_data, x='Bin', y='FPR')
            plt.title(f'False Positive Rate by {factor}')
            plt.ylabel('False Positive Rate')
            plt.xlabel(factor)
            plt.savefig(subgroup_dir / f'fpr_by_{factor.lower()}.png')
            plt.close()
            
            # Plot FNR
            plt.figure(figsize=(10, 6))
            sns.barplot(data=factor_data, x='Bin', y='FNR')
            plt.title(f'False Negative Rate by {factor}')
            plt.ylabel('False Negative Rate (Missed Fires)')
            plt.xlabel(factor)
            plt.savefig(subgroup_dir / f'fnr_by_{factor.lower()}.png')
            plt.close()

if __name__ == "__main__":
    evaluator = BalancedSpatialCV()
    evaluator.run_comprehensive_evaluation()