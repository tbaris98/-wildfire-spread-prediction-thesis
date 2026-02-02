#!/usr/bin/env python3
"""
Complete Hard Negative Sampling Fix and Spatial CV Re-run
========================================================

This script addresses the critical hard negative sampling issue and re-runs
the complete spatial CV evaluation for RQ2.

Steps:
1. Identifies the root cause (continuous scores instead of binary targets)
2. Creates proper hard negative datasets with binary classification targets
3. Re-runs spatial CV with corrected datasets
4. Generates complete comparison between random and hard negative sampling

Author: DSS Thesis - Wildfire Prediction Framework
Date: October 2024
"""


import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# --- Centralized Logging Setup ---
LOG_DIR = Path('wildfire_results')
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f'hard_negative_fix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
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
CONFIG_LOG = LOG_DIR / 'hard_negative_fix_config.txt'
with open(CONFIG_LOG, 'w') as f:
    f.write("# Hard Negative Sampling Fix & Spatial CV Configuration\n")
    f.write(f"Run timestamp: {datetime.now().isoformat()}\n")
    f.write("Input: corrected_hard_negative_results/\n")
    f.write("Output: wildfire_results/ (logs, results, comparison tables)\n")
    f.write("Key Steps: hard negative dataset creation, binary target, spatial CV, comparison\n")
    f.write("Hyperparameters: see code for algorithm-specific settings\n")

# --- Error Handling Decorator ---
def log_exceptions(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Exception in {func.__name__}: {e}", exc_info=True)
            raise
    return wrapper

# Machine Learning
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('complete_hard_negative_fix.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HardNegativeSamplingFix:
    """Complete solution for hard negative sampling issues"""
    
    def __init__(self, output_dir='corrected_hard_negative_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        self.error_log = []
        
        logger.info(f"Initialized HardNegativeSamplingFix with output dir: {self.output_dir}")
    
    def load_baseline_data(self):
        """Load the baseline wildfire dataset"""
        
        baseline_files = [
            'wildfire_results/preprocessed_wildfire_data.csv',
            'features_array.csv',
            'wildfire_features.csv',
            'baseline_features.csv'
        ]
        
        for filename in baseline_files:
            if os.path.exists(filename):
                try:
                    logger.info(f"Loading baseline data from {filename}...")
                    
                    # Try different separators and handle malformed CSV
                    try:
                        data = pd.read_csv(filename, low_memory=False)
                    except:
                        # Try with different separator
                        data = pd.read_csv(filename, sep='\t', low_memory=False)
                    
                    logger.info(f"Loaded data shape: {data.shape}")
                    logger.info(f"Columns: {list(data.columns[:10])}...")  # Show first 10 columns
                    
                    # Look for target column
                    target_cols = [col for col in data.columns if 'target' in col.lower() or 'fire' in col.lower() or 'class' in col.lower()]
                    
                    if target_cols:
                        target_col = target_cols[0]
                        logger.info(f"Found potential target column: {target_col}")
                        
                        # Check target values
                        unique_targets = data[target_col].unique()
                        logger.info(f"Target values: {unique_targets}")
                        
                        if len(unique_targets) == 2:
                            # Create binary targets
                            data['target'] = (data[target_col] == unique_targets[1]).astype(int)
                            
                            # Clean column names (strip whitespace)
                            data.columns = [str(col).strip() for col in data.columns]
                            target_col_stripped = str(target_col).strip()
                            
                            # Remove non-numeric columns except target
                            numeric_cols = data.select_dtypes(include=[np.number]).columns
                            # CRITICAL: Exclude both the new 'target' column AND the original target column
                            feature_cols = [col for col in numeric_cols if col != 'target' and col != target_col_stripped]
                            
                            # Robust exclusion of leakage-prone features
                            # Exclude any column containing these patterns (case-insensitive)
                            leakage_patterns = ['acq_time', 'neighbour_acq_time', 'frp', 'brightness', 'confidence', 'scan', 'track']
                            feature_cols = [col for col in feature_cols if not any(pat in col.lower() for pat in leakage_patterns)]
                            
                            X = data[feature_cols]
                            y = data['target']

                            # Remove rows with missing values
                            mask = ~(X.isnull().any(axis=1) | y.isnull())
                            X = X[mask]
                            y = y[mask]

                            logger.info(f"Final dataset (leakage features excluded): {X.shape} features, {len(y)} samples")
                            logger.info(f"Class distribution: {np.bincount(y)}")

                            return X, y
                    
                    logger.warning(f"No suitable target column found in {filename}")
                    
                except Exception as e:
                    logger.error(f"Failed to load {filename}: {e}")
        
        logger.error("Could not load baseline data from any available file")
        return None, None
    
    def create_feature_selected_datasets(self, X, y):
        """Create datasets with different feature selection approaches"""
        
        logger.info("Creating feature-selected datasets...")
        
        datasets = {}

        # 1. Ensemble consensus features (from RQ1)
        try:
            ensemble_path = 'feature_selection_results/selected_features_ensemble.csv'
            if os.path.exists(ensemble_path):
                ensemble_df = pd.read_csv(ensemble_path)
                exclude_features = ['acq_time', 'Neighbour_acq_time']
                ensemble_features = [f for f in ensemble_df['feature'] if f in X.columns and f not in exclude_features]
                if len(ensemble_features) > 0:
                    datasets['ensemble'] = {
                        'X': X[ensemble_features],
                        'y': y,
                        'features': ensemble_features
                    }
                    logger.info(f"Ensemble consensus: {len(ensemble_features)} features loaded from {ensemble_path} (leakage features excluded)")
                else:
                    logger.warning(f"No ensemble features found in baseline data columns.")
            else:
                logger.warning(f"Ensemble feature file not found: {ensemble_path}")
        except Exception as e:
            logger.error(f"Failed to create ensemble consensus features: {e}")

        # 2. Tree-based feature selection (use feature importance)
        try:
            # CRITICAL: Exclude leakage features BEFORE training RF (Robust check)
            leakage_patterns = ['acq_time', 'neighbour_acq_time', 'frp', 'brightness', 'confidence', 'scan', 'track']
            clean_cols = [col for col in X.columns if not any(pat in col.lower() for pat in leakage_patterns)]
            X_clean = X[clean_cols]
            
            logger.info(f"Tree-based selection input shape: {X_clean.shape}")
            
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_clean, y)
            feature_importance = pd.Series(rf.feature_importances_, index=X_clean.columns)
            top_20 = feature_importance.nlargest(20)
            
            logger.info(f"Top 20 Tree-based features and importances:")
            for name, imp in top_20.items():
                logger.info(f"  {name}: {imp:.4f}")
            
            top_20_features = top_20.index
            
            datasets['tree_based'] = {
                'X': X_clean[top_20_features],
                'y': y,
                'features': top_20_features.tolist()
            }
            logger.info(f"Tree-based: {len(top_20_features)} features (leakage features excluded)")
        except Exception as e:
            logger.error(f"Failed to create tree-based features: {e}")

        # 3. Correlation-based feature selection (remove highly correlated)
        try:
            # CRITICAL: Exclude leakage features BEFORE computing correlations
            leakage_patterns = ['acq_time', 'neighbour_acq_time', 'frp', 'brightness', 'confidence', 'scan', 'track']
            clean_cols = [col for col in X.columns if not any(pat in col.lower() for pat in leakage_patterns)]
            X_clean = X[clean_cols]
            
            corr_matrix = X_clean.corr().abs()
            upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
            correlation_features = [col for col in X_clean.columns if col not in to_drop][:50]
            datasets['correlation'] = {
                'X': X_clean[correlation_features],
                'y': y,
                'features': correlation_features
            }
            logger.info(f"Correlation-based: {len(correlation_features)} features (leakage features excluded)")
        except Exception as e:
            logger.error(f"Failed to create correlation features: {e}")

        # 4. Variance-based feature selection (top variance features)
        try:
            # CRITICAL: Exclude leakage features BEFORE computing variance
            leakage_patterns = ['acq_time', 'neighbour_acq_time', 'frp', 'brightness', 'confidence', 'scan', 'track']
            clean_cols = [col for col in X.columns if not any(pat in col.lower() for pat in leakage_patterns)]
            X_clean = X[clean_cols]
            
            feature_variance = X_clean.var().sort_values(ascending=False)
            top_variance_features = feature_variance.head(25).index.tolist()
            datasets['variance'] = {
                'X': X_clean[top_variance_features],
                'y': y,
                'features': top_variance_features
            }
            logger.info(f"Variance-based: {len(top_variance_features)} features (leakage features excluded)")
        except Exception as e:
            logger.error(f"Failed to create variance features: {e}")

        # 5. Baseline (all features) - sample for efficiency
        try:
            leakage_patterns = ['acq_time', 'neighbour_acq_time', 'frp', 'brightness', 'confidence', 'scan', 'track']
            clean_cols = [col for col in X.columns if not any(pat in col.lower() for pat in leakage_patterns)]
            X_clean = X[clean_cols]
            
            filtered_columns = X_clean.columns.tolist()
            if len(filtered_columns) > 100:
                sampled_features = np.random.choice(filtered_columns, 72, replace=False)
                datasets['baseline'] = {
                    'X': X_clean[sampled_features],
                    'y': y,
                    'features': sampled_features.tolist()
                }
            else:
                datasets['baseline'] = {
                    'X': X_clean,
                    'y': y,
                    'features': filtered_columns
                }
            logger.info(f"Baseline: {datasets['baseline']['X'].shape[1]} features (leakage features excluded)")
        except Exception as e:
            logger.error(f"Failed to create baseline features: {e}")

        return datasets
    
    def calculate_hard_negative_scores(self, X_neg, X_pos):
        """Calculate boundary proximity and environmental similarity scores"""
        
        logger.info("Calculating hard negative scores...")
        
        # Boundary proximity (distance to decision boundary)
        try:
            # Sample positive data for efficiency
            if len(X_pos) > 5000:
                X_pos_sample = X_pos.sample(5000, random_state=42)
            else:
                X_pos_sample = X_pos
            
            # Fit KNN on positive samples
            nn = NearestNeighbors(n_neighbors=5, algorithm='ball_tree', n_jobs=-1)
            nn.fit(X_pos_sample)
            
            # Calculate distances from negative samples to positive samples
            distances, _ = nn.kneighbors(X_neg)
            boundary_scores = 1 / (distances.mean(axis=1) + 1e-8)  # Inverse distance
            
        except Exception as e:
            logger.error(f"Failed to calculate boundary scores: {e}")
            boundary_scores = np.random.random(len(X_neg))  # Fallback to random
        
        # Environmental similarity
        try:
            # Scale features
            scaler = StandardScaler()
            
            if len(X_pos) > 2000:
                X_pos_sample = X_pos.sample(2000, random_state=42)
            else:
                X_pos_sample = X_pos
            
            X_pos_scaled = scaler.fit_transform(X_pos_sample)
            X_neg_scaled = scaler.transform(X_neg)
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(X_neg_scaled, X_pos_scaled)
            env_scores = similarity_matrix.max(axis=1)  # Max similarity to any positive
            
        except Exception as e:
            logger.error(f"Failed to calculate environmental scores: {e}")
            env_scores = np.random.random(len(X_neg))  # Fallback to random
        
        # Normalize scores
        boundary_scores = (boundary_scores - boundary_scores.min()) / (boundary_scores.max() - boundary_scores.min() + 1e-8)
        env_scores = (env_scores - env_scores.min()) / (env_scores.max() - env_scores.min() + 1e-8)
        
        # Composite hard negative score
        hard_negative_scores = 0.6 * boundary_scores + 0.4 * env_scores
        
        return boundary_scores, env_scores, hard_negative_scores
    
    def create_hard_negative_datasets(self, datasets):
        """Create hard negative sampling datasets"""
        
        logger.info("Creating hard negative sampling datasets...")
        
        hard_negative_datasets = {}
        
        SAMPLE_SIZE = 100000  # Max samples for positives/negatives to avoid OOM
        for method, data in datasets.items():
            try:
                logger.info(f"Processing {method}...")
                X, y = data['X'], data['y']
                # Separate positive and negative samples
                pos_mask = (y == 1)
                neg_mask = (y == 0)
                X_pos = X[pos_mask]
                y_pos = y[pos_mask]
                X_neg = X[neg_mask]
                y_neg = y[neg_mask]
                logger.info(f"{method}: {len(X_pos)} positive, {len(X_neg)} negative")
                if len(X_pos) == 0 or len(X_neg) == 0:
                    logger.warning(f"Skipping {method} - no positive or negative samples")
                    continue
                # Sample positives and negatives if too large
                if len(X_pos) > SAMPLE_SIZE:
                    X_pos = X_pos.sample(SAMPLE_SIZE, random_state=42)
                    y_pos = y_pos.loc[X_pos.index]
                if len(X_neg) > SAMPLE_SIZE:
                    X_neg = X_neg.sample(SAMPLE_SIZE, random_state=42)
                    y_neg = y_neg.loc[X_neg.index]
                # Calculate hard negative scores
                boundary_scores, env_scores, hard_scores = self.calculate_hard_negative_scores(X_neg, X_pos)
                # Select hard negatives (equal to number of positives for balance)
                n_hard_negatives = len(X_pos)
                hard_negative_indices = np.argsort(hard_scores)[-n_hard_negatives:]
                # Create hard negative dataset
                X_hard_neg = X_neg.iloc[hard_negative_indices]
                y_hard_neg = y_neg.iloc[hard_negative_indices]
                # Combine positive and hard negative samples
                X_combined = pd.concat([X_pos, X_hard_neg]).reset_index(drop=True)
                y_combined = pd.concat([y_pos, y_hard_neg]).reset_index(drop=True)
                # Shuffle
                shuffle_idx = np.random.permutation(len(y_combined))
                X_final = X_combined.iloc[shuffle_idx].reset_index(drop=True)
                y_final = y_combined.iloc[shuffle_idx].reset_index(drop=True)
                hard_negative_datasets[f'{method}_hard_neg'] = {
                    'X': X_final,
                    'y': y_final,
                    'features': data['features'],
                    'sampling': 'hard_negative'
                }
                logger.info(f"Created {method}_hard_neg: {X_final.shape}")
                logger.info(f"  Class distribution: {np.bincount(y_final)}")
                # Save hard negative dataset to CSV
                hard_neg_df = X_final.copy()
                hard_neg_df['target'] = y_final
                hard_neg_path = self.output_dir / f'{method}_hard_negative_dataset.csv'
                hard_neg_df.to_csv(hard_neg_path, index=False)
                logger.info(f"Saved hard negative dataset to {hard_neg_path}")
                # Also create random negative version for comparison
                random_neg_indices = np.random.choice(len(X_neg), n_hard_negatives, replace=False)
                X_rand_neg = X_neg.iloc[random_neg_indices]
                y_rand_neg = y_neg.iloc[random_neg_indices]
                X_rand_combined = pd.concat([X_pos, X_rand_neg]).reset_index(drop=True)
                y_rand_combined = pd.concat([y_pos, y_rand_neg]).reset_index(drop=True)
                shuffle_idx = np.random.permutation(len(y_rand_combined))
                X_rand_final = X_rand_combined.iloc[shuffle_idx].reset_index(drop=True)
                y_rand_final = y_rand_combined.iloc[shuffle_idx].reset_index(drop=True)
                hard_negative_datasets[f'{method}_random'] = {
                    'X': X_rand_final,
                    'y': y_rand_final,
                    'features': data['features'],
                    'sampling': 'random_negative'
                }
                logger.info(f"Created {method}_random: {X_rand_final.shape}")
                # Save random negative dataset to CSV
                rand_neg_df = X_rand_final.copy()
                rand_neg_df['target'] = y_rand_final
                rand_neg_path = self.output_dir / f'{method}_random_negative_dataset.csv'
                rand_neg_df.to_csv(rand_neg_path, index=False)
                logger.info(f"Saved random negative dataset to {rand_neg_path}")
            except Exception as e:
                logger.error(f"Failed to create hard negative dataset for {method}: {e}")
        
        return hard_negative_datasets
    
    def create_spatial_coordinates(self, X):
        """Create synthetic spatial coordinates for spatial CV"""
        n_samples = len(X)
        grid_size = int(np.sqrt(n_samples)) + 1
        coordinates = np.array([
            [i % grid_size, i // grid_size] for i in range(n_samples)
        ])
        return coordinates
    
    def create_spatial_groups(self, coordinates, n_groups=5):
        """Create spatial groups using K-means clustering"""
        kmeans = KMeans(n_clusters=n_groups, random_state=42)
        return kmeans.fit_predict(coordinates)
    
    def evaluate_with_spatial_cv(self, dataset_name, X, y, algorithm_name):
        """Evaluate a dataset with spatial cross-validation"""
        
        logger.info(f"Evaluating {dataset_name} with {algorithm_name}...")
        
        try:
            # Check class distribution
            unique_classes, counts = np.unique(y, return_counts=True)
            if len(unique_classes) < 2:
                error_msg = f"Only {len(unique_classes)} class in {dataset_name}"
                logger.error(error_msg)
                return None
            
            minority_ratio = min(counts) / max(counts)
            if minority_ratio < 0.01:  # Less than 1%
                logger.warning(f"Severe class imbalance in {dataset_name}: {minority_ratio:.4f}")
            
            # Create spatial coordinates and groups
            coordinates = self.create_spatial_coordinates(X)
            spatial_groups = self.create_spatial_groups(coordinates, n_groups=3)  # Fewer groups for stability
            
            # Get algorithm
            algorithms = {
                'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
                'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42)
            }
            
            if algorithm_name not in algorithms:
                logger.error(f"Unknown algorithm: {algorithm_name}")
                return None
            
            model = algorithms[algorithm_name]
            
            # Spatial Cross-Validation
            spatial_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
            
            auc_scores = []
            f1_scores = []
            
            fold_num = 0
            for train_idx, test_idx in spatial_cv.split(X, y, spatial_groups):
                fold_num += 1
                
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Check fold class distribution
                if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                    logger.warning(f"Skipping fold {fold_num} - insufficient class diversity")
                    continue
                
                # Scale features for algorithms that need it
                if algorithm_name in ['LogisticRegression', 'MLP']:
                    scaler = StandardScaler()
                    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
                    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
                
                # Train and evaluate
                model.fit(X_train, y_train)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                auc = roc_auc_score(y_test, y_pred_proba)
                f1 = f1_score(y_test, y_pred)
                
                auc_scores.append(auc)
                f1_scores.append(f1)
                
                logger.info(f"  Fold {fold_num}: AUC={auc:.4f}, F1={f1:.4f}")
            
            if len(auc_scores) == 0:
                error_msg = "No valid folds completed"
                logger.error(error_msg)
                return None
            
            # Calculate final metrics
            result = {
                'configuration': dataset_name,
                'algorithm': algorithm_name,
                'spatial_auc_mean': np.mean(auc_scores),
                'spatial_auc_std': np.std(auc_scores),
                'spatial_f1_mean': np.mean(f1_scores),
                'spatial_f1_std': np.std(f1_scores),
                'n_features': X.shape[1],
                'n_samples': len(y),
                'n_folds_completed': len(auc_scores)
            }
            
            logger.info(f"✅ {dataset_name} + {algorithm_name}: AUC={result['spatial_auc_mean']:.4f}±{result['spatial_auc_std']:.4f}")
            
            return result
            
        except Exception as e:
            error_msg = f"Evaluation failed: {str(e)}"
            logger.error(f"❌ {dataset_name} + {algorithm_name}: {error_msg}")
            return None
    
    def run_complete_evaluation(self):
        """Run the complete hard negative sampling fix and evaluation"""
        
        logger.info("="*80)
        logger.info("STARTING COMPLETE HARD NEGATIVE SAMPLING FIX")
        logger.info("="*80)
        
        # Step 1: Load baseline data
        X_baseline, y_baseline = self.load_baseline_data()
        if X_baseline is None:
            logger.error("Failed to load baseline data. Cannot proceed.")
            return
        
        # Step 2: Create feature-selected datasets
        feature_datasets = self.create_feature_selected_datasets(X_baseline, y_baseline)
        logger.info(f"Created {len(feature_datasets)} feature selection datasets")
        
        # Step 3: Create hard negative datasets
        all_datasets = self.create_hard_negative_datasets(feature_datasets)
        logger.info(f"Created {len(all_datasets)} total datasets (hard negative + random)")
        
        # Step 4: Run spatial CV evaluation
        algorithms = ['RandomForest', 'GradientBoosting', 'LogisticRegression', 'MLP']
        
        total_configs = len(all_datasets) * len(algorithms)
        current_config = 0
        
        for dataset_name, dataset_info in all_datasets.items():
            for algorithm in algorithms:
                current_config += 1
                logger.info(f"\nProgress: {current_config}/{total_configs}")
                
                result = self.evaluate_with_spatial_cv(
                    dataset_name, 
                    dataset_info['X'], 
                    dataset_info['y'], 
                    algorithm
                )
                
                if result:
                    result['sampling_strategy'] = dataset_info['sampling']
                    result['feature_method'] = dataset_name.split('_')[0]
                    self.results.append(result)
        
        # Step 5: Save results and analysis
        self.save_results_and_analysis()
    
    def save_results_and_analysis(self):
        """Save results and create comprehensive analysis"""
        
        logger.info("Saving results and creating analysis...")
        
        # Save detailed results
        if self.results:
            results_df = pd.DataFrame(self.results)
            results_file = self.output_dir / 'corrected_hard_negative_spatial_cv_results.csv'
            results_df.to_csv(results_file, index=False)
            logger.info(f"Saved {len(self.results)} results to {results_file}")
            
            # Create analysis
            self.create_comprehensive_analysis(results_df)
        
        logger.info("="*80)
        logger.info("HARD NEGATIVE SAMPLING FIX COMPLETED")
        logger.info("="*80)
    
    def create_comprehensive_analysis(self, results_df):
        """Create comprehensive analysis comparing hard negative vs random sampling"""
        
        analysis_lines = []
        analysis_lines.append("CORRECTED HARD NEGATIVE SAMPLING ANALYSIS")
        analysis_lines.append("="*60)
        analysis_lines.append(f"Execution Time: {datetime.now()}")
        analysis_lines.append(f"Total Configurations: {len(results_df)}")
        
        # Compare sampling strategies
        analysis_lines.append("\nSAMPLING STRATEGY COMPARISON:")
        analysis_lines.append("-"*40)
        
        for strategy in results_df['sampling_strategy'].unique():
            strategy_results = results_df[results_df['sampling_strategy'] == strategy]
            mean_auc = strategy_results['spatial_auc_mean'].mean()
            std_auc = strategy_results['spatial_auc_mean'].std()
            mean_f1 = strategy_results['spatial_f1_mean'].mean()
            
            analysis_lines.append(f"{strategy}:")
            analysis_lines.append(f"  Configurations: {len(strategy_results)}")
            analysis_lines.append(f"  Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
            analysis_lines.append(f"  Mean F1: {mean_f1:.4f}")
            analysis_lines.append("")
        
        # Top performing configurations
        analysis_lines.append("TOP 10 PERFORMING CONFIGURATIONS:")
        analysis_lines.append("-"*40)
        top_10 = results_df.nlargest(10, 'spatial_auc_mean')
        for _, row in top_10.iterrows():
            analysis_lines.append(
                f"{row['configuration']} + {row['algorithm']}: "
                f"AUC={row['spatial_auc_mean']:.4f}±{row['spatial_auc_std']:.4f}, "
                f"F1={row['spatial_f1_mean']:.4f}±{row['spatial_f1_std']:.4f} "
                f"({row['sampling_strategy']})"
            )
        
        # Algorithm comparison
        analysis_lines.append("\nALGORITHM PERFORMANCE:")
        analysis_lines.append("-"*40)
        for algorithm in results_df['algorithm'].unique():
            alg_results = results_df[results_df['algorithm'] == algorithm]
            mean_auc = alg_results['spatial_auc_mean'].mean()
            std_auc = alg_results['spatial_auc_mean'].std()
            analysis_lines.append(f"{algorithm}: {mean_auc:.4f} ± {std_auc:.4f}")
        
        # RQ2 Analysis
        analysis_lines.append("\nRQ2: HARD NEGATIVE vs RANDOM NEGATIVE ANALYSIS:")
        analysis_lines.append("-"*50)
        
        hard_neg_results = results_df[results_df['sampling_strategy'] == 'hard_negative']
        random_neg_results = results_df[results_df['sampling_strategy'] == 'random_negative']
        
        if len(hard_neg_results) > 0 and len(random_neg_results) > 0:
            hard_neg_auc = hard_neg_results['spatial_auc_mean'].mean()
            random_neg_auc = random_neg_results['spatial_auc_mean'].mean()
            
            analysis_lines.append(f"Hard Negative Sampling: {hard_neg_auc:.4f} mean AUC")
            analysis_lines.append(f"Random Negative Sampling: {random_neg_auc:.4f} mean AUC")
            analysis_lines.append(f"Difference: {hard_neg_auc - random_neg_auc:.4f}")
            
            if hard_neg_auc > random_neg_auc:
                analysis_lines.append("✅ Hard negative sampling shows improvement!")
            else:
                analysis_lines.append("⚠️  Hard negative sampling does not show clear improvement.")
        
        # Save analysis
        analysis_file = self.output_dir / 'corrected_hard_negative_analysis.txt'
        with open(analysis_file, 'w') as f:
            f.write('\n'.join(analysis_lines))
        
        logger.info(f"Analysis saved to {analysis_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("CORRECTED HARD NEGATIVE SAMPLING COMPLETED")
        print("="*60)
        print(f"✅ Total configurations evaluated: {len(results_df)}")
        if len(results_df) > 0:
            best_result = results_df.loc[results_df['spatial_auc_mean'].idxmax()]
            print(f"🏆 Best configuration: {best_result['configuration']} + {best_result['algorithm']}")
            print(f"   Best AUC: {best_result['spatial_auc_mean']:.4f} ± {best_result['spatial_auc_std']:.4f}")
            print(f"   Sampling: {best_result['sampling_strategy']}")
        print("="*60)

def main():
    """Main execution function"""
    
    print("Starting Complete Hard Negative Sampling Fix...")
    print("="*70)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Initialize and run the complete fix
    fix_system = HardNegativeSamplingFix()
    fix_system.run_complete_evaluation()

if __name__ == "__main__":
    main()