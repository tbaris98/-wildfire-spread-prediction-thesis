#!/usr/bin/env python3
"""
Wildfire Spread Prediction: Feature Engineering Pipeline - Server Version (Fixed)

DSS Thesis - Tilburg University
Author: Tuna Baris Unal
Date: September 2025

Fixed to work with actual dataset structure (Neighbour as target variable)
"""


import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import os
from datetime import datetime
from tqdm import tqdm

# --- Centralized Logging Setup ---
import logging
LOG_DIR = Path('wildfire_results')
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f'feature_engineering_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
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
CONFIG_LOG = LOG_DIR / 'feature_engineering_config.txt'
with open(CONFIG_LOG, 'w') as f:
    f.write("# Feature Engineering Pipeline Configuration\n")
    f.write(f"Run timestamp: {datetime.now().isoformat()}\n")
    f.write("Input: features_array.csv\n")
    f.write("Output: wildfire_results/ (feature_list_appendix.csv/xlsx, transformation_summary.txt, logs)\n")
    f.write("Key Steps: missing value handling, low-variance/correlation filtering, log transformation, outlier capping, interaction features, temporal/spatial features\n")
    f.write("Hyperparameters: see code for thresholds (variance, correlation, etc.)\n")

# --- Error Handling Decorator ---
def log_exceptions(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Exception in {func.__name__}: {e}", exc_info=True)
            raise
    return wrapper

# Machine Learning & Feature Engineering
try:
    import sklearn
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.impute import SimpleImputer
    from sklearn.feature_selection import (
        SelectKBest, f_classif, mutual_info_classif,
        RFE, SelectFromModel, VarianceThreshold
    )
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
    print("✅ Scikit-learn loaded successfully")
except ImportError as e:
    print(f"⚠️  Scikit-learn import error: {e}")
    sys.exit(1)

# Optional Advanced Libraries
try:
    import shap
    print("✅ SHAP loaded successfully")
    SHAP_AVAILABLE = True
except ImportError:
    print("⚠️  SHAP not available")
    SHAP_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE
    from imblearn.under_sampling import EditedNearestNeighbours, TomekLinks
    print("✅ Imbalanced-learn loaded successfully")
    IMBLEARN_AVAILABLE = True
except ImportError:
    print("⚠️  Imbalanced-learn not available")
    IMBLEARN_AVAILABLE = False

# Utility imports
import joblib

def create_output_dir():
    """Create output directory for results and plots"""
    output_dir = Path('wildfire_results')
    output_dir.mkdir(exist_ok=True)
    logging.info(f"Output directory: {output_dir.absolute()}")
    return output_dir

def print_header():
    """Print script header information"""
    logging.info("="*60)
    logging.info("🔥 WILDFIRE FEATURE ENGINEERING PIPELINE - SERVER VERSION")
    logging.info("="*60)
    logging.info(f"Python: {sys.version}")
    logging.info(f"Pandas: {pd.__version__}")
    logging.info(f"NumPy: {np.__version__}")
    try:
        logging.info(f"Scikit-learn: {sklearn.__version__}")
    except:
        logging.info("Scikit-learn version not available")
    logging.info(f"Running on: {os.uname().nodename if hasattr(os, 'uname') else 'Unknown'}")
    logging.info(f"Start time: {datetime.now()}")
    logging.info("="*60)

def load_and_explore_dataset(data_path):
    """
    Load the wildfire dataset and perform basic exploration
    
    Args:
        data_path (str): Path to the features_array.csv file
    
    Returns:
        pd.DataFrame: Loaded dataset
    """
    logging.info("\n" + "="*50)
    logging.info("1. LOADING AND EXPLORING DATASET")
    logging.info("="*50)
    
    try:
        # Load the wildfire dataset
        df = pd.read_csv(data_path, 
                        sep='\t',  # TSV format
                        low_memory=False,
                        index_col=0)  # Prevent indexing issues

        logging.info(f"Dataset shape: {df.shape}")
        logging.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Check for target variable - likely 'Neighbour' column
        target_col = 'Neighbour'
        if target_col in df.columns:
            logging.info(f"\n✅ Target variable found: '{target_col}'")
            # Handle target variable - convert to binary if needed
            target_before = df[target_col].value_counts()
            logging.info(f"Target variable distribution (before processing): {target_before.to_dict()}")
            # Create binary target: 1 if Neighbour exists (not NaN), 0 otherwise
            df['fire_spread'] = (~df[target_col].isna()).astype(int)
            target_after = df['fire_spread'].value_counts()
            logging.info(f"Binary target 'fire_spread' distribution: {target_after.to_dict()}")
            logging.info(f"Class balance: {target_after[1]/len(df)*100:.1f}% positive class")
        else:
            logging.warning(f"⚠️  Target column '{target_col}' not found!")
            logging.warning(f"Available columns: {df.columns.tolist()[:10]} ...")
            return None
        
        # Check for any loading issues
        logging.info(f"\nMissing values per column (top 10):")
        missing_counts = df.isnull().sum().sort_values(ascending=False).head(10)
        logging.info(f"{missing_counts}")
        
        # Save basic info to file
        output_dir = Path('wildfire_results')
        with open(output_dir / 'dataset_summary.txt', 'w') as f:
            f.write(f"Dataset shape: {df.shape}\n")
            f.write(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")
            f.write(f"\nColumn data types:\n{df.dtypes.value_counts()}\n")
            f.write(f"\nTarget variable (fire_spread) distribution:\n{target_after}\n")
            f.write(f"Class balance: {target_after[1]/len(df)*100:.1f}% positive class\n")
            f.write(f"\nMissing values (top 10):\n{missing_counts}\n")
        
        logging.info("Dataset summary saved to wildfire_results/dataset_summary.txt")
        
        return df
        
    except Exception as e:
        logging.error(f"❌ Error loading dataset: {e}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)

def handle_missing_values(df, missing_threshold=20):
    """
    Handle missing values by dropping high-missing features and imputing remainder
    
    Args:
        df (pd.DataFrame): Input dataframe
        missing_threshold (float): Percentage threshold for dropping features
    
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    print("\n" + "="*50)
    print("2. HANDLING MISSING VALUES")
    print("="*50)
    
    # Use 'fire_spread' as target variable
    target_col = 'fire_spread'
    
    # Identify high-missing features
    missing_percentages = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    high_missing = missing_percentages[missing_percentages > missing_threshold]

    print(f"Features with >{missing_threshold}% missing values:")
    for feature, pct in tqdm(high_missing.items(), desc="High-missing features", total=len(high_missing)):
        print(f"  {feature}: {pct:.1f}%")

    # Strategy: Drop high-missing features, impute remainder
    features_to_drop = high_missing.index.tolist()
    
    # Don't drop our target variable
    if target_col in features_to_drop:
        features_to_drop.remove(target_col)
        print(f"Keeping target variable '{target_col}' despite missing values")
    
    print(f"\nDropping {len(features_to_drop)} high-missing features")

    df_cleaned = df.drop(columns=features_to_drop)

    # Handle remaining missing values
    remaining_missing = df_cleaned.isnull().sum()
    remaining_missing = remaining_missing[remaining_missing > 0]

    print(f"\nRemaining features with missing values: {len(remaining_missing)}")

    if len(remaining_missing) > 0:
        print("Applying simple imputation strategies (avoiding median/mode per supervisor feedback):")
        
        # Separate target from features
        feature_cols = [col for col in df_cleaned.columns if col != target_col]
        
        # Only impute numerical features
        numeric_features = df_cleaned[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        # Use simple imputation: forward-fill then zero-fill for environmental data
        if numeric_features:
            print("Using forward-fill followed by zero-fill for vegetation/environmental features")
            for feature in tqdm(numeric_features, desc="Imputing missing values"):
                df_cleaned[feature] = df_cleaned[feature].fillna(method='ffill').fillna(0.0)
            
            # Create a dummy imputer for consistency (not actually used)
            imputer = SimpleImputer(strategy='constant', fill_value=0.0)
            imputer.fit(df_cleaned[numeric_features])  # Fit for consistency with pipeline
            
            # Save imputer
            joblib.dump(imputer, 'wildfire_results/missing_value_imputer.pkl')
            print(f"Simple imputation strategy completed for {len(numeric_features)} numeric features")
            print("Imputer saved to wildfire_results/missing_value_imputer.pkl")

    print(f"\nFinal dataset shape: {df_cleaned.shape}")
    print(f"Removed features: {len(df.columns) - len(df_cleaned.columns)}")
    print(f"Remaining features: {len(df_cleaned.columns) - 1}")  # Exclude target
    
    return df_cleaned

def remove_low_variance_and_correlated_features(df, var_threshold=0.01, corr_threshold=0.95):
    """
    Remove low-variance and highly correlated features
    
    Args:
        df (pd.DataFrame): Input dataframe
        var_threshold (float): Variance threshold
        corr_threshold (float): Correlation threshold
    
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    print("\n" + "="*50)
    print("3. REMOVING LOW-VARIANCE AND REDUNDANT FEATURES")
    print("="*50)
    
    target_col = 'fire_spread'
    feature_cols = [col for col in df.columns if col != target_col]
    
    # Only work with numeric features for variance calculation
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    print(f"Working with {len(numeric_features)} numeric features")

    # Calculate variance for numerical features
    if numeric_features:
        variances = df[numeric_features].var()
        low_variance_features = variances[variances < var_threshold].index.tolist()

        print(f"Features with variance < {var_threshold}: {len(low_variance_features)}")
        for feature in low_variance_features[:10]:  # Show first 10
            print(f"  {feature}: {variances[feature]:.6f}")

        # Remove low variance features
        df_filtered = df.drop(columns=low_variance_features)
        numeric_features = [col for col in numeric_features if col not in low_variance_features]

        print(f"Removed {len(low_variance_features)} low-variance features")

        # Remove highly correlated features
        if len(numeric_features) > 1:
            print(f"\nRemoving highly correlated features (>{corr_threshold})...")
            correlation_matrix = df_filtered[numeric_features].corr().abs()

            # Find pairs of highly correlated features
            high_corr_pairs = []

            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    if correlation_matrix.iloc[i, j] > corr_threshold:
                        col1 = correlation_matrix.columns[i]
                        col2 = correlation_matrix.columns[j]
                        high_corr_pairs.append((col1, col2, correlation_matrix.iloc[i, j]))

            print(f"Found {len(high_corr_pairs)} highly correlated pairs (r > {corr_threshold})")

            # Remove one feature from each highly correlated pair
            features_to_remove = []
            for col1, col2, corr_val in high_corr_pairs:
                if col1 not in features_to_remove and col2 not in features_to_remove:
                    features_to_remove.append(col2)
                    print(f"  Removing {col2} (correlated with {col1}: r={corr_val:.3f})")

            df_filtered = df_filtered.drop(columns=features_to_remove)
        else:
            print("Not enough numeric features for correlation analysis")
            features_to_remove = []
            high_corr_pairs = []
    else:
        print("No numeric features found for variance analysis")
        df_filtered = df.copy()
        low_variance_features = []
        features_to_remove = []
        high_corr_pairs = []

    print(f"\nFinal dataset shape after filtering: {df_filtered.shape}")
    print(f"Features remaining: {len(df_filtered.columns) - 1}")  # Exclude target
    
    # Save feature filtering summary
    with open('wildfire_results/feature_filtering_summary.txt', 'w') as f:
        f.write(f"Low-variance features removed: {len(low_variance_features)}\n")
        f.write(f"Highly correlated features removed: {len(features_to_remove)}\n")
        f.write(f"Final feature count: {len(df_filtered.columns) - 1}\n")
        f.write(f"\nLow-variance features:\n")
        for feature in low_variance_features:
            if feature in variances:
                f.write(f"  {feature}: {variances[feature]:.6f}\n")
        f.write(f"\nHighly correlated pairs removed:\n")
        for col1, col2, corr_val in high_corr_pairs:
            if col2 in features_to_remove:
                f.write(f"  {col2} (corr with {col1}: {corr_val:.3f})\n")
    
    return df_filtered

def apply_transformations_and_scaling(df):
    """
    Apply feature transformations (log transform, outlier capping) - SCALING REMOVED
    
    Note: StandardScaler removed to avoid redundancy with downstream algorithm-specific scaling
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        pd.DataFrame: Transformed dataframe (without scaling)
    """
    print("\n" + "="*50)
    print("4. FEATURE TRANSFORMATIONS (SCALING REMOVED FOR OPTIMIZATION)")
    print("="*50)
    
    target_col = 'fire_spread'
    feature_cols = [col for col in df.columns if col != target_col]
    
    # Only work with numeric features
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    print(f"Applying transformations to {len(numeric_features)} numeric features")

    if not numeric_features:
        print("No numeric features found for transformation")
        return df

    # Check for skewed distributions (skewness > 2 or < -2)
    skewness = df[numeric_features].skew()
    highly_skewed = skewness[(skewness > 2) | (skewness < -2)]

    print(f"Highly skewed features (|skewness| > 2): {len(highly_skewed)}")
    if len(highly_skewed) > 0:
        print("Top 10 most skewed features:")
        top_skewed = highly_skewed.abs().sort_values(ascending=False).head(10)
        print(top_skewed)

    # Apply log transformation to positively skewed features with positive values
    log_candidates = []
    for feature in highly_skewed.index:
        if highly_skewed[feature] > 2 and df[feature].min() > 0:
            log_candidates.append(feature)

    print(f"\nApplying log transformation to {len(log_candidates)} features")

    df_transformed = df.copy()
    for feature in tqdm(log_candidates, desc="Log-transforming features"):
        # Add small constant to avoid log(0)
        df_transformed[feature] = np.log1p(df_transformed[feature])
        print(f"  Log-transformed: {feature}")

    # Handle outliers using IQR method for remaining features
    print("\nHandling outliers using IQR capping...")
    outlier_count = 0

    for feature in numeric_features:
        Q1 = df_transformed[feature].quantile(0.25)
        Q3 = df_transformed[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        for feature in tqdm(log_candidates, desc="Log-transforming features"):
            # Add small constant to avoid log(0)
            df_transformed[feature] = np.log1p(df_transformed[feature])
            print(f"  Log-transformed: {feature}")
        if outliers_before > 0:
            # Cap outliers
            df_transformed[feature] = np.clip(df_transformed[feature], lower_bound, upper_bound)
            outlier_count += outliers_before

    print(f"Capped {outlier_count} outlier values across all features")

    # Skip scaling - let downstream algorithms handle it appropriately
    print("\n⚠️  Skipping StandardScaler to avoid redundancy with downstream processing")
    print("Note: Tree-based models work better with unscaled data")
    print("Note: Linear models will be scaled appropriately in later steps")

    print("Feature transformation (without scaling) completed")
    print(f"Final preprocessed dataset shape: {df_transformed.shape}")

    # Save transformation summary (without scaling info)
    with open('wildfire_results/transformation_summary.txt', 'w') as f:
        f.write(f"Log-transformed features: {len(log_candidates)}\n")
        f.write(f"Outliers capped: {outlier_count}\n")
        f.write(f"Scaling: SKIPPED (handled by downstream algorithms)\n")

        f.write(f"\nLog-transformed features:\n")
        for feature in log_candidates:
            f.write(f"  {feature}\n")

        # List outlier-capped features
        f.write(f"\nOutlier-capped features:\n")
        for feature in numeric_features:
            Q1 = df_transformed[feature].quantile(0.25)
            Q3 = df_transformed[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers_before = ((df_transformed[feature] < lower_bound) | (df_transformed[feature] > upper_bound)).sum()
            if outliers_before > 0:
                f.write(f"  {feature} ({outliers_before} capped)\n")

        # List interaction features if present
        interaction_feats = [col for col in df_transformed.columns if '_x_' in col]
        f.write(f"\nInteraction features:\n")
        for feature in interaction_feats:
            f.write(f"  {feature}\n")

        # List other engineered features (temporal/spatial)
        engineered_feats = [col for col in df_transformed.columns if col in ['season','month','day_of_week','time_of_day_cat','log_dist_to_prev_fire','spatial_region']]
        f.write(f"\nOther engineered features (temporal/spatial):\n")
        for feature in engineered_feats:
            f.write(f"  {feature}\n")

    return df_transformed

def main():
    # Create output directory
    output_dir = create_output_dir()

    # Print header
    print_header()

    # Configuration - MODIFY THIS PATH FOR SERVER
    data_path = "/home/u427312/wildfire_project/features_array.csv"

    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        print("Please update the data_path variable in the script")
        sys.exit(1)

    try:
        # Step 1: Load and explore dataset
        df = load_and_explore_dataset(data_path)
        if df is None:
            return

        # Step 2: Handle missing values
        df_cleaned = handle_missing_values(df)

        # Step 3: Remove low-variance and correlated features
        df_filtered = remove_low_variance_and_correlated_features(df_cleaned)

        # Step 4: Create interaction features (advanced feature engineering)
        print("\n" + "="*50)
        print("4. ADVANCED FEATURE ENGINEERING: INTERACTION FEATURES")
        print("="*50)
        df_interact = df_filtered.copy()
        interaction_pairs = [
            ("elevation", "vegetation_height"),
            ("temperature", "humidity")
        ]
        created_interactions = []
        skipped_interactions = []
        for col1, col2 in tqdm(interaction_pairs, desc="Creating interaction features"):
            if col1 in df_interact.columns and col2 in df_interact.columns:
                new_col = f"{col1}_x_{col2}"
                df_interact[new_col] = df_interact[col1] * df_interact[col2]
                print(f"  Created interaction feature: {new_col}")
                created_interactions.append(new_col)
            else:
                print(f"  Skipped interaction: {col1} × {col2} (missing column)")
                skipped_interactions.append(f"{col1} × {col2}")
        # Log interaction feature creation/skipping
        try:
            with open(output_dir / 'interaction_feature_log.txt', 'w') as f:
                f.write("Created interaction features:\n")
                for feat in created_interactions:
                    f.write(f"  {feat}\n")
                if skipped_interactions:
                    f.write("\nSkipped (missing column):\n")
                    for pair in skipped_interactions:
                        f.write(f"  {pair}\n")
        except Exception as e:
            print(f"⚠️  Could not write interaction_feature_log.txt: {e}")

        # Step 4b: Automated creation of temporal and spatial features
        print("\n" + "="*50)
        print("4b. ADVANCED FEATURE ENGINEERING: TEMPORAL & SPATIAL FEATURES")
        print("="*50)
        temporal_spatial_log = []
        # Temporal: Seasonal encoding from 'acquisition_date' or similar
        date_col = None
        for candidate in ["acquisition_date", "date", "timestamp"]:
            if candidate in df_interact.columns:
                date_col = candidate
                break
        if date_col:
            try:
                dates = pd.to_datetime(df_interact[date_col], errors='coerce')
                df_interact['season'] = dates.dt.month % 12 // 3 + 1  # 1=Winter, 2=Spring, 3=Summer, 4=Fall
                df_interact['month'] = dates.dt.month
                df_interact['day_of_week'] = dates.dt.dayofweek
                temporal_spatial_log.append(f"Created 'season', 'month', 'day_of_week' from {date_col}")
                print(f"  Created temporal features: season, month, day_of_week from {date_col}")
            except Exception as e:
                print(f"  Could not create temporal features from {date_col}: {e}")
                temporal_spatial_log.append(f"Failed to create temporal features from {date_col}: {e}")
        else:
            print("  No date column found for temporal features.")
            temporal_spatial_log.append("No date column found for temporal features.")

        # Temporal: Time-of-day (if hour available)
        hour_col = None
        for candidate in ["hour", "acquisition_hour", "time_of_day"]:
            if candidate in df_interact.columns:
                hour_col = candidate
                break
        if hour_col:
            try:
                df_interact['time_of_day_cat'] = pd.cut(df_interact[hour_col], bins=[-1,5,11,17,23], labels=["Night","Morning","Afternoon","Evening"])
                temporal_spatial_log.append(f"Created 'time_of_day_cat' from {hour_col}")
                print(f"  Created time_of_day_cat from {hour_col}")
            except Exception as e:
                print(f"  Could not create time_of_day_cat from {hour_col}: {e}")
                temporal_spatial_log.append(f"Failed to create time_of_day_cat from {hour_col}: {e}")
        else:
            print("  No hour column found for time-of-day feature.")
            temporal_spatial_log.append("No hour column found for time-of-day feature.")

        # Spatial: Distance to previous fires (if columns available)
        dist_col = None
        for candidate in ["dist_to_prev_fire", "distance_to_previous_fire", "fire_distance", "dist"]:
            if candidate in df_interact.columns:
                dist_col = candidate
                break
        if dist_col:
            df_interact['log_dist_to_prev_fire'] = np.log1p(df_interact[dist_col])
            temporal_spatial_log.append(f"Created 'log_dist_to_prev_fire' from {dist_col}")
            print(f"  Created log_dist_to_prev_fire from {dist_col}")
        else:
            print("  No distance column found for spatial feature.")
            temporal_spatial_log.append("No distance column found for spatial feature.")

        # Spatial: Spatial autocorrelation index (if latitude/longitude available)
        lat_col, lon_col = None, None
        for candidate in ["latitude", "lat", "c_latitude"]:
            if candidate in df_interact.columns:
                lat_col = candidate
                break
        for candidate in ["longitude", "lon", "c_longitude"]:
            if candidate in df_interact.columns:
                lon_col = candidate
                break
        if lat_col and lon_col:
            # Simple spatial autocorrelation proxy: bin lat/lon into regions
            df_interact['spatial_region'] = pd.cut(df_interact[lat_col], bins=4, labels=["South","South-Mid","North-Mid","North"])
            temporal_spatial_log.append(f"Created 'spatial_region' from {lat_col}")
            print(f"  Created spatial_region from {lat_col}")
        else:
            print("  No lat/lon columns found for spatial region feature.")
            temporal_spatial_log.append("No lat/lon columns found for spatial region feature.")

        # Save temporal/spatial feature creation log
        try:
            with open(output_dir / 'temporal_spatial_feature_log.txt', 'w') as f:
                for line in temporal_spatial_log:
                    f.write(line + '\n')
        except Exception as e:
            print(f"⚠️  Could not write temporal_spatial_feature_log.txt: {e}")

        # Step 5: Apply transformations (scaling removed for optimization)
        print("\n" + "="*50)
        print("5. FEATURE TRANSFORMATIONS (SCALING REMOVED FOR OPTIMIZATION)")
        print("="*50)
        df_final = apply_transformations_and_scaling(df_interact)
        
        # Save final dataset
        final_path = output_dir / 'preprocessed_wildfire_data.csv'
        try:
            df_final.to_csv(final_path, index=True)
            print(f"\n✅ Final preprocessed dataset saved to: {final_path}")
        except Exception as e:
            print(f"❌ Could not save final dataset: {e}")

        # Save final feature list (excluding target)
        try:
            feature_list = [col for col in df_final.columns if col != 'fire_spread']
            with open(output_dir / 'final_feature_list.txt', 'w') as f:
                f.write("Final features after all processing (excluding target):\n")
                for feat in feature_list:
                    f.write(f"{feat}\n")
            print(f"Final feature list saved to: {output_dir / 'final_feature_list.txt'}")
        except Exception as e:
            print(f"⚠️  Could not write final_feature_list.txt: {e}")

        # Print final summary
        print("\n" + "="*50)
        print("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
        print("="*50)
        print(f"Original dataset shape: {df.shape}")
        print(f"Final dataset shape: {df_final.shape}")
        print(f"Features removed: {df.shape[1] - df_final.shape[1]}")
        print(f"Final feature count: {df_final.shape[1] - 1}")  # Exclude target
        print(f"Target variable: fire_spread")
        if 'fire_spread' in df_final.columns:
            target_dist = df_final['fire_spread'].value_counts()
            print(f"Final class distribution: {target_dist.to_dict()}")
        print(f"End time: {datetime.now()}")
        
        # Save final summary
        try:
            with open(output_dir / 'pipeline_summary.txt', 'w') as f:
                f.write(f"Wildfire Feature Engineering Pipeline - Completed {datetime.now()}\n")
                f.write(f"Original dataset shape: {df.shape}\n")
                f.write(f"Final dataset shape: {df_final.shape}\n")
                f.write(f"Features removed: {df.shape[1] - df_final.shape[1]}\n")
                f.write(f"Final feature count: {df_final.shape[1] - 1}\n")
                f.write(f"Target variable: fire_spread\n")
                if 'fire_spread' in df_final.columns:
                    f.write(f"Final class distribution: {df_final['fire_spread'].value_counts().to_dict()}\n")
        except Exception as e:
            print(f"⚠️  Could not write pipeline_summary.txt: {e}")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()