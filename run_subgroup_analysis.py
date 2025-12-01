
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / 'corrected_hard_negative_results/tree_based_random_negative_dataset.csv'
OUTPUT_DIR = BASE_DIR / 'figures'
OUTPUT_DIR.mkdir(exist_ok=True)

def load_data():
    print(f"Loading dataset from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)
    return df

# --- Logic adapted from corrected_spatial_cv_framework.py ---

def get_algorithm_config(algorithm_name, is_imbalanced=False):
    """Get algorithm-specific configuration matching the framework"""
    configs = {
        'RandomForest': {
            'balanced': RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            ),
            'regular': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
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

def create_spatial_coordinates(n_samples):
    """Create synthetic spatial coordinates matching framework logic"""
    # Create grid-like coordinates
    grid_size = int(np.sqrt(n_samples)) + 1
    coordinates = np.array([
        [i % grid_size, i // grid_size] for i in range(n_samples)
    ])
    return coordinates

def create_spatial_groups(n_samples, n_groups=5):
    """Create spatial groups using K-means clustering on synthetic coords"""
    coordinates = create_spatial_coordinates(n_samples)
    kmeans = KMeans(n_clusters=n_groups, random_state=42, n_init=10)
    return kmeans.fit_predict(coordinates)

def validate_class_distribution(y):
    """Check for imbalance"""
    unique_classes, counts = np.unique(y, return_counts=True)
    if len(unique_classes) < 2:
        return False, "single_class"
    
    minority_ratio = min(counts) / max(counts)
    if minority_ratio < 0.001:  # Less than 0.1%
        return False, "extreme_imbalance"
    
    return True, "balanced"

def get_balanced_indices(y, strategy='balanced_random', target_ratio=0.3):
    """Return indices for a balanced sample"""
    unique_classes, counts = np.unique(y, return_counts=True)
    majority_class = unique_classes[np.argmax(counts)]
    minority_class = unique_classes[np.argmin(counts)]
    
    majority_mask = (y == majority_class)
    minority_mask = (y == minority_class)
    
    n_minority = np.sum(minority_mask)
    n_majority_target = int(n_minority / target_ratio)
    
    n_majority_available = np.sum(majority_mask)
    n_majority_target = min(n_majority_target, n_majority_available)
    
    majority_indices = np.where(majority_mask)[0]
    selected_majority = np.random.choice(
        majority_indices, n_majority_target, replace=False
    )
    minority_indices = np.where(minority_mask)[0]
    
    selected_indices = np.concatenate([minority_indices, selected_majority])
    np.random.shuffle(selected_indices)
    
    return selected_indices

def run_analysis(df):
    
    target_col = 'target'
    y_raw = df[target_col].values
    
    # 1. Handle Class Imbalance (Framework Logic)
    is_valid, issue = validate_class_distribution(y_raw)
    was_balanced = False
    
    if not is_valid and issue == "extreme_imbalance":
        print("Detected extreme imbalance. Applying balancing strategy...")
        indices = get_balanced_indices(y_raw, strategy='balanced_random', target_ratio=0.2) # Framework uses 0.2 in evaluate_configuration
        df = df.iloc[indices].reset_index(drop=True)
        was_balanced = True
    
    print(f"Final dataset shape: {df.shape}")
    print(f"Class distribution: {df[target_col].value_counts().to_dict()}")
    
    # 2. Prepare X and y
    # Framework drops ONLY target, so X includes metadata and Unnamed: 0 if present
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    
    # Metadata for analysis (we extract from the (potentially balanced) df)
    metadata_cols = ['TEMP_ave', 'Neighbour_EVC_mean', 'c_latitude']
    available_metadata = [c for c in metadata_cols if c in df.columns]
    
    # 3. Create Spatial Groups (Framework Logic)
    groups = create_spatial_groups(len(y), n_groups=5)
    
    cv = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=42)
    
    results = []
    algorithms = ['RandomForest', 'GradientBoosting', 'LogisticRegression']
    
    for name in algorithms:
        print(f"Running analysis for {name}...")
        
        # Get configured model
        model = get_algorithm_config(name, is_imbalanced=was_balanced)
        
        fold = 1
        for train_idx, test_idx in cv.split(X, y, groups):
            print(f"  Fold {fold}...")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Scale for LR/MLP
            if name in ['LogisticRegression', 'MLP']:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Store predictions with metadata
            fold_results = pd.DataFrame({
                'Algorithm': name,
                'Fold': fold,
                'True_Label': y_test,
                'Predicted_Label': y_pred
            })
            
            # Add metadata back
            for col in available_metadata:
                fold_results[col] = df.iloc[test_idx][col].values
                
            results.append(fold_results)
            fold += 1
            
    return pd.concat(results, ignore_index=True)

def analyze_subgroups(full_results):
    print("Analyzing subgroups...")
    
    # 1. Temperature Analysis
    # Bin Temperature
    full_results['Temp_Bin'] = pd.cut(full_results['TEMP_ave'], 
                                      bins=[-np.inf, 10, 20, 30, np.inf],
                                      labels=['<10°C', '10-20°C', '20-30°C', '>30°C'])
    
    # 2. Vegetation Analysis (EVC = Environmental Vegetation Cover?)
    # Assuming Neighbour_EVC_mean is a percentage or index. Let's check range later.
    # For now, use quantiles
    full_results['Veg_Bin'] = pd.qcut(full_results['Neighbour_EVC_mean'], q=4, labels=['Low', 'Medium', 'High', 'Very High'], duplicates='drop')

    # 3. Region (Latitude)
    full_results['Lat_Bin'] = pd.qcut(full_results['c_latitude'], q=4, labels=['South', 'South-Mid', 'North-Mid', 'North'])
    
    # Calculate Error Rates
    metrics = []
    
    for algo in full_results['Algorithm'].unique():
        algo_df = full_results[full_results['Algorithm'] == algo]
        
        for factor, bin_col in [('Temperature', 'Temp_Bin'), ('Vegetation', 'Veg_Bin'), ('Region', 'Lat_Bin')]:
            for bin_val in algo_df[bin_col].unique():
                subset = algo_df[algo_df[bin_col] == bin_val]
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
    
    return pd.DataFrame(metrics)

def plot_results(metrics_df):
    print("Generating plots...")
    sns.set_style("whitegrid")
    
    factors = metrics_df['Factor'].unique()
    
    for factor in factors:
        factor_data = metrics_df[metrics_df['Factor'] == factor].sort_values('Bin')
        
        # Plot FPR
        plt.figure(figsize=(10, 6))
        sns.barplot(data=factor_data, x='Bin', y='FPR', hue='Algorithm')
        plt.title(f'False Positive Rate by {factor}')
        plt.ylabel('False Positive Rate')
        plt.xlabel(factor)
        plt.savefig(OUTPUT_DIR / f'fpr_by_{factor.lower()}.png')
        plt.close()
        
        # Plot FNR
        plt.figure(figsize=(10, 6))
        sns.barplot(data=factor_data, x='Bin', y='FNR', hue='Algorithm')
        plt.title(f'False Negative Rate by {factor}')
        plt.ylabel('False Negative Rate (Missed Fires)')
        plt.xlabel(factor)
        plt.savefig(OUTPUT_DIR / f'fnr_by_{factor.lower()}.png')
        plt.close()

def main():
    df = load_data()
    
    # Run analysis (models are defined inside run_analysis now)
    full_results = run_analysis(df)
    
    # Save raw predictions
    full_results.to_csv(OUTPUT_DIR / 'predictions_with_metadata.csv', index=False)
    
    metrics_df = analyze_subgroups(full_results)
    metrics_df.to_csv(OUTPUT_DIR / 'subgroup_error_metrics.csv', index=False)
    
    plot_results(metrics_df)
    
    print("Analysis complete. Results saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
