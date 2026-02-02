
"""
Feature List Extraction and Annotation Script

This script generates a table of all features used in the wildfire prediction project, annotating:
- Whether a feature was selected for modeling
- Whether it is considered 'engineered' (log-transformed, outlier-capped, or inferred from naming patterns)
- Whether it is one-hot encoded
- The reason for selection or removal

Engineered features are annotated based on:
- Explicit transformation logs (log transformation, outlier capping)
- Pattern-based inference (feature names containing terms like 'capped', 'outlier', '_x_', '_sq', 'day', 'dist', etc.)

This approach is used because not all engineered features are explicitly listed in transformation_summary.txt. The annotation is thus both log-based and pattern-inferred, as described in the thesis Feature Engineering subsection.
"""

import os

# Path to your preprocessed data file
csv_path = "wildfire_results/preprocessed_wildfire_data.csv"

# Read only the first line (header) without loading entire file
with open(csv_path, 'r') as f:
    header_line = f.readline().strip()

# Parse the header
columns = header_line.split(',')

# Remove index column if present
if columns[0].lower() in ['index', 'id']:
    columns = columns[1:]

# Remove target variable
features = [col for col in columns if col != 'fire_spread']

# Load log-transformed features from transformation summary (if available)
log_transformed = set()
try:
    with open("wildfire_results/transformation_summary.txt") as f:
        log_section = False
        for line in f:
            if line.strip().startswith("Log-transformed features:"):
                log_section = True
                continue
            if log_section:
                if line.strip() == "" or line.strip().startswith("Outliers capped:"):
                    break
                log_transformed.add(line.strip())
except Exception:
    pass

# Load removed features and reasons from feature_filtering_summary.txt
removal_reasons = {}
try:
    with open("wildfire_results/feature_filtering_summary.txt") as f:
        section = None
        for line in f:
            if line.startswith("Low-variance features"):
                section = "variance"
            elif line.startswith("Highly correlated pairs removed"):
                section = "correlation"
            elif line.startswith("Final feature count"):
                section = None
            elif section and line.strip().startswith("  "):
                feat = line.strip().split(':')[0].strip()
                if section == "variance":
                    removal_reasons[feat] = "Low variance"
                elif section == "correlation":
                    removal_reasons[feat] = "High correlation"
except Exception:
    pass

import csv

# Load all features from features_array.csv (excluding target)
fa_features = []
with open("/home/u427312/wildfire_project/features_array.csv", "r") as f:
    reader = csv.reader(f)
    fa_header = next(reader)
    # If the header is tab-separated, split it properly
    if len(fa_header) == 1 and '\t' in fa_header[0]:
        fa_features = fa_header[0].split('\t')
    elif fa_header[0].lower() in ['index', 'id']:
        fa_features = fa_header[1:]
    else:
        fa_features = fa_header
    # Remove target if present
    if "fire_spread" in fa_features:
        fa_features.remove("fire_spread")

# Load selected features from preprocessed_wildfire_data.csv (excluding target)
pp_features = []
with open("wildfire_results/preprocessed_wildfire_data.csv", "r") as f:
    reader = csv.reader(f)
    pp_header = next(reader)
    if pp_header[0].lower() in ['index', 'id']:
        pp_features = pp_header[1:]
    else:
        pp_features = pp_header
    if "fire_spread" in pp_features:
        pp_features.remove("fire_spread")

# Annotate features
feature_table = []
# List of known leakage-prone features from the feature engineering script
leaked_features = {"acq_time", "Neighbour_acq_time"}
import re
# Patterns for engineered features
engineered_patterns = [
    r"capped", r"outlier", r"_x_", r"_mul_", r"_sq$", r"_cubed$", r"\^2$", r"\^3$",
    r"day", r"hour", r"time", r"month", r"season", r"dist", r"elev", r"slope", r"aspect", r"lat", r"lon",
    r"mean", r"max", r"min", r"sum", r"std", r"window"
]

def get_engineering_steps(feat):
    steps = []
    if feat in log_transformed:
        steps.append("log")
    # Outlier capping: infer from transformation_summary.txt if possible (not currently tracked per-feature)
    # For now, use pattern-based inference
    for pat in engineered_patterns:
        if re.search(pat, feat, re.IGNORECASE):
            steps.append("pattern-inferred")
            break
    return ", ".join(steps) if steps else ""

for feat in fa_features:
    # Default: not selected
    is_selected = "No"
    engineering_steps = get_engineering_steps(feat)
    is_engineered = "Yes" if engineering_steps else ""
    is_onehot = "Yes" if ("_" in feat and not feat.startswith("Neighbour")) else ""
    # Logic for selection/removal
    if feat in leaked_features:
        selection_reason = "Removed (data leakage prevention)"
        engineering_steps = engineering_steps + (", removed: data leakage" if engineering_steps else "removed: data leakage")
    elif feat in removal_reasons:
        selection_reason = f"Removed ({removal_reasons[feat]})"
        engineering_steps = engineering_steps + (", removed: " + removal_reasons[feat] if engineering_steps else "removed: " + removal_reasons[feat])
    elif feat not in pp_features:
        selection_reason = "Removed (missingness or other)"
        engineering_steps = engineering_steps + (", removed: missingness/other" if engineering_steps else "removed: missingness/other")
    else:
        is_selected = "Yes"
        selection_reason = "Selected (passed all filters: not leaked, not high-missing, not low-variance, not highly correlated)"
    feature_table.append([feat, is_selected, is_engineered, is_onehot, selection_reason, engineering_steps])


output_path = "wildfire_results/feature_list_appendix.csv"
with open(output_path, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Feature Name", "Selected", "Engineered (log/outlier)", "One-hot Encoded", "Selection Reason", "Engineering Step(s)"])
    for row in feature_table:
        writer.writerow(row)
    writer.writerow([])
    writer.writerow([f"Total features in features_array.csv (excluding target): {len(fa_features)}"])
    writer.writerow([f"Total features selected for preprocessed_wildfire_data.csv: {len(pp_features)}"])

# Also write to Excel for easier visualization
try:
    import pandas as pd
    df = pd.DataFrame(feature_table, columns=["Feature Name", "Selected", "Engineered (log/outlier)", "One-hot Encoded", "Selection Reason", "Engineering Step(s)"])
    excel_path = "wildfire_results/feature_list_appendix.xlsx"
    df.to_excel(excel_path, index=False)
    print(f"Feature list with selection and annotation saved to {output_path} and {excel_path}")
except ImportError:
    print(f"Feature list with selection and annotation saved to {output_path} (install pandas to get Excel output)")
