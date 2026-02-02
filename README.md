# Wildfire Spread Prediction - Machine Learning Pipeline

This repository contains the complete implementation of the machine learning pipeline for wildfire spread prediction under data sparsity and class imbalance, supporting the Master's thesis:

**"Advancing Wildfire Spread Prediction Under Data Sparsity and Imbalance: The Impact of Feature Selection, Sampling Strategies, and Environmental Bias"**

**Author:** Tuna Barış Ünal  
**Program:** Cognitive Science & Artificial Intelligence  
**Institution:** Tilburg University  
**Supervisor:** Dr. Sharon Ong

---

## Overview

This pipeline addresses critical challenges in wildfire spread prediction:
- **Data sparsity and extreme class imbalance** (9.57:1 ratio)
- **High dimensionality** (149 original features → 20-75 selected features)
- **Spatial generalization** via spatial cross-validation
- **Environmental bias analysis** across temperature, vegetation, and geographic regions

The framework systematically evaluates:
- **6 feature selection methods** (tree-based, correlation, mutual information, RFE, L1, ensemble)
- **2 sampling strategies** (random negative vs. hard negative)
- **4 ML algorithms** (Random Forest, Gradient Boosting, Logistic Regression, MLP)
- **40 total configurations** with spatial cross-validation

---

## Dataset

**WildfireDB** (Singla et al., 2021): Spatio-temporal wildfire spread dataset covering continental US (2012-2017)
- **17.8 million observations** structured as cell-neighbor pairs
- **149 original features** from VIIRS fire detections, LANDFIRE topography/vegetation, NOAA meteorology
- **Target variable:** Binary fire spread (1 = spread occurred, 0 = no spread)

---

## Repository Structure

```
github/
├── README.md                                    # This file
├── requirements.txt                             # Python dependencies
├── environment.yml                              # Conda environment specification
├── run_full_pipeline.sh                         # Master execution script
│
├── wildfire_feature_engineering_fixed.py        # Step 1: Feature engineering
├── extract_feature_list.py                      # Step 2: Feature list extraction
├── feature_selection_framework.py               # Step 3: Feature selection methods
├── complete_hard_negative_fix.py                # Step 4: Hard negative sampling
├── corrected_spatial_cv_framework.py            # Step 5: Spatial cross-validation
├── run_subgroup_analysis.py                     # Step 6: Subgroup reliability analysis
├── spatial_cv_sample_distribution_analysis.py   # Step 8: Sample difficulty analysis
│
└── Visualization Scripts:
    ├── create_thesis_visualizations.py          # Main results visualizations
    ├── create_confusion_matrix_figures.py       # Confusion matrices
    ├── create_spatial_cv_summary_figures.py     # Spatial CV summaries
    ├── create_hard_negative_sampling_figure.py  # Sampling strategy visuals
    ├── create_confusion_figures.py              # Additional confusion analysis
    └── thesis_all_figures.py                    # Comprehensive figure generation
```

---

## Installation

### Option 1: Using Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate wildfire-prediction
```

### Option 2: Using pip

```bash
pip install -r requirements.txt
```

### Required Dependencies

- Python 3.12+
- scikit-learn 1.3.0+
- pandas, numpy
- matplotlib, seaborn
- imbalanced-learn
- scipy

---

## Usage

### Quick Start: Run Full Pipeline

Execute all steps sequentially:

```bash
bash run_full_pipeline.sh
```

This will run:
1. Feature engineering from WildfireDB
2. Feature list extraction (for appendix)
3. Feature selection (6 methods)
4. Hard negative sampling and evaluation
5. Spatial cross-validation training (40 configurations)
6. Subgroup analysis (temperature, vegetation, region)
7. All visualization scripts
8. Sample distribution analysis
9. Generate results in `wildfire_results/` and `figures/`

**Expected Runtime:** ~8 hours on 64 CPU cores, 251GB RAM (Tilburg University GPU4EDU server)

### Run Individual Steps

Each script can be executed independently:

```bash
# Step 1: Feature Engineering
python wildfire_feature_engineering_fixed.py

# Step 3: Feature Selection
python feature_selection_framework.py

# Step 5: Main Model Training
python corrected_spatial_cv_framework.py

# Step 6: Subgroup Analysis
python run_subgroup_analysis.py

# Step 8: Sample Distribution Analysis
python spatial_cv_sample_distribution_analysis.py
```

---

## Pipeline Details

### Step 1: Feature Engineering
**Script:** `wildfire_feature_engineering_fixed.py`

- Loads WildfireDB raw data
- Applies preprocessing (missing value imputation, winsorization at 1st/99th percentiles)
- Engineers 75 final features from topography, vegetation, meteorology, wind
- Outputs: `features_array.csv`

### Step 2: Feature List Extraction
**Script:** `extract_feature_list.py`

- Generates comprehensive feature list for thesis appendix
- Documents feature types, categories, and descriptions
- Outputs: Feature list tables for Appendix A

### Step 3: Feature Selection
**Script:** `feature_selection_framework.py`

- Implements 6 feature selection methods:
  - **Tree-based importance** (Random Forest)
  - **Correlation-based** (Pearson correlation thresholding)
  - **Mutual information** (information gain)
  - **Recursive Feature Elimination (RFE)**
  - **L1 regularization** (Lasso)
  - **Ensemble consensus** (voting across multiple methods)
- Selects top 20 features per method
- Baseline: all 75 features
- Outputs: Feature selection results, rankings

### Step 4: Hard Negative Sampling
**Script:** `complete_hard_negative_fix.py`

- Trains initial Random Forest on full dataset
- Identifies hard negatives: non-spread cases with highest predicted probabilities
- Creates balanced datasets (1:1 ratio, 3.37M samples)
- Compares random negative vs. hard negative sampling
- Outputs: Sampled datasets, probability distributions

### Step 5: Spatial Cross-Validation
**Script:** `corrected_spatial_cv_framework.py`

- **Validation strategy:** StratifiedGroupKFold (groups = geographic regions)
- **Configurations:** 40 total (7 feature methods × 2 sampling × 4 algorithms)
- **Algorithms:** Random Forest, Gradient Boosting, Logistic Regression, MLP
- **Metrics:** AUC, F1-score, PR-AUC, training time
- **Key results:**
  - Best: Tree-based + Random Forest (AUC=0.8542, F1=0.7965)
  - No significant difference: Hard negative vs. random sampling (ΔAUC ≈ 0.0001)
- Outputs: `corrected_spatial_cv_results_new.csv`, `corrected_spatial_cv_analysis.txt`

### Step 6: Subgroup Analysis
**Script:** `run_subgroup_analysis.py`

- Analyzes error patterns across environmental subgroups:
  - **Temperature:** <10°C, 10-20°C, 20-30°C, >30°C
  - **Vegetation density:** Quartiles (Low, Medium, High, Very High)
  - **Geographic region:** Latitude bands (South, South-Mid, North-Mid, North)
- Calculates FPR and FNR per subgroup
- Tests for systematic biases
- Outputs: Subgroup error analysis tables, bias detection results

### Step 7: Visualizations
**Scripts:** `create_thesis_visualizations.py`, `create_confusion_matrix_figures.py`, etc.

- Generates all figures for thesis:
  - Top 10 configurations (AUC and F1)
  - Feature selection method comparison
  - Sampling strategy comparison
  - Model stability (AUC standard deviation)
  - Confusion matrices
  - Spatial CV summaries
  - Hard negative sampling probability distributions
- Outputs: `figures/` directory with publication-ready plots

### Step 8: Sample Distribution Analysis
**Script:** `spatial_cv_sample_distribution_analysis.py`

- Validates sample difficulty assumption for hard negatives
- Generates predicted probability distributions for hard vs. random negatives
- Statistical tests: Kolmogorov-Smirnov, Mann-Whitney U, Cohen's d
- **Key finding:** Hard negatives and random negatives have identical distributions (d < 0.005)
- Outputs: Probability distribution figures, statistical test results

---

## Key Findings

### Research Question 1: Feature Selection Methods
- **Winner:** Tree-based feature selection (AUC=0.8542, 20 features)
- **Surprise:** Baseline (all 75 features) performed nearly identically (AUC=0.8453)
- **Ensemble approach:** No improvement over baseline (AUC=0.8449)
- **Implication:** Tree-based selection achieves best performance with 73% fewer features

### Research Question 2: Sampling Strategies
- **Finding:** Hard negative sampling shows **no advantage** over random sampling
- **Evidence:** ΔAUC = -0.0001 to +0.0013 across all methods
- **Probability analysis:** Distributions statistically indistinguishable (Cohen's d < 0.005)
- **Implication:** Random sampling is superior (equal accuracy, lower computational cost)

### Research Question 3: Environmental Bias
- **Temperature:** Model maintains consistent performance across all temperature ranges
- **Vegetation:** No systematic bias across fuel load densities
- **Geography:** Reliable predictions across latitude bands
- **Conclusion:** Model generalizes well to diverse environmental conditions

### Algorithm Performance
1. **Random Forest:** AUC=0.8388 ± 0.0123 (best)
2. **Gradient Boosting:** AUC=0.8143 ± 0.0042
3. **MLP:** AUC=0.7984 ± 0.0086
4. **Logistic Regression:** AUC=0.7726 ± 0.0077

---

## Output Files

### Results Directory: `wildfire_results/`
- `corrected_spatial_cv_results_new.csv` - Full results (40 configurations)
- `corrected_spatial_cv_analysis.txt` - Summary statistics and rankings
- `subgroup_analysis_results.csv` - Environmental subgroup error rates
- `feature_selection_rankings.csv` - Feature importance scores by method

### Figures Directory: `figures/`
- `top10_configurations.png` - Top performing models
- `auc_by_method.png` - Feature selection comparison
- `auc_by_sampling_strategy.png` - Sampling strategy comparison
- `model_stability_auc.png` - Standard deviation comparison
- `confusion_matrix_*.png` - Confusion matrices for top models
- `probability_distributions_*.png` - Hard vs. random negative distributions

---

## Reproducibility

### Hardware Requirements
- **Minimum:** 16GB RAM, 4 CPU cores
- **Recommended:** 64GB+ RAM, 16+ CPU cores for full pipeline
- **Tested on:** Tilburg University GPU4EDU server (64 cores, 251GB RAM, NVIDIA A40 GPU)

### Random Seeds
All experiments use `random_state=42` for reproducibility:
- scikit-learn models
- train-test splits
- spatial cross-validation folds
- sampling procedures

### Validation Protocol
- **Spatial cross-validation** with StratifiedGroupKFold
- **No data leakage:** Feature selection and sampling performed within training folds
- **Strict evaluation:** No hyperparameter tuning (scikit-learn defaults used)

---

## Citation

If you use this code, please cite:

```bibtex
@mastersthesis{Unal2026,
  author = {Ünal, Tuna Barış},
  title = {Advancing Wildfire Spread Prediction Under Data Sparsity and Imbalance: 
           The Impact of Feature Selection, Sampling Strategies, and Environmental Bias},
  school = {Tilburg University},
  year = {2026},
  type = {Master's Thesis},
  program = {Cognitive Science \& Artificial Intelligence}
}
```

### Related Publications

**WildfireDB Dataset:**
```bibtex
@article{Singla2021,
  author = {Singla, S. and others},
  title = {WildfireDB: An Open-Source Dataset Connecting Wildfire Occurrence with 
           Relevant Determinants},
  journal = {Earth System Science Data},
  year = {2021}
}
```

---

## Contact

**Author:** Tuna Barış Ünal  
**Supervisor:** Dr. Sharon Ong  
**Institution:** Tilburg University, Department of Cognitive Science & Artificial Intelligence

For questions or issues, please open an issue in the repository or contact the author.

---

## License

This project is part of academic research conducted at Tilburg University. Please contact the author for usage permissions.

---

## Acknowledgments

- **Supervisor:** Dr. Sharon Ong (guidance and support)
- **Committee:** Dr. Ben McEwen
- **Infrastructure:** Tilburg University GPU4EDU server
- **Dataset:** WildfireDB (Singla et al., 2021)
- **Tools:** scikit-learn, pandas, matplotlib, seaborn, GitHub Copilot (code assistance)

---

**Last Updated:** February 2, 2026
