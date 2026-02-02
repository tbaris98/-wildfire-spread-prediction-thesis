#!/bin/bash
# run_full_pipeline.sh
# This script runs the entire wildfire prediction pipeline step-by-step.


# 1. Feature Engineering
python wildfire_feature_engineering_fixed.py

# 2. Feature List Extraction (for appendix)
python extract_feature_list.py

# 3. Feature Selection
python feature_selection_framework.py

# 4. Hard Negative Sampling & Evaluation
python complete_hard_negative_fix.py

# 5. Main Model Training & Spatial CV
python corrected_spatial_cv_framework.py

# 6. Subgroup Analysis
python run_subgroup_analysis.py

# 7. Visualization Scripts
python create_thesis_visualizations.py
python create_confusion_matrix_figures.py
python create_spatial_cv_summary_figures.py
python thesis_all_figures.py

# 8. Sample Distribution Analysis
python github/spatial_cv_sample_distribution_analysis.py

# 9. (Optional) Any additional scripts
# python github/your_additional_script.py

# 10. Completion message
echo "\nAll pipeline steps completed. Check wildfire_results/ and figures/ for outputs, logs, and appendix tables."
