# Severe Neutropenia Prediction

Machine-learning pipeline in R for predicting **severe neutropenia** in adult cancer patients receiving chemotherapy. The pipeline compares four model families (Random Forest, XGBoost, SVM, Logistic regression) under four class-imbalance strategies, calibrates the predicted probabilities, and evaluates clinical usefulness with decision curve analysis and SHAP.

> Adjust the clinical description above to match your study protocol before publishing.

## Methods overview

| Step | Description |
|------|-------------|
| Data cleaning | Adults only (age >= 18), removal of identifier/leakage columns and zero-variance columns, outlier inspection |
| Split | Stratified 70/30 train/test split |
| Imputation | Predictive mean matching (`mice`) fitted on the training set only; test set filled with the training median |
| Encoding | Dummy encoding (`caret::dummyVars`, full rank) fitted on the training set |
| Feature selection | Recursive feature elimination (10-fold CV) run separately for each model family; the smallest subset within 0.01 AUC of the best is retained |
| Imbalance handling | None, down-sampling (3:1), SMOTE, class weights |
| Models | 14 models: RF x4, XGBoost x4, SVM x4, Logistic regression and LASSO |
| Tuning | 10-fold CV grid search optimising AUC |
| Calibration | Platt scaling fitted on out-of-fold training predictions |
| Evaluation | AUC with 95% CI, sensitivity, specificity, PPV, accuracy, F1 at three thresholds (0.5, Youden, sensitivity >= 0.70) |
| Calibration & utility | Calibration curves, Brier score, decision curve analysis |
| Interpretation | SHAP values for the best model (`fastshap`, `shapviz`) |

## Repository structure

```
severe-neutropenia-prediction/
├── README.md
├── run_all.R                      # runs the full pipeline
├── config.R                       # paths, seed, column names, colours, plot labels
├── R/                             # reusable functions
│   ├── 00_packages.R
│   ├── utils.R
│   ├── rfe_functions.R            # custom RFE wrappers (XGBoost, SVM, LASSO)
│   ├── model_helpers.R            # tuning and prediction helpers
│   ├── calibration_helpers.R      # out-of-fold scores and Platt scaling
│   └── evaluation_helpers.R       # metrics, confidence intervals, calibration, net benefit
├── scripts/                       # pipeline steps, executed in numeric order
│   ├── 01_data_preprocessing.R
│   ├── 02_feature_selection.R
│   ├── 03_model_training.R
│   ├── 04_tuning_summary.R
│   ├── 05_platt_calibration.R
│   ├── 06_test_evaluation.R
│   ├── 07_performance_plots.R
│   ├── 08_data_diagnostics.R
│   ├── 09_shap_analysis.R
│   ├── 10_calibration_analysis.R
│   └── 11_decision_curve_analysis.R
├── data/                          # place the dataset here (not tracked by git)
└── output/
    └── tables/                    # generated Excel result tables
```

## Requirements

- R >= 4.2
- Packages:

```r
install.packages(c(
  "readxl", "writexl", "dplyr", "tidyr", "mice", "caret", "pROC",
  "ranger", "randomForest", "glmnet", "xgboost", "e1071", "doParallel",
  "ggplot2", "gridExtra", "ggrepel", "scales", "fastshap", "shapviz",
  "themis"
))
```

`caret` uses `themis` for `sampling = "smote"`; `randomForest` is required by the RFE step.

## Data

The clinical dataset is **not included** because it contains patient-level information. To run the pipeline, place an Excel file at:

```
data/neutropenia_data.xlsx
```

The expected column names are defined in `config.R` (they follow the original Vietnamese-language dataset). To use a different dataset, edit `DATA_PATH`, `TARGET_COL`, `NUMERIC_COLS`, `CATEGORICAL_COLS` and `DROP_COLS` in `config.R`.

## Usage

Open the project folder in RStudio (or set it as the working directory), then run:

```r
source("run_all.R")
```

Scripts share objects through the global environment, so they must run in numeric order. The simplest way to do this is through `run_all.R`. Result tables are written to `output/tables/`; plots are drawn to the active graphics device.

## Reproducibility

- Global seed `123` (`SEED` in `config.R`) with `RNGkind(sample.kind = "Rejection")`.
- Parallel jobs use `clusterSetRNGStream` with the same seed.
- Numerical results can still differ slightly across operating systems, R versions, package versions and core counts.
