# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-06-07

### Added
- Initial project structure with uv + Python 3.12
- UCI Credit Card Default dataset (30,000 samples, 22.12% default rate)
- EDA notebook: target distribution, correlation heatmap, feature distributions, box plots, payment status analysis
- Modeling notebook: feature engineering (9 new features), imputation, winsorization, StandardScaler, SMOTE
- 4 models: Logistic Regression, Random Forest, XGBoost, LightGBM with 5-fold stratified CV
- Best model: Random Forest (Test AUC: 0.7729, Gini: 0.4251, KS: 0.4145)
- Banking metrics: ROC/PR curves, confusion matrix, threshold analysis, Gini, KS
- SHAP interpretability: summary, bar, dependence, waterfall, force plots
- 14 exported figures in reports/figures/
- 5 serialized model files in models/
- README with real results and project documentation
