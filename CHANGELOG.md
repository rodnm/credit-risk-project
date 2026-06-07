# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-06-07

### Added
- Initial project structure with uv + Python 3.12
- EDA notebook with class distribution, distributions, correlations, heatmap, and bivariate analysis
- Modeling notebook with preprocessing (imputation, winsorization, feature engineering, SMOTE)
- 4 models: Logistic Regression, Random Forest, XGBoost, LightGBM with 5-fold CV
- Banking metrics evaluation: AUC-ROC, Gini, KS, Average Precision
- ROC curves and cost-sensitive threshold analysis
- SHAP interpretability notebook: summary, beeswarm, waterfall, dependence plots
- 12 exported figures in reports/figures/
- Serialized models in models/
- README with real results
