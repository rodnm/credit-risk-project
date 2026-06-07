# Credit Risk Prediction Model

Predicting credit card default using the UCI Credit Card Default dataset (30,000 clients, 24 features).

## Dataset

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- **Samples:** 30,000 credit card clients
- **Features:** 24 (demographics, payment history, bill amounts, payment amounts)
- **Target:** `default` — binary (1 = default next month, 0 = no default)
- **Default rate:** 22.12%

## Tech Stack

| Component          | Tool                            |
| ------------------ | ------------------------------- |
| Language           | Python 3.12                     |
| Package Manager    | uv                              |
| ML Frameworks      | scikit-learn, XGBoost, LightGBM |
| Imbalance Handling | imbalanced-learn (SMOTE)        |
| Interpretability   | SHAP                            |
| Visualization      | matplotlib, seaborn             |
| Notebooks          | Jupyter                         |

## Project Structure

```
credit-risk-project/
├── data/raw/                    # UCI dataset (.xls)
├── notebooks/
│   ├── 01_EDA.ipynb            # Exploratory Data Analysis
│   ├── 02_Modeling.ipynb       # Preprocessing, training, evaluation
│   └── 03_Interpretability.ipynb # SHAP analysis
├── models/                      # Serialized models (.pkl)
├── reports/figures/             # Generated charts (14 PNGs)
├── src/
│   ├── utils.py                # Utility functions
│   ├── download_data.py        # Data download script
│   ├── create_notebooks.py     # Notebook generator (EDA)
│   ├── create_nb_modeling.py   # Notebook generator (Modeling)
│   └── create_nb_shap.py       # Notebook generator (SHAP)
├── docs/
│   └── credit-risk-project-plan.md
├── CHANGELOG.md
├── pyproject.toml
└── README.md
```

## Results

### Model Comparison

| Model                   | CV AUC (5-fold)  | Test AUC         | Gini             | KS Statistic     |
| ----------------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| Logistic Regression     | 0.7671           | 0.7450           | 0.3816           | 0.3929           |
| **Random Forest** | **0.8636** | **0.7729** | **0.4251** | **0.4145** |
| XGBoost                 | 0.9301           | 0.7607           | 0.4061           | 0.3954           |
| LightGBM                | 0.9344           | 0.7659           | 0.4141           | 0.3951           |

**Best model: Random Forest** (Test AUC: 0.7729, Gini: 0.4251, KS: 0.4145)

### Classification Report (Random Forest)

| Class              | Precision | Recall | F1-Score       |
| ------------------ | --------- | ------ | -------------- |
| No Default         | 0.88      | 0.83   | 0.85           |
| Default            | 0.49      | 0.59   | 0.53           |
| **Accuracy** |           |        | **0.77** |

### Optimal Threshold

- **Threshold:** 0.55 (max F1)
- **Precision:** 0.5395 | **Recall:** 0.5358 | **F1:** 0.5376

## Pipeline

1. **EDA** — Target distribution, correlation analysis, feature distributions, outlier detection, payment status analysis
2. **Feature Engineering** — 9 new features: avg/max/std delay, total delay months, avg bill, bill trend, avg payment, pay ratio, credit utilization
3. **Preprocessing** — Winsorization (1st-99th percentile), StandardScaler, SMOTE oversampling
4. **Modeling** — 4 models with 5-fold stratified CV, trained on SMOTE-balanced data
5. **Evaluation** — ROC/PR curves, confusion matrix, threshold analysis, Gini coefficient, KS statistic
6. **Interpretability** — SHAP summary, bar, dependence, waterfall, and force plots

## Generated Visualizations

| #  | Figure                | Description                       |
| -- | --------------------- | --------------------------------- |
| 01 | Target Distribution   | Class imbalance visualization     |
| 02 | Correlation Heatmap   | Feature correlation matrix        |
| 03 | Feature Distributions | Key features by default status    |
| 04 | Box Plots             | Outlier detection by class        |
| 05 | Payment Status        | Default rate by payment delay     |
| 06 | ROC Curves            | All 4 models comparison           |
| 07 | PR Curves             | Precision-Recall curves           |
| 08 | Confusion Matrix      | Best model (Random Forest)        |
| 09 | Threshold Analysis    | Precision/Recall/F1 vs threshold  |
| 10 | SHAP Summary          | Feature importance + direction    |
| 11 | SHAP Bar              | Feature importance ranking        |
| 12 | SHAP Dependence       | Top feature interactions          |
| 13 | SHAP Waterfall        | Individual prediction explanation |
| 14 | SHAP Force            | Force plot visualization          |

## How to Run

```bash
# Install dependencies
uv sync

# Download dataset
uv run python src/download_data.py

# Execute notebooks
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_EDA.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_Modeling.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/03_Interpretability.ipynb
```

## Key Takeaways

- **Random Forest** outperforms gradient boosting models on test AUC despite lower CV scores, suggesting better generalization
- **Payment delay history** (PAY_0, avg_delay) is the strongest predictor of default
- **Credit utilization** (BILL_AMT1 / LIMIT_BAL) is a key engineered feature
- SMOTE helps address class imbalance but introduces some overfitting risk (high CV vs test AUC gap for boosting models)
- Banking-relevant metrics (Gini > 0.4, KS > 0.4) indicate the model meets minimum acceptance thresholds for credit scoring
