# Credit Risk Prediction Model

Predicting credit card default using the UCI Credit Card Default dataset (30,000 clients, 23 predictors after dropping ID).

## Dataset

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- **Samples:** 30,000 credit card clients
- **Predictors:** 23 original columns (demographics, payment history, bill amounts, payment amounts), plus 9 derived features
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
│   ├── preprocessing.py        # Shared feature engineering and fold-local pipeline
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

The previous CV scores were contaminated: SMOTE was applied to the complete training set before folds were created. Winsorization also used the test set, and scaling used future validation folds. The corrected pipeline fits every learned preprocessing step and SMOTE inside each fold.

Corrected measured results are stored in `models/evaluation_metrics.json` and rendered in [the HTML report](reports/credit_risk_report.html). The table below comes from the corrected execution on 2026-10-01.

<!-- corrected-results:start -->
| Model | CV AUC (mean ± std) | Test AUC | CV − test |
| --- | ---: | ---: | ---: |
| Logistic Regression | 0.7602 ± 0.0060 | 0.7446 | 0.0156 |
| **Random Forest** | 0.7798 ± 0.0065 | 0.7738 | 0.0060 |
| XGBoost | 0.7649 ± 0.0053 | 0.7658 | -0.0009 |
| LightGBM | 0.7678 ± 0.0047 | 0.7667 | 0.0011 |

Selected model: **Random Forest**. OOF threshold: **0.50**. Test precision: 0.4931, recall: 0.5885, F1: 0.5366. Normalized Gini: 0.5476; KS: 0.4208.
<!-- corrected-results:end -->

The model is selected by mean CV AUC. Its classification threshold maximizes F1 on training out-of-fold predictions, using the existing 0.10–0.85 grid in steps of 0.05. Test results are descriptive and never select the winner or threshold. CV standard deviation describes fold variability; it is not a confidence interval.

The historical test has already informed earlier model selection, so this rerun is a comparison with the earlier procedure, not a fresh independent validation. New untouched data are needed for that confirmation. Correcting leakage does not guarantee a higher test AUC.

### Historical comparison (contaminated CV)

| Model | Historical CV AUC | Historical test AUC |
| --- | ---: | ---: |
| Logistic Regression | 0.7671 | 0.7450 |
| Random Forest | 0.8636 | 0.7729 |
| XGBoost | 0.9301 | 0.7607 |
| LightGBM | 0.9344 | 0.7659 |

These scores are retained only to explain the original gap. Synthetic descendants and their source observations could appear on opposite sides of a fold, and validation contained artificial clients. This makes CV optimistic; a changed class ratio alone does not explain a ROC AUC drop. The rerun changes winsorization, scaling, and SMOTE isolation together, so it does not isolate the contribution of SMOTE alone.

See [imbalanced-learn's leakage guidance](https://imbalanced-learn.org/stable/common_pitfalls.html).

## Pipeline

1. **EDA** — Target distribution, correlation analysis, feature distributions, outlier detection, payment status analysis
2. **Feature Engineering** — 9 new features: avg/max/std delay, total delay months, avg bill, bill trend, avg payment, pay ratio, credit utilization
3. **Preprocessing** — Split first; each fold fits zero imputation, winsorization (1st–99th percentile), StandardScaler, and SMOTE inside an imbalanced-learn pipeline
4. **Modeling** — 4 models with 5-fold stratified CV on original training rows; validation and test are never resampled
5. **Evaluation** — Select model by CV AUC and threshold by out-of-fold F1; evaluate test after fixing both. Report normalized Gini (2 × AUC − 1) and KS
6. **Interpretability** — Load the selected pipeline; explain its estimator with fitted preprocessing, using a deterministic sample of up to 500 test clients

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
| 08 | Confusion Matrix      | Model selected by CV        |
| 09 | Threshold Analysis    | Training out-of-fold Precision/Recall/F1 vs threshold  |
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

# Render the measured results
uv run python src/generate_report.py

# Verify isolation and serialization
uv run python -m unittest discover -s tests -v
```

## Reproducibility and artifacts

Run commands from the project root. After changing a generator, regenerate its notebook before execution:

```bash
uv run python src/create_nb_modeling.py
uv run python src/create_nb_shap.py
```

`models/best_pipeline.pkl` is the selected fitted pipeline; named model files also contain complete pipelines, replacing the old estimator-only format. Inference accepts engineered columns in the saved order, without prior imputation or scaling. Use `engineer_features` from `src.preprocessing` for the original predictor columns. Keep the project root on Python's import path when loading these artifacts.

The metrics JSON records model selection, the OOF threshold, per-fold scores, feature columns, and dataset hash. Regenerate the SHAP notebook and HTML report after retraining. SHAP uses original training clients as background where needed and never resamples test clients.

Feature contributions are model explanations, not causal findings. AUC, Gini, and KS alone do not establish suitability for lending decisions.
