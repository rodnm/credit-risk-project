"""Generate the modeling notebook (generation requires only nbformat)."""
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
cells = []

def markdown(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

markdown('''# Credit Risk Prediction — Modeling & Evaluation

Previously SMOTE ran before fold splitting: related original and synthetic samples could
cross validation boundaries, inflating AUC. Here each fold independently fits imputation,
winsorization, scaling and SMOTE. Validation and test retain original clients.
The historical test has already informed decisions; independent confirmation requires new data.
The goal is reliable evaluation, not a guaranteed reduction in the AUC gap.''')
code('''from pathlib import Path
import sys
import json
import hashlib
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, f1_score, precision_score, recall_score,
    average_precision_score)
ROOT = Path.cwd() if (Path.cwd() / 'src').is_dir() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
from src.preprocessing import engineer_features, make_pipeline
FIGURES = ROOT / 'reports/figures'
MODELS = ROOT / 'models'
FIGURES.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)
sns.set_style('whitegrid')
''')
markdown('## 1. Load data, derive row-local features and reserve test')
code('''data_path = ROOT / 'data/raw/default of credit card clients.xls'
df = pd.read_excel(data_path, header=1, engine='xlrd')
df = df.rename(columns={'default payment next month': 'default'}).drop(columns='ID')
X = engineer_features(df.drop(columns='default'))
y = df['default']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f'Train: {len(X_train)}; test: {len(X_test)}; features: {X.shape[1]}')
''')
markdown('''## 2. CV with SMOTE inside each fold

Each pipeline is imputer → winsorizer → scaler → SMOTE → classifier.
All fitted statistics come exclusively from the fold training partition. SMOTE is skipped
at prediction. Select the model by mean validation AUC only.''')
code('''models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42,
                           eval_metric='logloss', use_label_encoder=False),
    'LightGBM': LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, verbose=-1)
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}
for name, classifier in models.items():
    pipeline = make_pipeline(classifier)
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc', error_score='raise')
    results[name] = {'cv_scores': scores.tolist(), 'cv_mean': float(scores.mean()),
                     'cv_std': float(scores.std()), 'model': pipeline}
    print(f'{name}: CV AUC {scores.mean():.4f} +/- {scores.std():.4f}')
best_model_name = max(results, key=lambda name: results[name]['cv_mean'])
print(f'Selected using CV: {best_model_name}')
''')
markdown('''## 3. Select threshold from training out-of-fold predictions

Maximize F1 over the original grid (0.10–0.85, step 0.05); ties use the smallest threshold.
These OOF scores tune the threshold and are not independent evaluation after model selection.''')
code('''oof_prob = cross_val_predict(results[best_model_name]['model'], X_train, y_train,
                             cv=cv, method='predict_proba')[:, 1]
metrics = []
for threshold in np.arange(0.1, 0.9, 0.05):
    predictions = (oof_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_train, predictions, labels=[0, 1]).ravel()
    metrics.append({'threshold': float(threshold),
                    'precision': float(precision_score(y_train, predictions, zero_division=0)),
                    'recall': float(recall_score(y_train, predictions, zero_division=0)),
                    'f1': float(f1_score(y_train, predictions, zero_division=0)),
                    'fpr': float(fp / (fp + tn))})
metrics_df = pd.DataFrame(metrics)
optimal_threshold = float(metrics_df.loc[metrics_df['f1'].idxmax(), 'threshold'])
print(f'OOF-selected threshold: {optimal_threshold:.2f}')
ax = metrics_df.plot(x='threshold', y=['precision', 'recall', 'f1'], marker='o', figsize=(12, 6))
ax.axvline(optimal_threshold, color='gray', linestyle='--', label='Selected threshold')
ax.set_title('Threshold selection — training out-of-fold predictions')
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES / '09_threshold_analysis.png', dpi=150)
plt.show()
''')
markdown('''## 4. Final fitting and test evaluation

Model and threshold are fixed before reading test outcomes. Other models' test scores are
descriptive comparisons and must not be used to revise the selection.''')
code('''for name, res in results.items():
    res['model'].fit(X_train, y_train)
    prob = res['model'].predict_proba(X_test)[:, 1]
    assert len(prob) == len(X_test)
    fpr, tpr, _ = roc_curve(y_test, prob)
    res.update(y_prob=prob, test_auc=float(roc_auc_score(y_test, prob)),
               average_precision=float(average_precision_score(y_test, prob)),
               gini=float(2 * roc_auc_score(y_test, prob) - 1), ks=float(np.max(tpr - fpr)))
    print(f"{name}: CV {res['cv_mean']:.4f} +/- {res['cv_std']:.4f}; test {res['test_auc']:.4f}")
best_pipeline = results[best_model_name]['model']
y_prob_best = results[best_model_name]['y_prob']
y_pred = (y_prob_best >= optimal_threshold).astype(int)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
best_test_metrics = {'precision': float(precision_score(y_test, y_pred, zero_division=0)),
                     'recall': float(recall_score(y_test, y_pred, zero_division=0)),
                     'f1': float(f1_score(y_test, y_pred, zero_division=0)),
                     'confusion_matrix': cm.tolist()}
print(classification_report(y_test, y_pred, target_names=['No Default', 'Default'], zero_division=0))
''')
markdown('## 5. Test curves and confusion matrix')
code('''for kind, filename in [('roc', '06_roc_curves.png'), ('pr', '07_pr_curves.png')]:
    plt.figure(figsize=(10, 8))
    for name, res in results.items():
        if kind == 'roc':
            horizontal, vertical, _ = roc_curve(y_test, res['y_prob'])
            label = f"{name} (AUC={res['test_auc']:.4f})"
        else:
            vertical, horizontal, _ = precision_recall_curve(y_test, res['y_prob'])
            label = f"{name} (AP={res['average_precision']:.4f})"
        plt.plot(horizontal, vertical, label=label)
    if kind == 'roc':
        plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate' if kind == 'roc' else 'Recall')
    plt.ylabel('True Positive Rate' if kind == 'roc' else 'Precision')
    plt.title('Test ROC curves' if kind == 'roc' else 'Test precision-recall curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / filename, dpi=150)
    plt.show()
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Default', 'Default'], yticklabels=['No Default', 'Default'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'{best_model_name} — OOF threshold {optimal_threshold:.2f}')
plt.tight_layout()
plt.savefig(FIGURES / '08_confusion_matrix.png', dpi=150)
plt.show()
''')
markdown('''## 6. Save complete pipelines and metrics

Inference: derive features using `src.preprocessing.engineer_features`, order columns using
`feature_columns`, and apply the saved threshold to `best_pipeline.predict_proba(X)[:, 1]`.
The classifier's `predict` method uses its default threshold, not the OOF-selected threshold.''')
code('''for name, res in results.items():
    joblib.dump(res['model'], MODELS / f"{name.lower().replace(' ', '_')}.pkl")
joblib.dump(best_pipeline, MODELS / 'best_pipeline.pkl')
metadata = {
    'schema_version': 1, 'pipeline_version': 1,
    'dataset_sha256': hashlib.sha256(data_path.read_bytes()).hexdigest(),
    'best_model': best_model_name, 'threshold': optimal_threshold,
    'selection_metric': 'cv_mean', 'threshold_source': 'training_out_of_fold',
    'feature_columns': X.columns.tolist(),
    'split': {'test_size': 0.2, 'random_state': 42, 'cv_folds': 5},
    'models': {name: {key: value for key, value in res.items() if key not in ('model', 'y_prob')}
               for name, res in results.items()},
    'best_test_metrics': best_test_metrics, 'threshold_metrics': metrics,
    'test_limitation': 'Historically reused test; independent confirmation requires new data.'
}
(MODELS / 'evaluation_metrics.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
print(f'Saved complete pipelines and evaluation_metrics.json to {MODELS}')
''')
nb.cells = cells
destination = Path(__file__).resolve().parents[1] / 'notebooks/02_Modeling.ipynb'
nbf.write(nb, destination)
print(f'Created {destination}')
