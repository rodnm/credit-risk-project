"""Generate interpretability notebook without requiring ML packages at generation."""
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
cells = []

def markdown(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

markdown('''# Credit Risk Prediction — Interpretability

Explain the explicit CV-selected pipeline, using its saved preprocessing without refitting
or SMOTE. Global plots use a reproducible sample of up to 500 original test clients.
Features shown are scaled values. SHAP values explain the estimator output (tree-dependent
probability or raw margin; logistic regression log-odds), not causal effects.''')
code('''from pathlib import Path
import sys
import json
import hashlib
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split

ROOT = Path.cwd() if (Path.cwd() / 'src').is_dir() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
from src.preprocessing import engineer_features, transform_for_explanation
FIGURES = ROOT / 'reports/figures'
metadata = json.loads((ROOT / 'models/evaluation_metrics.json').read_text(encoding='utf-8'))
pipeline = joblib.load(ROOT / 'models/best_pipeline.pkl')
best_model = pipeline.named_steps['classifier']
data_path = ROOT / 'data/raw/default of credit card clients.xls'
if hashlib.sha256(data_path.read_bytes()).hexdigest() != metadata['dataset_sha256']:
    raise ValueError('Dataset differs from training; regenerate modeling artifacts first.')
df = pd.read_excel(data_path, header=1, engine='xlrd')
df = df.rename(columns={'default payment next month': 'default'}).drop(columns='ID')
X = engineer_features(df.drop(columns='default'))[metadata['feature_columns']]
y = df['default']
X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=metadata['split']['test_size'], random_state=metadata['split']['random_state'], stratify=y)
X_sample = X_test.sample(n=min(500, len(X_test)), random_state=42)
X_test_scaled = pd.DataFrame(transform_for_explanation(pipeline, X_sample),
                             columns=X.columns, index=X_sample.index)
background = pd.DataFrame(transform_for_explanation(pipeline,
    X_train.sample(n=min(100, len(X_train)), random_state=42)), columns=X.columns)
print(f"Explaining {metadata['best_model']}: {len(X_sample)} original test clients")
''')
markdown('## 1. Explain the selected classifier')
code('''if hasattr(best_model, 'feature_importances_'):
    explainer = shap.TreeExplainer(best_model)
else:
    explainer = shap.LinearExplainer(best_model, background)
shap_values = explainer(X_test_scaled)
# Recent SHAP versions return (samples, features, classes) for binary Random Forest.
if shap_values.values.ndim == 3:
    shap_values = shap_values[:, :, 1]
if shap_values.values.shape != X_test_scaled.shape:
    raise ValueError('Unexpected binary SHAP shape')
''')
markdown('## 2. Global impact and feature importance')
code('''for plot_type, filename, title in [
    ('dot', '10_shap_summary.png', 'SHAP impact — sampled test clients'),
    ('bar', '11_shap_bar.png', 'SHAP importance — sampled test clients')]:
    plt.figure()
    shap.summary_plot(shap_values.values, X_test_scaled, plot_type=plot_type, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIGURES / filename, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close('all')
''')
markdown('## 3. Dependence plots')
code('''top_features = ['PAY_0', 'LIMIT_BAL', 'credit_util']
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for axis, feature in zip(axes, top_features):
    shap.dependence_plot(feature, shap_values.values, X_test_scaled, ax=axis, show=False)
    axis.set_title(f'SHAP dependence: {feature}')
plt.tight_layout()
plt.savefig(FIGURES / '12_shap_dependence.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close('all')
''')
markdown('## 4. Individual prediction')
code('''sample_idx = 0
plt.figure()
shap.plots.waterfall(shap_values[sample_idx], show=False)
plt.tight_layout()
plt.savefig(FIGURES / '13_shap_waterfall.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close('all')
probability = pipeline.predict_proba(X_sample.iloc[[sample_idx]])[0, 1]
print(f'Actual: {y_test.loc[X_sample.index[sample_idx]]}')
print(f'Probability: {probability:.4f}')
print(f"Prediction at saved threshold {metadata['threshold']:.2f}: {int(probability >= metadata['threshold'])}")
shap.force_plot(shap_values[sample_idx].base_values, shap_values[sample_idx].values,
                X_test_scaled.iloc[sample_idx], matplotlib=True, show=False)
plt.tight_layout()
plt.savefig(FIGURES / '14_shap_force.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close('all')
(ROOT / 'models/interpretability_metadata.json').write_text(json.dumps({
    'best_model': metadata['best_model'], 'sample_size': len(X_sample),
    'background_size': len(background), 'random_state': 42,
    'sample_indices': X_sample.index.tolist(),
    'dataset_sha256': metadata['dataset_sha256']}, indent=2), encoding='utf-8')
''')
nb.cells = cells
destination = Path(__file__).resolve().parents[1] / 'notebooks/03_Interpretability.ipynb'
nbf.write(nb, destination)
print(f'Created {destination}')
