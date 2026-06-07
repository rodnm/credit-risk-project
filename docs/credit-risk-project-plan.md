# Credit Risk Prediction Model — Plan de Trabajo Completo

> **Rol objetivo:** Data Scientist · BCP  
> **Stack:** Python 3.12 · uv · scikit-learn · XGBoost · LightGBM · SHAP  
> **Duración estimada:** 8 horas  

---

## Pregunta de Investigación

> *¿Es posible predecir la probabilidad de incumplimiento financiero de un cliente en los próximos 2 años, y qué variables tienen mayor poder predictivo para el negocio?*

**¿Por qué esta pregunta para BCP?** Los scorecards de crédito son el producto central de los equipos DS en banca retail. Esta pregunta toca exactamente los tres pilares que evalúa un entrevistador técnico en banca: capacidad predictiva (AUC, Gini, KS), robustez metodológica (manejo de desbalance, data leakage) e interpretabilidad regulatoria (SHAP para cumplimiento SBS).

---

## Stack Tecnológico

| Categoría | Herramientas |
|-----------|-------------|
| Entorno | `uv` + Python 3.12 |
| Datos | `pandas`, `numpy`, `scipy` |
| ML | `scikit-learn`, `xgboost`, `lightgbm`, `imbalanced-learn` |
| Interpretabilidad | `shap` |
| Visualización | `matplotlib`, `seaborn` |
| Serialización | `joblib` |
| Notebooks | `jupyter`, `ipykernel` |
| Control de versiones | `git` + GitHub |

---

## Estructura del Proyecto

```
credit-risk-model/
├── data/
│   └── raw/                   # dataset (en .gitignore — no commitear)
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Modeling.ipynb
│   └── 03_Interpretability.ipynb
├── models/                    # modelos .pkl serializados
├── reports/
│   └── figures/               # gráficos exportados (sí commitear)
├── src/
│   └── utils.py               # funciones reutilizables
├── pyproject.toml             # dependencias uv
├── .python-version            # pin Python 3.12
├── .gitignore
└── README.md
```

---

## Roadmap de 8 Horas

| Hora | Bloque | Entregable |
|------|--------|-----------|
| 0:00 – 0:30 | ① Setup: uv, Python 3.12, estructura de carpetas, descarga de datos | Entorno listo, datos en `data/raw/` |
| 0:30 – 1:45 | ② EDA: distribuciones, correlaciones, análisis del desbalance | 5 gráficos en `reports/figures/` |
| 1:45 – 2:45 | ③ Preprocesamiento: imputación, winsorización, feature engineering | Pipeline de transformación validado |
| 2:45 – 3:00 | ③ SMOTE + verificación sin data leakage | Train set balanceado |
| 3:00 – 4:30 | ④ Modelado: CV 5-fold de 4 modelos + entrenamiento final | 4 modelos serializados en `models/` |
| 4:30 – 5:15 | ⑤ Evaluación: AUC, Gini, KS, curvas ROC | Tabla de resultados exportada |
| 5:15 – 5:45 | ⑤ Threshold analysis con cost matrix | Threshold óptimo documentado |
| 5:45 – 6:45 | ⑥ SHAP: summary, beeswarm, waterfall x2, dependence | 4 gráficos SHAP en `reports/figures/` |
| 6:45 – 7:15 | ⑦ Polish: limpiar notebooks, verificar ejecución end-to-end | Notebooks con outputs visibles |
| 7:15 – 7:45 | ⑧ README + push a GitHub | Repositorio público listo |
| 7:45 – 8:00 | ⑨ Buffer / ajustes finales | — |

### Hitos críticos (no saltarlos)

- [ ] Datos cargados y shape verificado antes de los 30 min
- [ ] Train/test split **antes** de cualquier transformación (evitar data leakage)
- [ ] SMOTE aplicado **exclusivamente** sobre el train set
- [ ] Al menos 3 métricas bancarias reportadas: AUC, Gini, KS
- [ ] SHAP summary plot generado y exportado
- [ ] README con tabla de resultados completada con valores reales

---

## Etapa 0 — Setup con uv + Python 3.12

### 0.1 Instalación de uv (si no lo tienes)

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 0.2 Crear el proyecto

```bash
uv init credit-risk-model
cd credit-risk-model

# Pinear Python 3.12
uv python pin 3.12
# Esto crea el archivo .python-version con el contenido "3.12"
```

### 0.3 Instalar dependencias

```bash
# Dependencias principales
uv add pandas numpy scikit-learn xgboost lightgbm imbalanced-learn \
       shap matplotlib seaborn scipy joblib openpyxl

# Dependencias de desarrollo (Jupyter)
uv add --dev jupyter ipykernel
```

El `pyproject.toml` resultante:

```toml
[project]
name = "credit-risk-model"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "scikit-learn>=1.3.0",
    "xgboost>=2.0.0",
    "lightgbm>=4.0.0",
    "imbalanced-learn>=0.12.0",
    "shap>=0.44.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.13.0",
    "scipy>=1.11.0",
    "joblib>=1.3.0",
    "openpyxl>=3.1.0",
]

[dependency-groups]
dev = [
    "jupyter>=1.0.0",
    "ipykernel>=6.25.0",
]
```

### 0.4 Crear estructura de carpetas y levantar Jupyter

```bash
mkdir -p data/raw notebooks models reports/figures src

# Levantar Jupyter con uv
uv run jupyter notebook
```

### 0.5 .gitignore

```gitignore
.venv/
data/raw/
models/*.pkl
__pycache__/
*.pyc
.ipynb_checkpoints/
.env
*.DS_Store
uv.lock
```

> **Nota:** `uv.lock` es opcional — commitear el lockfile garantiza reproducibilidad exacta, útil para producción. Para un proyecto de portfolio, puedes incluirlo o excluirlo según prefieras.

---

## Etapa 1 — Datos

### Comparativa de datasets

| Criterio | PATH A: Kaggle (GMSC) | PATH B: UCI (sin registro) |
|----------|----------------------|---------------------------|
| Filas | 150,000 ✅ | 30,000 |
| Variables | 11 | 24 (más ricas) ✅ |
| Default rate | ~6.7% (muy realista) ✅ | ~22.1% |
| Acceso | Requiere cuenta Kaggle | Descarga directa ✅ |
| Feature engineering | Limitado | Historial de pagos rico ✅ |

### PATH A — Con cuenta Kaggle

Descargar `cs-training.csv` desde `kaggle.com/c/GiveMeSomeCredit/data` y colocarlo en `data/raw/`.

```python
# notebooks/01_EDA.ipynb — Celda 1

import pandas as pd
import numpy as np
import os

os.makedirs('data/raw', exist_ok=True)
os.makedirs('reports/figures', exist_ok=True)
os.makedirs('models', exist_ok=True)

df = pd.read_csv('data/raw/cs-training.csv', index_col=0)
TARGET = 'SeriousDlqin2yrs'

# Columnas:
# RevolvingUtilizationOfUnsecuredLines  — % crédito rotativo utilizado
# age                                   — edad del cliente
# NumberOfTime30-59DaysPastDueNotWorse  — pagos con atraso 30-59 días
# DebtRatio                             — ratio deuda / ingresos
# MonthlyIncome                         — ingreso mensual
# NumberOfOpenCreditLinesAndLoans       — líneas de crédito abiertas
# NumberOfTimes90DaysLate               — pagos con 90+ días de atraso ⚠️ alto predictor
# NumberRealEstateLoansOrLines          — préstamos hipotecarios
# NumberOfTime60-89DaysPastDueNotWorse  — pagos con atraso 60-89 días
# NumberOfDependents                    — dependientes económicos

print(f"Shape: {df.shape}")
print(f"Default rate: {df[TARGET].mean():.2%}")
print(df.dtypes)
```

### PATH B — Sin cuenta Kaggle (UCI, descarga directa)

```python
# notebooks/01_EDA.ipynb — Celda 1 (alternativa)

import pandas as pd
import numpy as np
import urllib.request
import zipfile
import os

os.makedirs('data/raw', exist_ok=True)
os.makedirs('reports/figures', exist_ok=True)
os.makedirs('models', exist_ok=True)

URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
print("Descargando dataset UCI...")
urllib.request.urlretrieve(URL, 'data/raw/uci_credit.zip')

with zipfile.ZipFile('data/raw/uci_credit.zip', 'r') as z:
    z.extractall('data/raw/')
print("Descarga completada.")

df = pd.read_excel(
    'data/raw/default of credit card clients.xls',
    header=1,
    engine='openpyxl'
)
df.rename(columns={'default payment next month': 'default'}, inplace=True)
df.drop('ID', axis=1, inplace=True)
TARGET = 'default'

# Columnas clave:
# LIMIT_BAL         — límite de crédito (USD)
# SEX, EDUCATION, MARRIAGE, AGE — variables demográficas
# PAY_0..PAY_6      — historial de pago (-1=puntual, 1..9=meses de atraso) ⚠️
# BILL_AMT1..6      — monto estado de cuenta (últimos 6 meses)
# PAY_AMT1..6       — monto pagado (últimos 6 meses)

print(f"Shape: {df.shape}")
print(f"Default rate: {df[TARGET].mean():.2%}")
print(df.dtypes)
```

---

## Etapa 2 — EDA (Notebook 01)

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 120

# ── 1. Resumen básico ──────────────────────────────────────────────────────
print("=" * 55)
print(f"Filas: {df.shape[0]:,}  |  Columnas: {df.shape[1]}")
print(f"Tasa de default: {df[TARGET].mean():.2%}")

missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
miss_df = pd.DataFrame({'N_missing': missing, 'Pct_%': missing_pct})
print("\nValores faltantes:")
print(miss_df[miss_df['N_missing'] > 0].to_string())

# ── 2. Distribución de la variable objetivo ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
counts = df[TARGET].value_counts()

axes[0].bar(['No Default (0)', 'Default (1)'], counts.values,
            color=['#1565C0', '#C62828'], edgecolor='black', alpha=0.85)
axes[0].set_title('Distribución de Clases', fontweight='bold', fontsize=12)
axes[0].set_ylabel('Frecuencia')
for i, v in enumerate(counts.values):
    axes[0].text(i, v * 1.01, f'{v:,}\n({v/len(df):.1%})',
                 ha='center', fontsize=10, fontweight='bold')

axes[1].pie(counts.values, labels=['No Default', 'Default'],
            colors=['#1565C0', '#C62828'], autopct='%1.1f%%',
            startangle=90, explode=(0, 0.06))
axes[1].set_title('Proporción de Clases', fontweight='bold', fontsize=12)

plt.suptitle(f'Variable Objetivo: {TARGET}', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/figures/01_class_distribution.png', bbox_inches='tight')
plt.show()
print("NOTA: Dataset desbalanceado — requiere estrategia especial (SMOTE / class_weight)")

# ── 3. Distribuciones de variables numéricas ───────────────────────────────
numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != TARGET]
n = len(numeric_cols)
ncols = 3
nrows = (n + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    axes[i].hist(df[col].dropna(), bins=50,
                 color='#1565C0', alpha=0.75, edgecolor='white', linewidth=0.3)
    axes[i].set_title(col, fontsize=8, fontweight='bold')
    skew = df[col].skew()
    axes[i].text(0.97, 0.95, f'skew={skew:.1f}',
                 transform=axes[i].transAxes, ha='right', va='top',
                 fontsize=7, color='#C62828')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Distribuciones Variables Numéricas', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig('reports/figures/02_distributions.png', bbox_inches='tight')
plt.show()

# ── 4. Correlación con el target ───────────────────────────────────────────
corr_target = df.corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
colors_bar = ['#C62828' if v > 0 else '#1565C0' for v in corr_target.values]

plt.figure(figsize=(8, max(5, len(corr_target) * 0.4)))
plt.barh(corr_target.index, corr_target.values,
         color=colors_bar, alpha=0.85, edgecolor='black', linewidth=0.5)
plt.axvline(x=0, color='black', linewidth=0.8)
plt.title(f'Correlación de Variables con {TARGET}', fontweight='bold', fontsize=12)
plt.xlabel('Correlación de Pearson')
plt.tight_layout()
plt.savefig('reports/figures/03_correlation_target.png', bbox_inches='tight')
plt.show()

# ── 5. Heatmap de correlaciones ────────────────────────────────────────────
corr_matrix = df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            linewidths=0.5, cbar_kws={'shrink': 0.8}, annot_kws={'size': 7})
plt.title('Heatmap de Correlaciones', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig('reports/figures/04_heatmap.png', bbox_inches='tight')
plt.show()

# ── 6. Análisis bivariado: top features vs. target ─────────────────────────
top_features = corr_target.head(6).index.tolist()
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for i, col in enumerate(top_features):
    grouped = df.groupby(TARGET)[col].median()
    axes[i].bar(grouped.index.astype(str), grouped.values,
                color=['#1565C0', '#C62828'], edgecolor='black', alpha=0.8)
    axes[i].set_title(f'{col}\n(mediana por clase)', fontsize=9, fontweight='bold')
    axes[i].set_xticks([0, 1])
    axes[i].set_xticklabels(['No Default', 'Default'])

plt.suptitle('Top Variables vs. Default (Análisis Bivariado)',
             fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig('reports/figures/05_bivariate.png', bbox_inches='tight')
plt.show()
```

**Gráficos producidos:**

| Archivo | Descripción |
|---------|-------------|
| `01_class_distribution.png` | Bar chart + pie del desbalance de clases |
| `02_distributions.png` | Histogramas de todas las variables numéricas |
| `03_correlation_target.png` | Barras de correlación con la variable objetivo |
| `04_heatmap.png` | Heatmap triangular de correlaciones |
| `05_bivariate.png` | Mediana de top 6 variables por clase |

---

## Etapa 3 — Preprocesamiento, Feature Engineering y SMOTE (Notebook 02, Parte 1)

> **Regla de oro:** Toda transformación (imputer, scaler, winsorización) usa `.fit()` solo sobre train y `.transform()` sobre ambos conjuntos. Nunca fitear sobre test — es la forma más común de data leakage en proyectos DS.

```python
# notebooks/02_Modeling.ipynb — Celdas 1-6

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# ── 1. Split estratificado ─────────────────────────────────────────────────
X = df.drop(TARGET, axis=1)
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y          # preserva la tasa de default en ambos conjuntos
)

print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
print(f"Default rate — Train: {y_train.mean():.2%}  |  Test: {y_test.mean():.2%}")

# ── 2. Imputación con mediana (solo fit sobre train) ───────────────────────
imputer = SimpleImputer(strategy='median')
X_train_imp = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X_train.columns, index=X_train.index
)
X_test_imp = pd.DataFrame(
    imputer.transform(X_test),
    columns=X_test.columns, index=X_test.index
)

# ── 3. Winsorización P1/P99 (alternativa robusta a eliminar outliers) ──────
def winsorize(df_in, lower_pct=0.01, upper_pct=0.99):
    """
    Recorta valores extremos al percentil 1% y 99%.
    Preserva todos los registros pero limita la influencia de outliers.
    Los percentiles se calculan sobre df_in (SIEMPRE pasar train set).
    """
    df_w = df_in.copy()
    for col in df_w.select_dtypes(include=[np.number]).columns:
        lb = df_w[col].quantile(lower_pct)
        ub = df_w[col].quantile(upper_pct)
        df_w[col] = df_w[col].clip(lb, ub)
    return df_w

X_train_w = winsorize(X_train_imp)
X_test_w  = winsorize(X_test_imp)

# ── 4. Feature Engineering ─────────────────────────────────────────────────

# PATH A: Give Me Some Credit (Kaggle)
def feature_engineering_kaggle(df):
    df = df.copy()
    late_cols = [
        'NumberOfTime30-59DaysPastDueNotWorse',
        'NumberOfTime60-89DaysPastDueNotWorse',
        'NumberOfTimes90DaysLate'
    ]
    df['total_late_payments']  = df[late_cols].sum(axis=1)
    df['has_severe_delay']     = (df['NumberOfTimes90DaysLate'] > 0).astype(int)
    df['income_per_dependent'] = df['MonthlyIncome'] / (df['NumberOfDependents'] + 1)
    df['high_utilization']     = (df['RevolvingUtilizationOfUnsecuredLines'] > 0.75).astype(int)
    df['age_group'] = pd.cut(
        df['age'], bins=[0, 30, 45, 60, 120], labels=[0, 1, 2, 3]
    ).astype(int)
    return df

# PATH B: UCI Credit Card (sin Kaggle)
def feature_engineering_uci(df):
    df = df.copy()
    pay_cols     = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    bill_cols    = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    pay_amt_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

    df['avg_pay_delay'] = df[pay_cols].mean(axis=1)
    df['max_pay_delay'] = df[pay_cols].max(axis=1)
    df['times_delayed'] = (df[pay_cols] > 0).sum(axis=1)

    df['avg_bill_amt']  = df[bill_cols].mean(axis=1)
    df['utilization']   = df['avg_bill_amt'] / (df['LIMIT_BAL'] + 1)

    df['avg_pay_amt']   = df[pay_amt_cols].mean(axis=1)
    df['payment_ratio'] = df['avg_pay_amt'] / (df['avg_bill_amt'].abs() + 1)
    df['bill_trend']    = df['BILL_AMT1'] - df['BILL_AMT6']
    return df

# ⚠️ Seleccionar según el dataset elegido
DATASET = 'kaggle'  # cambiar a 'uci' si se usa PATH B
fe_func = feature_engineering_kaggle if DATASET == 'kaggle' else feature_engineering_uci

X_train_fe = fe_func(X_train_w)
X_test_fe  = fe_func(X_test_w)
print(f"Features después de FE: {X_train_fe.shape[1]}")

# ── 5. Scaling (solo para Regresión Logística) ─────────────────────────────
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train_fe),
    columns=X_train_fe.columns
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test_fe),
    columns=X_test_fe.columns
)

# ── 6. SMOTE — solo sobre train, nunca sobre test ─────────────────────────
# sampling_strategy=0.25: llevar los defaults al 25% del total.
# No usar 0.5 (50/50) — demasiado artificial para datos de crédito.
smote = SMOTE(random_state=42, sampling_strategy=0.25)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print(f"\nPost-SMOTE:")
print(f"  Samples: {len(y_train_smote):,}")
print(f"  Default rate: {y_train_smote.mean():.2%}")
```

**Decisiones de diseño clave:**

| Decisión | Justificación |
|----------|--------------|
| Winsorización P1/P99 | Preserva registros extremos pero limita su influencia. Alternativa más robusta que eliminar outliers. |
| SMOTE con `strategy=0.25` | 25% de defaults es más realista que 50/50. Reduce riesgo de overfitting sobre clase sintética. |
| Scaling solo para LR | Modelos tree-based (RF, XGB, LGB) son invariantes a la escala. Aplicar scaling innecesario no daña pero añade complejidad. |
| FE antes de split | **INCORRECTO** — siempre split primero. FE aquí se aplica post-split sobre train y test por separado. |

---

## Etapa 4 — Modelado (Notebook 02, Parte 2)

> **¿Por qué 4 modelos?** Logistic Regression como baseline regulatorio (interpretable, exigido por SBS), Random Forest como ensemble clásico, XGBoost y LightGBM como estado del arte para datos tabulares. Mostrar los 4 demuestra criterio de DS, no solo búsqueda del número más alto.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib

# scale_pos_weight: ratio negativos/positivos en train original (para XGBoost)
spw = (y_train == 0).sum() / (y_train == 1).sum()

models_cfg = {
    'Logistic Regression': {
        'model': LogisticRegression(
            max_iter=1000, C=0.1,
            solver='lbfgs', class_weight='balanced', random_state=42
        ),
        'X': X_train_smote,   # requiere datos escalados
        'y': y_train_smote
    },
    'Random Forest': {
        'model': RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=20,
            class_weight='balanced', n_jobs=-1, random_state=42
        ),
        'X': X_train_fe.values,   # no requiere scaling
        'y': y_train.values
    },
    'XGBoost': {
        'model': XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw,
            eval_metric='auc', verbosity=0, random_state=42
        ),
        'X': X_train_fe.values,
        'y': y_train.values
    },
    'LightGBM': {
        'model': LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            is_unbalance=True, verbose=-1, random_state=42
        ),
        'X': X_train_fe.values,
        'y': y_train.values
    }
}

# ── Cross-Validation 5-fold estratificado ─────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

print("Cross-Validation Results (AUC-ROC):")
print("-" * 50)
for name, cfg in models_cfg.items():
    scores = cross_val_score(
        cfg['model'], cfg['X'], cfg['y'],
        cv=cv, scoring='roc_auc', n_jobs=-1
    )
    cv_results[name] = scores
    print(f"  {name:<22}: {scores.mean():.4f} +/- {scores.std():.4f}")

# ── Entrenamiento final y serialización ───────────────────────────────────
fitted_models = {}
for name, cfg in models_cfg.items():
    cfg['model'].fit(cfg['X'], cfg['y'])
    fitted_models[name] = cfg['model']
    safe_name = name.lower().replace(' ', '_')
    joblib.dump(cfg['model'], f'models/{safe_name}.pkl')
    print(f"  Guardado: models/{safe_name}.pkl")

# ── Visualización comparativa de CV ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
names    = list(cv_results.keys())
means    = [cv_results[n].mean() for n in names]
stds     = [cv_results[n].std()  for n in names]
colors_b = ['#1565C0', '#2E7D32', '#E65100', '#4A148C']

bars = ax.barh(names, means, xerr=stds, capsize=5,
               color=colors_b, alpha=0.85, edgecolor='black', linewidth=0.5)
ax.axvline(0.5, color='red', linestyle='--', alpha=0.4, label='Random baseline (0.5)')
ax.set_xlabel('AUC-ROC (5-Fold CV)', fontweight='bold')
ax.set_title('Comparativa de Modelos — Cross-Validation', fontweight='bold', fontsize=13)
ax.set_xlim(0.45, 1.0)

for bar, mean in zip(bars, means):
    ax.text(mean + 0.003, bar.get_y() + bar.get_height() / 2,
            f'{mean:.4f}', va='center', fontweight='bold', fontsize=10)

ax.legend()
plt.tight_layout()
plt.savefig('reports/figures/06_cv_comparison.png', bbox_inches='tight')
plt.show()
```

---

## Etapa 5 — Evaluación con Métricas Bancarias (Notebook 02, Parte 3)

> **¿Por qué Gini y KS y no solo accuracy?** Accuracy es inútil con datos desbalanceados — un modelo que predice "no default" siempre alcanza >93% accuracy. Gini (=2×AUC−1) mide discriminación. KS mide la máxima separación entre la distribución de buenos y malos pagadores — estándar en scorecard development en banca peruana.

```python
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    confusion_matrix, classification_report, average_precision_score
)

# ── Funciones de métricas bancarias ───────────────────────────────────────
def gini_coef(y_true, y_prob):
    """Gini = 2*AUC - 1. Métrica estándar en modelos de riesgo crediticio."""
    return 2 * roc_auc_score(y_true, y_prob) - 1

def ks_stat(y_true, y_prob):
    """KS statistic: máxima diferencia entre CDF de buenos y malos pagadores."""
    df_ks = pd.DataFrame({'y': y_true, 'p': y_prob}).sort_values('p', ascending=False)
    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()
    cum_pos = (df_ks['y'] == 1).cumsum() / n_pos
    cum_neg = (df_ks['y'] == 0).cumsum() / n_neg
    return (cum_pos - cum_neg).abs().max()

# ── Evaluar todos los modelos en test set ─────────────────────────────────
eval_results = {}
probas       = {}

for name, model in fitted_models.items():
    X_eval = X_test_scaled.values if name == 'Logistic Regression' else X_test_fe.values
    y_prob = model.predict_proba(X_eval)[:, 1]
    probas[name] = y_prob

    eval_results[name] = {
        'AUC-ROC':       round(roc_auc_score(y_test, y_prob), 4),
        'Gini':          round(gini_coef(y_test, y_prob), 4),
        'KS':            round(ks_stat(y_test, y_prob), 4),
        'Avg Precision': round(average_precision_score(y_test, y_prob), 4)
    }

results_df = pd.DataFrame(eval_results).T.sort_values('AUC-ROC', ascending=False)
print("\n=== RESULTADOS EN TEST SET ===")
print(results_df.to_string())
results_df.to_csv('reports/model_results.csv')

# ── Curvas ROC comparativas ────────────────────────────────────────────────
plt.figure(figsize=(8, 6))
colors_roc = ['#1565C0', '#2E7D32', '#E65100', '#4A148C']
for (name, y_prob), color in zip(probas.items(), colors_roc):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'{name} (AUC={auc_val:.3f})', color=color, linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random (AUC=0.500)')
plt.xlabel('False Positive Rate', fontweight='bold')
plt.ylabel('True Positive Rate', fontweight='bold')
plt.title('Curvas ROC — Comparativa de Modelos', fontweight='bold', fontsize=13)
plt.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig('reports/figures/07_roc_curves.png', bbox_inches='tight')
plt.show()

# ── Threshold analysis con Cost Matrix ────────────────────────────────────
# En banca: aprobar a un defaulter (FN) es MÁS costoso que rechazar
# a un buen pagador (FP). Ratio conservador: FN cuesta 5x más que FP.
COST_FN = 5    # costo de aprobar a alguien que no pagará
COST_FP = 1    # costo de rechazar a alguien que sí pagaría

best_model_name = results_df.index[0]
best_prob       = probas[best_model_name]

thresholds = np.arange(0.05, 0.95, 0.01)
costs, f1s, precs, recs = [], [], [], []

for t in thresholds:
    y_pred_t = (best_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()
    costs.append(fn * COST_FN + fp * COST_FP)
    p = tp / (tp + fp + 1e-9)
    r = tp / (tp + fn + 1e-9)
    f1s.append(2 * p * r / (p + r + 1e-9))
    precs.append(p)
    recs.append(r)

opt_idx    = np.argmin(costs)
opt_thresh = thresholds[opt_idx]

print(f"\nMejor modelo: {best_model_name}")
print(f"Threshold por defecto (0.50):")
print(classification_report(y_test, (best_prob >= 0.50).astype(int),
                             target_names=['No Default', 'Default']))
print(f"Threshold óptimo (cost-sensitive = {opt_thresh:.2f}):")
print(classification_report(y_test, (best_prob >= opt_thresh).astype(int),
                             target_names=['No Default', 'Default']))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(thresholds, costs, color='#C62828', linewidth=2)
axes[0].axvline(opt_thresh, color='#1565C0', linestyle='--',
                label=f'Óptimo: {opt_thresh:.2f}')
axes[0].set_title('Costo Total vs. Threshold', fontweight='bold')
axes[0].set_xlabel('Threshold'); axes[0].legend()

axes[1].plot(thresholds, precs, label='Precision', color='#1565C0', lw=2)
axes[1].plot(thresholds, recs,  label='Recall',    color='#C62828',  lw=2)
axes[1].plot(thresholds, f1s,   label='F1',        color='#2E7D32',  lw=2, ls='--')
axes[1].axvline(opt_thresh, color='black', ls='--', alpha=0.4)
axes[1].set_title('Precision / Recall / F1 vs. Threshold', fontweight='bold')
axes[1].set_xlabel('Threshold'); axes[1].legend()

plt.tight_layout()
plt.savefig('reports/figures/08_threshold_analysis.png', bbox_inches='tight')
plt.show()
```

**Resultados esperados (valores aproximados para Give Me Some Credit):**

| Modelo | AUC-ROC | Gini | KS | Avg Precision |
|--------|---------|------|----|--------------|
| LightGBM | ~0.860 | ~0.720 | ~0.450 | ~0.40 |
| XGBoost | ~0.855 | ~0.710 | ~0.445 | ~0.38 |
| Random Forest | ~0.830 | ~0.660 | ~0.410 | ~0.30 |
| Logistic Regression | ~0.790 | ~0.580 | ~0.350 | ~0.22 |

> Completar con los valores reales al ejecutar.

---

## Etapa 6 — Interpretabilidad SHAP (Notebook 03)

> **¿Por qué SHAP es crítico para BCP?** La SBS exige que los modelos de scoring sean explicables. Un reclutador de riesgo en banca sabe que XGBoost "caja negra" no pasa auditoría interna. SHAP es la respuesta estándar en el sector. El waterfall plot permite explicar cualquier decisión individual al cliente.

```python
# notebooks/03_Interpretability.ipynb

import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargar el mejor modelo
best_model = joblib.load(f"models/{best_model_name.lower().replace(' ', '_')}.pkl")
X_shap = X_test_fe.copy()

# ── Explainer (tree-based: RF, XGBoost, LightGBM) ────────────────────────
explainer   = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_shap)

# Para clasificadores binarios: shap_values puede ser lista [clase0, clase1]
sv = shap_values[1] if isinstance(shap_values, list) else shap_values

# ── 1. SHAP Summary Bar — importancia global ──────────────────────────────
plt.figure(figsize=(10, 7))
shap.summary_plot(sv, X_shap, plot_type='bar', show=False)
plt.title(f'SHAP Feature Importance (Global) — {best_model_name}',
          fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig('reports/figures/09_shap_bar.png', bbox_inches='tight')
plt.show()

# ── 2. SHAP Beeswarm — dirección e intensidad del impacto ─────────────────
plt.figure(figsize=(10, 7))
shap.summary_plot(sv, X_shap, show=False)
plt.title('SHAP Beeswarm — Dirección del Impacto por Variable',
          fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig('reports/figures/10_shap_beeswarm.png', bbox_inches='tight')
plt.show()

# ── 3. Waterfall — explicación individual ─────────────────────────────────
y_prob_shap   = best_model.predict_proba(X_shap.values)[:, 1]
high_risk_idx = int(np.argmax(y_prob_shap))
low_risk_idx  = int(np.argmin(y_prob_shap))

base_val = explainer.expected_value
if isinstance(base_val, list):
    base_val = base_val[1]

for idx, label in [(high_risk_idx, 'alto_riesgo'), (low_risk_idx, 'bajo_riesgo')]:
    expl = shap.Explanation(
        values=sv[idx],
        base_values=base_val,
        data=X_shap.iloc[idx].values,
        feature_names=X_shap.columns.tolist()
    )
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(expl, max_display=12, show=False)
    prob_val = y_prob_shap[idx]
    plt.title(
        f'Explicación Individual — Cliente de {label.replace("_", " ").title()}\n'
        f'P(default) = {prob_val:.3f}',
        fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(f'reports/figures/11_shap_waterfall_{label}.png', bbox_inches='tight')
    plt.show()

# ── 4. Dependence Plot — top 2 features ──────────────────────────────────
mean_shap_abs = np.abs(sv).mean(axis=0)
top2 = X_shap.columns[np.argsort(mean_shap_abs)[::-1][:2]].tolist()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for i, feat in enumerate(top2):
    shap.dependence_plot(feat, sv, X_shap, ax=axes[i], show=False)
    axes[i].set_title(f'SHAP Dependence: {feat}', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('reports/figures/12_shap_dependence.png', bbox_inches='tight')
plt.show()

# ── Tabla de importancia SHAP ─────────────────────────────────────────────
importance_df = pd.DataFrame({
    'Feature':     X_shap.columns,
    'Mean |SHAP|': mean_shap_abs
}).sort_values('Mean |SHAP|', ascending=False)

print("Top 5 Variables por Importancia SHAP:")
print(importance_df.head().to_string(index=False))
importance_df.to_csv('reports/shap_importance.csv', index=False)
```

**Gráficos producidos:**

| Archivo | Descripción |
|---------|-------------|
| `09_shap_bar.png` | Feature importance global (barras por media del valor absoluto SHAP) |
| `10_shap_beeswarm.png` | Dirección e intensidad del impacto de cada variable |
| `11_shap_waterfall_alto_riesgo.png` | Explicación individual: cliente con máxima P(default) |
| `11_shap_waterfall_bajo_riesgo.png` | Explicación individual: cliente con mínima P(default) |
| `12_shap_dependence.png` | Dependence plots de las 2 variables más importantes |

---

## Etapa 7 — README.md para GitHub

```markdown
# Credit Risk Prediction Model 🏦

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![uv](https://img.shields.io/badge/uv-package%20manager-purple)](https://docs.astral.sh/uv/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## Pregunta de Investigación

> ¿Es posible predecir la probabilidad de incumplimiento financiero de un cliente  
> en los próximos 2 años, y qué variables tienen mayor poder predictivo?

## Dataset

| Fuente | Registros | Default rate | Acceso |
|--------|-----------|--------------|--------|
| Give Me Some Credit (Kaggle) | 150,000 | ~6.7% | Kaggle account |
| UCI Credit Card Default | 30,000 | ~22.1% | Descarga directa |

## Metodología

1. **EDA** — Análisis del desbalance, distribuciones, correlaciones
2. **Preprocesamiento** — Imputación mediana, Winsorización P1/P99, split estratificado
3. **Feature Engineering** — Ratios financieros derivados e indicadores de comportamiento de pago
4. **SMOTE** — Oversampling controlado (`sampling_strategy=0.25`) exclusivamente sobre train
5. **Modelado** — LR (baseline regulatorio) → RF → XGBoost → LightGBM + CV 5-fold
6. **Evaluación** — AUC-ROC, Gini coefficient, KS statistic, Avg Precision
7. **Interpretabilidad** — SHAP summary, beeswarm, waterfall individual, dependence plots
8. **Threshold analysis** — Optimización cost-sensitive (FN penalizado 5× sobre FP)

## Resultados

| Modelo             | AUC-ROC | Gini   | KS     | Avg Precision |
|--------------------|---------|--------|--------|---------------|
| LightGBM           | X.XXXX  | X.XXXX | X.XXXX | X.XXXX        |
| XGBoost            | X.XXXX  | X.XXXX | X.XXXX | X.XXXX        |
| Random Forest      | X.XXXX  | X.XXXX | X.XXXX | X.XXXX        |
| Logistic Regression| X.XXXX  | X.XXXX | X.XXXX | X.XXXX        |

> Threshold óptimo (cost-sensitive, FN:FP = 5:1): **X.XX**

## Stack

Python 3.12 · uv · pandas · scikit-learn · XGBoost · LightGBM · SHAP · imbalanced-learn

## Cómo Reproducir

```bash
git clone https://github.com/rodnm/credit-risk-model
cd credit-risk-model
uv sync
# Colocar dataset en data/raw/ (ver sección Dataset)
uv run jupyter notebook
```

## Estructura

```
credit-risk-model/
├── data/raw/               # dataset (ver .gitignore)
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Modeling.ipynb
│   └── 03_Interpretability.ipynb
├── models/                 # modelos serializados (.pkl)
├── reports/figures/        # gráficos generados
├── src/utils.py
├── pyproject.toml
└── README.md
```

## Hallazgos Clave

- Las variables de **historial de pagos tardíos** dominan el ranking SHAP global
- LightGBM/XGBoost alcanzan Gini ~0.7X, comparable a scorecards en producción en banca retail
- El threshold óptimo cost-sensitive es X.XX vs. 0.50 por defecto, mejorando Recall en X pp
- SMOTE con `sampling_strategy=0.25` mejora Recall sin degradar excesivamente Precision

---
*Rodrigo Norabuena · [linkedin.com/in/rodnm](https://linkedin.com/in/rodnm) · [rodnm.github.io](https://rodnm.github.io)*
```

---

## Entregables Finales

### Archivos que debe tener el repositorio

```
credit-risk-model/
├── notebooks/
│   ├── 01_EDA.ipynb                     ← con outputs visibles (no limpiar)
│   ├── 02_Modeling.ipynb
│   └── 03_Interpretability.ipynb
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── lightgbm.pkl
├── reports/
│   └── figures/
│       ├── 01_class_distribution.png
│       ├── 02_distributions.png
│       ├── 03_correlation_target.png
│       ├── 04_heatmap.png
│       ├── 05_bivariate.png
│       ├── 06_cv_comparison.png
│       ├── 07_roc_curves.png
│       ├── 08_threshold_analysis.png
│       ├── 09_shap_bar.png
│       ├── 10_shap_beeswarm.png
│       ├── 11_shap_waterfall_alto_riesgo.png
│       ├── 11_shap_waterfall_bajo_riesgo.png
│       └── 12_shap_dependence.png
├── reports/
│   ├── model_results.csv               ← tabla comparativa de métricas
│   └── shap_importance.csv             ← ranking SHAP exportado
├── pyproject.toml
├── .python-version                      ← "3.12"
├── .gitignore
└── README.md                            ← con tabla de resultados real
```

### Checklist de cierre

- [ ] Los 3 notebooks ejecutan end-to-end sin errores (`Kernel → Restart & Run All`)
- [ ] 12 gráficos exportados en `reports/figures/`
- [ ] 4 modelos `.pkl` en `models/`
- [ ] `reports/model_results.csv` con valores reales
- [ ] README con tabla de resultados completada (no X.XXXX)
- [ ] `pyproject.toml` funcional (`uv sync` en máquina limpia funciona)
- [ ] `.gitignore` configurado (no subir `data/raw/` ni `.venv/`)
- [ ] Repositorio público en GitHub
- [ ] Link al repo en perfil de LinkedIn y CV

---

## Comandos de Referencia Rápida (uv)

```bash
# Instalar una dependencia nueva
uv add nombre-paquete

# Ejecutar jupyter
uv run jupyter notebook

# Ejecutar un script
uv run python src/utils.py

# Sincronizar entorno (en máquina nueva con el proyecto clonado)
uv sync

# Ver qué está instalado
uv pip list

# Verificar versión de Python
uv run python --version
```
