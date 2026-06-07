import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3"
}

cells = []

cells.append(nbf.v4.new_markdown_cell("# Credit Risk Prediction - EDA\n\nExploratory Data Analysis for UCI Credit Card Default dataset"))

cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

df = pd.read_excel('../data/raw/default of credit card clients.xls', header=1, engine='xlrd')
df.rename(columns={'default payment next month': 'default'}, inplace=True)
df.drop('ID', axis=1, inplace=True)

TARGET = 'default'
print(f"Shape: {df.shape}")
print(f"Default rate: {df[TARGET].mean():.2%}")
"""))

cells.append(nbf.v4.new_code_cell("""df.info()
df.describe()
"""))

cells.append(nbf.v4.new_markdown_cell("## 1. Target Distribution"))
cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(8, 6))
df[TARGET].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title('Distribution of Default Status')
plt.xlabel('Default (0=No, 1=Yes)')
plt.ylabel('Count')
plt.xticks(rotation=0)
for i, v in enumerate(df[TARGET].value_counts()):
    plt.text(i, v + 100, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('../reports/figures/01_target_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("## 2. Correlation Heatmap"))
cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(14, 10))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', center=0,
            square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('../reports/figures/02_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

print("\\nTop correlations with target:")
print(corr[TARGET].sort_values(ascending=False)[1:11])
"""))

cells.append(nbf.v4.new_markdown_cell("## 3. Distribution of Key Features"))
cells.append(nbf.v4.new_code_cell("""key_features = ['LIMIT_BAL', 'AGE', 'PAY_0', 'BILL_AMT1', 'PAY_AMT1']
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, feat in enumerate(key_features):
    df[df[TARGET]==0][feat].hist(bins=50, alpha=0.6, label='No Default', ax=axes[i], color='green')
    df[df[TARGET]==1][feat].hist(bins=50, alpha=0.6, label='Default', ax=axes[i], color='red')
    axes[i].set_title(feat)
    axes[i].legend()

axes[-1].set_visible(False)
plt.tight_layout()
plt.savefig('../reports/figures/03_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("## 4. Box Plots for Outlier Detection"))
cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, feat in enumerate(key_features):
    sns.boxplot(data=df, x=TARGET, y=feat, ax=axes[i], palette=['green', 'red'])
    axes[i].set_title(f'{feat} by Default Status')

axes[-1].set_visible(False)
plt.tight_layout()
plt.savefig('../reports/figures/04_boxplots.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("## 5. Payment Status Analysis"))
cells.append(nbf.v4.new_code_cell("""pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, col in enumerate(pay_cols):
    default_rates = df.groupby(col)[TARGET].mean()
    default_rates.plot(kind='bar', ax=axes[i], color='steelblue')
    axes[i].set_title(f'Default Rate by {col}')
    axes[i].set_ylabel('Default Rate')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('../reports/figures/05_payment_status.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("## Summary Statistics"))
cells.append(nbf.v4.new_code_cell("""print("=== Summary Statistics by Default Status ===")
print(df.groupby(TARGET).agg({
    'LIMIT_BAL': ['mean', 'median', 'std'],
    'AGE': ['mean', 'median'],
    'PAY_0': ['mean', 'std'],
    'BILL_AMT1': ['mean', 'median'],
    'PAY_AMT1': ['mean', 'median']
}).round(2))
"""))

nb.cells = cells

with open('notebooks/01_EDA.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Created notebooks/01_EDA.ipynb")
