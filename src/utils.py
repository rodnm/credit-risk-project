import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score


def winsorize(df_in, lower_pct=0.01, upper_pct=0.99):
    df_w = df_in.copy()
    for col in df_w.select_dtypes(include=[np.number]).columns:
        lb = df_w[col].quantile(lower_pct)
        ub = df_w[col].quantile(upper_pct)
        df_w[col] = df_w[col].clip(lb, ub)
    return df_w


def gini_coef(y_true, y_prob):
    return 2 * roc_auc_score(y_true, y_prob) - 1


def ks_stat(y_true, y_prob):
    df_ks = pd.DataFrame({'y': y_true, 'p': y_prob}).sort_values('p', ascending=False)
    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()
    cum_pos = (df_ks['y'] == 1).cumsum() / n_pos
    cum_neg = (df_ks['y'] == 0).cumsum() / n_neg
    return (cum_pos - cum_neg).abs().max()


def feature_engineering_kaggle(df):
    df = df.copy()
    late_cols = [
        'NumberOfTime30-59DaysPastDueNotWorse',
        'NumberOfTime60-89DaysPastDueNotWorse',
        'NumberOfTimes90DaysLate'
    ]
    df['total_late_payments'] = df[late_cols].sum(axis=1)
    df['has_severe_delay'] = (df['NumberOfTimes90DaysLate'] > 0).astype(int)
    df['income_per_dependent'] = df['MonthlyIncome'] / (df['NumberOfDependents'] + 1)
    df['high_utilization'] = (df['RevolvingUtilizationOfUnsecuredLines'] > 0.75).astype(int)
    df['age_group'] = pd.cut(
        df['age'], bins=[0, 30, 45, 60, 120], labels=[0, 1, 2, 3]
    ).astype(int)
    return df
