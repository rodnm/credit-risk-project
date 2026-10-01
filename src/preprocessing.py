"""Reusable fold-local preprocessing for credit-risk pipelines."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


class Winsorizer(TransformerMixin, BaseEstimator):
    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        if not 0 <= self.lower < self.upper <= 1:
            raise ValueError('Require 0 <= lower < upper <= 1')
        values = check_array(X, dtype=float)
        self.n_features_in_ = values.shape[1]
        self.lower_bounds_, self.upper_bounds_ = np.quantile(values, [self.lower, self.upper], axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self, ['lower_bounds_', 'upper_bounds_'])
        values = check_array(X, dtype=float)
        if values.shape[1] != self.n_features_in_:
            raise ValueError('Feature count differs from fitted data')
        return np.clip(values, self.lower_bounds_, self.upper_bounds_)


def engineer_features(raw):
    """Derive row-local features without learning population statistics."""
    X = raw.copy()
    delays = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    X['avg_delay'] = X[delays].mean(axis=1)
    X['max_delay'] = X[delays].max(axis=1)
    X['delay_std'] = X[delays].std(axis=1)
    X['total_delay_months'] = (X[delays] > 0).sum(axis=1)
    X['avg_bill'] = X[[f'BILL_AMT{i}' for i in range(1, 7)]].mean(axis=1)
    X['bill_trend'] = X['BILL_AMT1'] - X['BILL_AMT6']
    X['avg_pay_amt'] = X[[f'PAY_AMT{i}' for i in range(1, 7)]].mean(axis=1)
    X['pay_ratio'] = X['avg_pay_amt'] / (X['avg_bill'] + 1)
    X['credit_util'] = X['BILL_AMT1'] / (X['LIMIT_BAL'] + 1)
    return X.replace([np.inf, -np.inf], np.nan)


def make_pipeline(classifier):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0, keep_empty_features=True)),
        ('winsorizer', Winsorizer()), ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42, k_neighbors=5)), ('classifier', classifier),
    ])


def transform_for_explanation(pipeline, X):
    """Apply fitted preprocessing, never oversampling or refitting."""
    for name in ('imputer', 'winsorizer', 'scaler'):
        X = pipeline.named_steps[name].transform(X)
    return X
