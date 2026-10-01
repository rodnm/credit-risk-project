"""Regression tests for leakage boundaries, inference and saved pipelines."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import joblib
import numpy as np
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from imblearn.over_sampling import SMOTE
from src.preprocessing import Winsorizer, make_pipeline, transform_for_explanation


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_classification(n_samples=120, n_features=4,
            n_informative=3, n_redundant=0, weights=[0.75, 0.25], random_state=8)

    def test_held_out_extremes_do_not_change_bounds(self):
        training = np.arange(100, dtype=float).reshape(50, 2)
        winsorizer = Winsorizer().fit(training)
        expected = np.quantile(training, [.01, .99], axis=0)
        np.testing.assert_allclose(winsorizer.transform([[-1e9, 1e9]]),
                                   [[expected[0, 0], expected[1, 1]]])
        np.testing.assert_allclose(winsorizer.lower_bounds_, expected[0])
        np.testing.assert_allclose(winsorizer.upper_bounds_, expected[1])
        self.assertEqual(clone(winsorizer).get_params(), winsorizer.get_params())

    def test_every_fold_fits_only_training_statistics_and_resamples_only_training(self):
        cv = StratifiedKFold(3, shuffle=True, random_state=42)
        folds = list(cv.split(self.X, self.y))
        calls = []
        original = SMOTE.fit_resample
        def record(sampler, X, y, **kwargs):
            calls.append((np.array(X), np.array(y)))
            return original(sampler, X, y, **kwargs)
        with patch.object(SMOTE, 'fit_resample', record):
            output = cross_validate(make_pipeline(LogisticRegression()), self.X, self.y,
                cv=folds, scoring='roc_auc', return_estimator=True, error_score='raise')
            self.assertEqual(len(calls), len(folds))
            for pipeline, (train, validation), (sampled_input, sampled_y) in zip(output['estimator'], folds, calls):
                expected = np.quantile(self.X[train], [.01, .99], axis=0)
                winsorizer = pipeline.named_steps['winsorizer']
                np.testing.assert_allclose(winsorizer.lower_bounds_, expected[0])
                np.testing.assert_allclose(winsorizer.upper_bounds_, expected[1])
                clipped = np.clip(self.X[train], *expected)
                np.testing.assert_allclose(pipeline.named_steps['scaler'].mean_, clipped.mean(axis=0))
                self.assertEqual(pipeline.named_steps['scaler'].n_samples_seen_, len(train))
                np.testing.assert_array_equal(sampled_y, self.y[train])
                np.testing.assert_allclose(sampled_input, transform_for_explanation(pipeline, self.X[train]))
                self.assertEqual(len(pipeline.predict_proba(self.X[validation])), len(validation))
            self.assertEqual(len(calls), len(folds), 'Prediction must never call SMOTE')

    def test_serialized_pipeline_preserves_missing_value_predictions(self):
        X = self.X.copy()
        X[0, 0] = np.nan
        pipeline = make_pipeline(LogisticRegression()).fit(X, self.y)
        probe = X[:7].copy()
        probe[1, 1] = 1e9
        expected = pipeline.predict_proba(probe)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pipeline.pkl'
            joblib.dump(pipeline, path)
            restored = joblib.load(path)
            np.testing.assert_allclose(restored.predict_proba(probe), expected)
            self.assertEqual(transform_for_explanation(restored, probe).shape, probe.shape)


if __name__ == '__main__':
    unittest.main()
