from sklearn.datasets import make_regression

from pyqit.tests.scenarios.base import BaseScenario


class LinearRegressionScenario(BaseScenario):
    _tags = {"estimator_type": "regressor", "n_classes": None}

    def __init__(self, n_samples=16, n_features=4, n_classes=None, seed=0):
        super().__init__(n_samples, n_features, n_classes, seed)

    def _load(self):
        X, y = make_regression(
            n_samples=self.n_samples,
            n_features=self.n_features,
            noise=0.1,
            random_state=self.seed,
        )
        X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        return X, 2.0 * (y - y.min()) / (y.max() - y.min()) - 1.0
