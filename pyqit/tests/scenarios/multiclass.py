import numpy as np
from sklearn.datasets import make_blobs

from pyqit import datasets
from pyqit.tests.scenarios.base import BaseScenario


class BlobsScenario(BaseScenario):
    _tags = {"n_classes": tuple(range(3, 11))}

    def _load(self):
        X, y = make_blobs(
            n_samples=self.n_samples,
            n_features=self.n_features,
            centers=self.n_classes,
            cluster_std=1.0,
            random_state=self.seed,
        )
        span = X.max(axis=0) - X.min(axis=0)
        return (X - X.min(axis=0)) / np.where(span == 0, 1.0, span), y


class IrisScenario(BaseScenario):
    _tags = {"n_classes": (3,), "n_features": (4,)}

    def _load(self):
        X, y = datasets.load_iris()
        rows = np.random.RandomState(self.seed).permutation(len(y))[: self.n_samples]
        return X[rows], y[rows]
