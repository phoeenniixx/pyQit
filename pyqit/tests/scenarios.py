import numpy as np
from sklearn.datasets import make_blobs, make_classification


def _generate_embedding_data(instance, batch_size=8):
    n_qubits = getattr(instance, "n_qubits", 1)

    if "Amplitude" in instance.__class__.__name__:
        n_features = 2**n_qubits
        is_amplitude = True
    else:
        n_features = n_qubits
        is_amplitude = False

    x_single = np.random.uniform(0.1, 1.0, size=(n_features,))
    x_batch = np.random.uniform(0.1, 1.0, size=(batch_size, n_features))

    if is_amplitude:
        x_single = x_single / np.linalg.norm(x_single)
        x_batch = x_batch / np.linalg.norm(x_batch, axis=1, keepdims=True)

    return {"x_single": x_single, "x_batch": x_batch, "n_features": n_features}


def _make_binary(n_samples=40, n_features=4, seed=42) -> dict:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=seed,
    )
    return {
        "name": "binary_classification",
        "X": X.astype(np.float64),
        "y": y.astype(np.float64),
        "n_classes": 2,
        "n_features": n_features,
    }


def _make_multiclass(n_samples=60, n_features=4, n_classes=3, seed=42) -> dict:
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_classes,
        cluster_std=1.0,
        random_state=seed,
    )
    X = (X - X.min(axis=0)) / np.where(
        X.max(axis=0) - X.min(axis=0) == 0, 1.0, X.max(axis=0) - X.min(axis=0)
    )
    return {
        "name": "multiclass_blobs",
        "X": X.astype(np.float64),
        "y": y.astype(np.float64),
        "n_classes": n_classes,
        "n_features": n_features,
    }


def _make_linearly_separable(n_samples=30, n_features=2, seed=0) -> dict:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_clusters_per_class=1,
        class_sep=2.0,
        random_state=seed,
    )
    return {
        "name": "linear_2d",
        "X": X.astype(np.float64),
        "y": y.astype(np.float64),
        "n_classes": 2,
        "n_features": n_features,
    }


SCENARIOS: list[dict] = [
    _make_binary(),
    _make_multiclass(),
    _make_linearly_separable(),
]


def make_scenario(
    n_samples: int = 30,
    n_features: int = 4,
    n_classes: int = 2,
    seed: int = 0,
) -> dict:
    if n_classes == 2:
        return _make_binary(n_samples=n_samples, n_features=n_features, seed=seed)
    return _make_multiclass(
        n_samples=n_samples, n_features=n_features, n_classes=n_classes, seed=seed
    )
