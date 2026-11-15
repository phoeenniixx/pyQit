import numpy as np

from pyqit.tests.scenarios.base import BaseScenario, all_scenarios
from pyqit.tests.scenarios.binary import AdHocScenario
from pyqit.tests.scenarios.multiclass import BlobsScenario
from pyqit.tests.scenarios.regression import LinearRegressionScenario

__all__ = ["BaseScenario", "all_scenarios", "make_scenario"]


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


def make_scenario(
    n_samples: int = 30,
    n_features: int = 4,
    n_classes: int | None = 2,
    seed: int = 0,
) -> dict:
    if n_classes is None:
        cls = LinearRegressionScenario
    elif n_classes == 2:
        cls = AdHocScenario
    else:
        cls = BlobsScenario
    return cls(n_samples, n_features, n_classes, seed).load()
