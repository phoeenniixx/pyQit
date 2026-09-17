import numpy as np
import pytest

import pyqit
from pyqit.core.pipeline import QuantumPipeline
from pyqit.core.trainer import Trainer
from pyqit.data.datamodule import DataModule
from pyqit.models.layers import DenseClassifier
from pyqit.tests._fixture_generators import BaseFixtureGenerator
from pyqit.tests.scenarios import make_scenario
from pyqit.tests.test_trainer import BACKENDS, _require, _to_numpy


class TestAllLayers(BaseFixtureGenerator):
    object_type_filter = "layer"
    exclude_objects = ["DenseClassifier"]

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_layer_trains_in_front_of_a_classical_head(self, object_instance, backend):
        """A layer emits features, so it is trained through the stage after it."""
        _require(backend)
        pyqit.set_seed(0)
        layer = object_instance.clone()
        n_in = getattr(layer, "n_features", getattr(layer, "n_qubits", None))
        n_out = getattr(layer, "n_out", getattr(layer, "n_qubits", None))
        scenario = make_scenario(n_samples=16, n_features=n_in, seed=0)
        before = {k: np.array(_to_numpy(v)) for k, v in layer.weights.items()}
        pipe = QuantumPipeline([layer, DenseClassifier(n_out)], fit_mode="joint")

        Trainer(max_epochs=1, learning_rate=0.1, verbose=0).fit(
            pipe, DataModule(scenario["X"], scenario["y"], normalize="minmax")
        )

        for key, value in before.items():
            assert not np.allclose(value, _to_numpy(layer.weights[key])), key
