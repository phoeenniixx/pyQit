import pennylane as qml
from pennylane import numpy as np
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from pyqit.core.embeddings import BaseEmbedding
from pyqit.tests._fixture_generators import BaseFixtureGenerator
from pyqit.tests.scenarios import _generate_embedding_data


class TestAllEmbeddings(BaseFixtureGenerator):
    """Package-level tests for all ``BaseEmbedding`` subclasses."""

    object_type_filter = "embedding"

    def test_forward_execution_autograd(self, object_instance):
        """Verifies forward queues operations correctly for np data."""

        scenario = _generate_embedding_data(object_instance)
        dev = qml.device("default.qubit", wires=object_instance.n_qubits)

        @qml.qnode(dev, interface="autograd")
        def dummy_qnode(inputs):
            object_instance.forward(inputs)
            return qml.expval(qml.PauliZ(0))

        # Test 1: Single Sample
        res_single = dummy_qnode(scenario["x_single"])

        assert np.isscalar(
            res_single.unwrap() if hasattr(res_single, "unwrap") else res_single
        )

        # Test 2: Batched Data
        res_batch = dummy_qnode(scenario["x_batch"])

        assert len(res_batch) == len(
            scenario["x_batch"]
        ), "Batched output size mismatch"

    def test_forward_execution_torch(self, object_instance):
        """Verifies the embedding integrates cleanly with the PyTorch interface."""

        if not _check_soft_dependencies("torch", severity="none"):
            pytest.skip("PyTorch is not installed. Skipping torch execution test.")

        import torch

        scenario = _generate_embedding_data(object_instance)
        dev = qml.device("default.qubit", wires=object_instance.n_qubits)

        x_single_torch = torch.tensor(scenario["x_single"], dtype=torch.float64)

        @qml.qnode(dev, interface="torch")
        def dummy_qnode(inputs):
            object_instance.forward(inputs)
            return qml.expval(qml.PauliZ(0))

        result = dummy_qnode(x_single_torch)
        assert isinstance(result, torch.Tensor), "Output must be a torch.Tensor"
