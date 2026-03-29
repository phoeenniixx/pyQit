import pennylane as qml
from pennylane import numpy as np
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from pyqit.ansatzes.base import BaseAnsatz
from pyqit.tests._fixture_generators import BaseFixtureGenerator


class TestAllAnsatz(BaseFixtureGenerator):
    object_type_filter = "ansatz"

    def test_weight_shapes_format(self, object_instance):
        shapes = object_instance.get_weight_shapes()

        assert isinstance(shapes, dict), "get_weight_shapes must return a dict"

        for weight_name, shape in shapes.items():
            assert isinstance(weight_name, str), "Weight names must be strings"
            assert isinstance(shape, tuple), f"Shape for '{weight_name}' \
                must be a tuple, got {type(shape)}"
            assert all(
                isinstance(dim, int) and dim > 0 for dim in shape
            ), f"Shape dimensions for '{weight_name}' must \
                    be positive integers, got {shape}"

    def test_build_circuit_execution_autograd(self, object_instance):
        """Verifies circuit builds successfully with PennyLane's default np."""
        shapes = object_instance.get_weight_shapes()
        dummy_weights = {
            name: np.random.uniform(0, 2 * np.pi, size=shape)
            for name, shape in shapes.items()
        }

        dev = qml.device("default.qubit", wires=object_instance.n_qubits)

        @qml.qnode(dev, interface="autograd")
        def dummy_qnode(weights):
            object_instance.build_circuit(weights)
            return qml.expval(qml.PauliZ(0))

        result = dummy_qnode(dummy_weights)

        assert np.isscalar(result.unwrap() if hasattr(result, "unwrap") else result)

    def test_build_circuit_execution_torch(self, object_instance):
        """Verifies circuit builds successfully with PyTorch tensors and interface."""

        if not _check_soft_dependencies("torch", severity="none"):
            pytest.skip("PyTorch is not installed. Skipping torch execution test.")

        import torch

        shapes = object_instance.get_weight_shapes()
        dummy_weights = {
            name: torch.rand(shape, dtype=torch.float64, requires_grad=True)
            for name, shape in shapes.items()
        }

        dev = qml.device("default.qubit", wires=object_instance.n_qubits)

        @qml.qnode(dev, interface="torch")
        def dummy_qnode(weights):
            object_instance.build_circuit(weights)
            return qml.expval(qml.PauliZ(0))

        result = dummy_qnode(dummy_weights)

        assert isinstance(result, torch.Tensor), "Output must be a torch.Tensor"
        assert result.requires_grad, "The computational graph was broken"
