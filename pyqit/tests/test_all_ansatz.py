import pennylane as qml
from pennylane import numpy as np
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

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


@pytest.mark.parametrize("name", ["real_amplitudes", "efficient_su2"])
@pytest.mark.parametrize("entanglement", ["reverse_linear", "full", "circular"])
@pytest.mark.parametrize("n_qubits, reps, skip_final", [(2, 1, False), (4, 3, True)])
def test_hardware_efficient_ansatz_matches_qiskit_reference_circuit(
    name, entanglement, n_qubits, reps, skip_final
):
    """The wrapped circuit prepares the same state as Qiskit's, wire order included."""
    pytest.importorskip("pennylane_qiskit")
    from qiskit.quantum_info import Statevector

    from pyqit.ansatzes import EfficientSU2Ansatz, RealAmplitudesAnsatz

    cls = RealAmplitudesAnsatz if name == "real_amplitudes" else EfficientSU2Ansatz
    ansatz = cls(n_qubits, reps, entanglement, skip_final)
    theta = np.random.default_rng(n_qubits).uniform(
        0, 2 * np.pi, ansatz.get_weight_shapes()["weights"]
    )
    reference = ansatz.circuit.assign_parameters(theta)
    reference_state = Statevector(reference).reverse_qargs().data

    @qml.qnode(qml.device("default.qubit", wires=n_qubits))
    def state(w):
        ansatz.build_circuit({"weights": w})
        return qml.state()

    fidelity = abs(np.vdot(state(theta), reference_state)) ** 2
    assert fidelity == pytest.approx(1.0, abs=1e-10)
