import pennylane as qml

from pyqit.ansatzes.base import BaseAnsatz


class BasicEntanglerAnsatz(BaseAnsatz):
    """Basic entangler layers: one rotation per qubit and a CNOT ring per layer.

    Wraps PennyLane's `BasicEntanglerLayers` (Schuld et al. 2020 lineage).

    Parameters
    ----------
    n_qubits : int
    n_layers : int, default 2
    rotation : type, optional
        Single-qubit rotation gate class. PennyLane's default is `qml.RX`.
    """

    def __init__(self, n_qubits: int, n_layers: int = 2, rotation=None):
        self.rotation = rotation
        super().__init__(n_qubits, n_layers)

    def build_circuit(self, weights):
        """Apply the layers. Expects `weights["weights"]` of shape
        `(n_layers, n_qubits)`."""
        qml.BasicEntanglerLayers(
            weights["weights"], wires=range(self.n_qubits), rotation=self.rotation
        )

    def get_weight_shapes(self) -> dict:
        """Return `{"weights": (n_layers, n_qubits)}`."""
        return {"weights": qml.BasicEntanglerLayers.shape(self.n_layers, self.n_qubits)}

    @classmethod
    def get_test_params(cls):
        """List constructor kwargs used to parametrize this class in the test suite."""
        return [{"n_qubits": 3, "n_layers": 2}, {"n_qubits": 2, "rotation": qml.RY}]
