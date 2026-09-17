import pennylane as qml

from pyqit.ansatzes.base import BaseAnsatz


class CNOTLadderAnsatz(BaseAnsatz):
    """The variational block of Mari et al. (2020): a CNOT ladder, then RY.

    Each layer applies CNOT to the wire pairs ``(0, 1), (2, 3), ...``, then to
    ``(1, 2), (3, 4), ...``, then one RY per wire.

    Parameters
    ----------
    n_qubits : int
    n_layers : int, default 6
        ``q_depth`` in the paper.

    References
    ----------
    Mari, Bromley, Izaac, Schuld, Killoran, "Transfer learning in hybrid
    classical-quantum neural networks", Quantum 4, 340 (2020). PennyLane's
    "Quantum transfer learning" demo is the reference implementation.
    """

    def __init__(self, n_qubits: int, n_layers: int = 6):
        super().__init__(n_qubits, n_layers)

    def build_circuit(self, weights):
        """Apply the layers. Expects `weights["weights"]` of shape
        `(n_layers, n_qubits)`."""
        for layer in range(self.n_layers):
            for start in (0, 1):
                for i in range(start, self.n_qubits - 1, 2):
                    qml.CNOT(wires=[i, i + 1])
            for w in range(self.n_qubits):
                qml.RY(weights["weights"][layer, w], wires=w)

    def get_weight_shapes(self) -> dict:
        """Return `{"weights": (n_layers, n_qubits)}`."""
        return {"weights": (self.n_layers, self.n_qubits)}

    @classmethod
    def get_test_params(cls):
        """List constructor kwargs used to parametrize this class in the test suite."""
        return [{"n_qubits": 3, "n_layers": 2}, {"n_qubits": 2}]
