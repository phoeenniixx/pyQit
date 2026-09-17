import pennylane as qml

from pyqit.ansatzes.base import BaseAnsatz


class SimplifiedTwoDesignAnsatz(BaseAnsatz):
    """Simplified two-design ansatz of Cerezo et al. 2021 (Nat. Commun.).

    Wraps PennyLane's `SimplifiedTwoDesign`: an initial RY layer, then
    `n_layers` of controlled-Z gates on alternating pairs each followed by RY
    rotations. This is the circuit the local-cost trainability result was
    proved on, so it pairs with the barren-plateau diagnostic.

    There are two weight tensors, `initial_layer_weights` of shape
    `(n_qubits,)` and `weights` of shape `(n_layers, n_qubits - 1, 2)`.

    Parameters
    ----------
    n_qubits : int
        At least 2.
    n_layers : int, default 2

    References
    ----------
    Cerezo, Sone, Volkoff, Cincio, Coles, "Cost function dependent barren
    plateaus in shallow parametrized quantum circuits", Nat. Commun. 12, 1791
    (2021).

    Examples
    --------
    >>> from pyqit.ansatzes import SimplifiedTwoDesignAnsatz
    >>> SimplifiedTwoDesignAnsatz(n_qubits=3, n_layers=2).get_weight_shapes()
    {'initial_layer_weights': (3,), 'weights': (2, 2, 2)}
    """

    _tags = {"n_qubits_min": 2}

    def __init__(self, n_qubits: int, n_layers: int = 2):
        super().__init__(n_qubits, n_layers)

    def build_circuit(self, weights):
        """Apply the layers. Expects `weights["initial_layer_weights"]` of shape
        `(n_qubits,)` and `weights["weights"]` of shape `(n_layers, n_qubits - 1, 2)`.
        """
        qml.SimplifiedTwoDesign(
            weights["initial_layer_weights"],
            weights["weights"],
            wires=range(self.n_qubits),
        )

    def get_weight_shapes(self) -> dict:
        """Return the two weight shapes, `initial_layer_weights` and `weights`."""
        initial, layers = qml.SimplifiedTwoDesign.shape(self.n_layers, self.n_qubits)
        return {"initial_layer_weights": initial, "weights": layers}

    @classmethod
    def get_test_params(cls):
        """List constructor kwargs used to parametrize this class in the test suite."""
        return [{"n_qubits": 3, "n_layers": 2}, {"n_qubits": 2, "n_layers": 1}]
