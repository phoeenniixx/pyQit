import pennylane as qml

from pyqit.ansatzes.base import BaseAnsatz


class SELAnsatz(BaseAnsatz):
    """Strongly entangling layers of Schuld et al. (2020).

    Wraps PennyLane's `StronglyEntanglingLayers`. Each layer applies three
    rotations to every qubit, then a CNOT layer whose range grows with the
    layer index. The weights are one tensor, `weights`, of shape
    `(n_layers, n_qubits, 3)`.

    Parameters
    ----------
    n_qubits : int
    n_layers : int, default 2

    References
    ----------
    Schuld, Bocharov, Svore, Wiebe, "Circuit-centric quantum classifiers",
    Phys. Rev. A 101, 032308 (2020).

    Examples
    --------
    >>> from pyqit.ansatzes import SELAnsatz
    >>> from pyqit.models import VQCClassifier
    >>> model = VQCClassifier(n_qubits=4, n_layers=3, ansatz=SELAnsatz)
    """

    def __init__(self, n_qubits: int, n_layers: int = 2):
        super().__init__(n_qubits, n_layers)

    def build_circuit(self, weights):
        """
        Construct and apply the strongly entangling layers to the quantum circuit.

        Parameters
        ----------
        weights : dict
            A dictionary containing the parameter tensors. Must include the
            key `"weights"` with a tensor of shape `(n_layers, n_qubits, 3)`.

        """
        w_tensor = weights["weights"]
        qml.templates.StronglyEntanglingLayers(w_tensor, wires=range(self.n_qubits))

    def get_weight_shapes(self) -> dict:
        """
        Get the shapes of the trainable weights required by the ansatz.

        Returns
        -------
        dict
            A dictionary mapping the weight parameter name (`"weights"`) to
            its expected shape tuple `(n_layers, n_qubits, 3)`.
        """
        shape = (self.n_layers, self.n_qubits, 3)
        return {"weights": shape}

    @classmethod
    def get_test_params(cls):
        """
        Retrieve a set of default parameters for testing the ansatz.

        Returns
        -------
        list of dict
            A list containing a dictionary of valid initialization parameters
            for the class.
        """
        return [{"n_qubits": 3, "n_layers": 2}]
