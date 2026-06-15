import pennylane as qml

from pyqit.ansatzes.base import BaseAnsatz


class SELAnsatz(BaseAnsatz):
    """
    Strongly Entangling Layers (SEL) ansatz.

    This class implements a parameterized quantum circuit using PennyLane's
    `StronglyEntanglingLayers` template. It applies single-qubit rotations
    and entangling gates across the specified number of layers.

    Parameters
    ----------
    n_qubits : int
        The number of qubits the ansatz acts upon.
    n_layers : int, optional
        The number of entangling layers in the circuit. Default is 2.
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
