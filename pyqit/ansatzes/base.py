from abc import abstractmethod

from pyqit.base.base_object import _PyQitObject


class BaseAnsatz(_PyQitObject):
    _tags = {
        "object_type": "ansatz",
        "ansatz_type": None,
        "n_qubits_min": 1,
        "differentiable": None,
    }

    def __init__(self, n_qubits: int, n_layers: int = 1):
        """
        Base configuration for any Quantum Ansatz.

        Parameters
        ----------
            n_qubits: int
                The number of wires in the circuit.
            n_layers: int
                The depth of the circuit (repeating blocks).
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        super().__init__()

    @abstractmethod
    def build_circuit(self, weights):
        """
        The actual PennyLane circuit logic.

        Parameters
        ----------
            weights:
                Trainable parameters.
        """
        pass

    @abstractmethod
    def get_weight_shapes(self) -> dict:
        """
        Returns the shape of trainable weights required by this ansatz.
        Used by wrappers to initialize parameters.

        Returns
        -------
            dict:
                e.g., {"weights": (n_layers, n_qubits, 3)}
        """
        pass

    def get_circuit_func(self):
        """
        Returns the bound method to be passed to a QNode.
        """
        return self.build_circuit
