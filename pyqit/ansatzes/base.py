from abc import ABC, abstractmethod
import pennylane as qml

class BaseAnsatz(ABC):
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