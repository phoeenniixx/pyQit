import pennylane as qml
from pennylane import numpy as np
from pyqit.embeddings import AngleEmbedding
from pyqit.measurements import measure_expval_z

class QuantumModel:
    def __init__(self, ansatz, embedding=None, measure_fn=None,
                 measure_wires=None, device_name="default.qubit", shots=None):
        """
        Quantum Model wrapper.
        
        Parameters
        ----------
        ansatz : BaseAnsatz
            The blueprint logic (e.g., VQC, QLSTM).
        device_name : str
            The PennyLane device string (e.g., 'lightning.qubit').
        """
        self.ansatz = ansatz
        self.device = qml.device(device_name, wires=ansatz.n_qubits, shots=shots)
        self.embedding = embedding if embedding else AngleEmbedding(ansatz.n_qubits)

        if measure_wires is None:
            self.measure_wires = [0]
        else:
            self.measure_wires = measure_wires

        if measure_fn is None:
            self.measure_fn = measure_expval_z
        else:
            self.measure_fn = measure_fn

        self.shapes = ansatz.get_weight_shapes()
        self.weights = self._init_weights(self.shapes)
        
        self.qnode = qml.QNode(self._circuit_wrapper, self.device)

    def _init_weights(self, shapes):
        """Helper to create random initial weights."""
        weights = {}
        for name, shape in shapes.items():
            weights[name] = np.random.random(shape, requires_grad=True)
        return weights

    def _circuit_wrapper(self, inputs, weights):
        self.embedding.forward(inputs)
        
        self.ansatz.build_circuit(weights)
        return self.measure_fn(self.measure_wires)

    def __call__(self, inputs, weights=None):
        if weights is None:
            weights = self.weights
        return self.qnode(inputs, weights)

    def update_weights(self, new_weights):
        """Updates the internal state."""
        self.weights = new_weights