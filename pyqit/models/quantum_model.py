import pennylane as qml
import pennylane.numpy as np
from pyqit.base import BaseModel
from pyqit.embeddings import AngleEmbedding
from pyqit.measurements import measure_expval_z

class QuantumModel(BaseModel):
    def __init__(self, ansatz, embedding=None, measure_fn=None,
                 measure_wires=None, device="default.qubit", shots=None):
        self.ansatz = ansatz
        self.device = qml.device(device, wires=ansatz.n_qubits, shots=shots)
        self.embedding = embedding if embedding else AngleEmbedding(ansatz.n_qubits)
        self.measure_wires = measure_wires or [0]
        self.measure_fn = measure_fn or measure_expval_z
        self.shapes = ansatz.get_weight_shapes()
        self.weight_keys = list(self.shapes.keys())
        self.weights = self._init_weights(self.shapes)
        self.qnode = qml.QNode(self._circuit, self.device, interface='autograd')

    def _init_weights(self, shapes):
        return {name: np.random.random(shape, requires_grad=True) 
                for name, shape in shapes.items()}

    def _circuit(self, inputs, *flat_weights):
        weights_dict = dict(zip(self.weight_keys, flat_weights))
        self.embedding.forward(inputs)
        self.ansatz.build_circuit(weights_dict)
        return self.measure_fn(self.measure_wires)

    def forward(self, X):
        flat = [self.weights[k] for k in self.weight_keys]
        X = np.array(X, requires_grad=False)
        if X.ndim == 1:
            return float(self.qnode(X, *flat))
        return np.array([float(self.qnode(np.atleast_1d(x), *flat)) for x in X])

    def forward_from_tensors(self, inputs, *weight_tensors):
        return self.qnode(inputs, *weight_tensors)

    def fit(self, X, y, trainer=None, **kwargs):
        if trainer:
            trainer.fit(self, X, y, **kwargs)
        return self

    def update_weights(self, new_weights):
        self.weights = new_weights

    def __call__(self, X):
        return self.forward(X)