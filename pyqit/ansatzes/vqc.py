import pennylane as qml
from pyqit.ansatzes.base import BaseAnsatz


class VQCAnatz(BaseAnsatz):
    def __init__(self, n_qubits: int, n_layers: int = 2):
        super().__init__(n_qubits, n_layers)


    def build_circuit(self, weights):
        w_tensor = weights['weights']
        qml.templates.StronglyEntanglingLayers(w_tensor, wires=range(self.n_qubits))
        

    def get_weight_shapes(self) -> dict:
        shape = (self.n_layers, self.n_qubits, 3)
        return {"weights": shape}