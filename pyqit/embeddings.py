from abc import ABC, abstractmethod
import pennylane as qml

class BaseEmbedding(ABC):
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits

    @abstractmethod
    def forward(self, inputs):
        pass


class AngleEmbedding(BaseEmbedding):
    def forward(self, inputs):
        qml.AngleEmbedding(features=inputs, wires=range(self.n_qubits), rotation='X')
