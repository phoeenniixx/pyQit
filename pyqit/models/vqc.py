from pyqit.core.quantum_model import QuantumModel
from pyqit.ansatzes.sel import SELAnsatz
from pyqit.embeddings import AngleEmbedding
from pyqit.measurements import measure_expval_z, measure_probs
import pennylane as qml
import pennylane.numpy as np
import inspect

class VQCClassifier(QuantumModel):
    def __init__(self, 
                 n_qubits=4, 
                 n_layers=3, 
                 ansatz=SELAnsatz,  
                 encoder=AngleEmbedding, 
                 n_classes=2,
                 measure_fn=None,
                 measure_wires=None,
                 device="default.qubit"):
        
        if not inspect.isclass(ansatz):
            raise TypeError(
                f"'ansatz' must be a class (e.g., SELAnsatz), "
                f"got {type(ansatz).__name__}"
            )
        
        if not inspect.isclass(encoder):
            raise TypeError(
                f"'encoder' must be a class (e.g., AngleEmbedding), "
                f"got {type(encoder).__name__}"
            )
        
        if n_classes > 2 and n_classes > 2**n_qubits:
            raise ValueError(
                f"Cannot classify {n_classes} classes with {n_qubits} qubits. "
                f"Maximum: {2**n_qubits}. Increase n_qubits or reduce n_classes."
            )

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.ansatz = ansatz
        self.encoder = encoder
        self.n_classes = n_classes  
        self.measure_fn = measure_fn
        self.measure_wires = measure_wires
        self.device = device

        self._ansatz_name = self.ansatz.__name__
        self._encoder_name = self.encoder.__name__
        
        self.ansatz_obj = ansatz(n_qubits=n_qubits, n_layers=n_layers)
        self.embedding_obj = encoder(n_qubits=n_qubits)

        if self.measure_fn is None:
            if n_classes == 2:
                self._measure_fn = measure_expval_z
            else:
                self._measure_fn = measure_probs
        else:
            self._measure_fn = self.measure_fn

        if self.measure_wires is None:
            if n_classes == 2:
                self._measure_wires = [0]
            else:
                self._measure_wires = list(range(n_qubits)) 
        else:
            self._measure_wires = self.measure_wires

        super().__init__(
            ansatz=self.ansatz_obj,
            embedding=self.embedding_obj,
            measure_fn=self._measure_fn,
            measure_wires=self._measure_wires,
            device=device
        )

    def __repr__(self):
        return (
            f"VQCClassifier(n_qubits={self.n_qubits}, n_layers={self.n_layers}, "
            f"n_classes={self.n_classes}, ansatz={self._ansatz_name}, "
            f"encoder={self._encoder_name}, device='{self.device}')"
        )
    
    def __call__(self, inputs, weights=None):
        raw_output = super().__call__(inputs, weights)
        
        if self.n_classes == 2:
            return (raw_output + 1.0) / 2.0
        else:
            class_probs = raw_output[:self.n_classes]
            return class_probs / np.sum(class_probs)