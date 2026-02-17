from pyqit.core.quantum_model import QuantumModel
from pyqit.ansatzes.sel import SELAnsatz
from pyqit.embeddings import AngleEmbedding
from pyqit.measurements import measure_expval_z, measure_probs
import pennylane as qml
import inspect  

class VQCClassifier(QuantumModel):
    def __init__(self, 
                 n_qubits=4, 
                 n_layers=3, 
                 ansatz=SELAnsatz,  
                 encoder=AngleEmbedding, 
                 n_classes=2,
                 measure_fn=measure_expval_z,
                 measure_wires=None,
                 device="default.qubit"):
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.ansatz = ansatz
        self.encoder = encoder
        self.n_classes = n_classes  
        self.measure_fn = measure_fn
        self.measure_wires = measure_wires
        self.device = device

        if not inspect.isclass(ansatz):
            raise TypeError(
                f"The 'ansatz' argument must be a class type (e.g., SELAnsatz), "
                f"but received {type(ansatz).__name__}. "
                f"Did you accidentally instantiate it (e.g. SELAnsatz())?"
            )
        
        self.ansatz_obj = ansatz(n_qubits=n_qubits, n_layers=n_layers)

        if not inspect.isclass(encoder):
            raise TypeError(
                f"The 'encoder' argument must be a class type (e.g., AngleEmbedding), "
                f"but received {type(encoder).__name__}."
            )

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