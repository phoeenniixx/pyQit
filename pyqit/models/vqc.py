import inspect

import pennylane as qml
import pennylane.numpy as np

from pyqit.ansatzes.sel import SELAnsatz
from pyqit.core.embeddings import AngleEmbedding
from pyqit.core.measurements import measure_expval_z, measure_probs
from pyqit.models.base.quantum_model import BaseQuantumModel
from pyqit.utils import _is_torch


class VQCClassifier(BaseQuantumModel):
    _tags = {
        "object_type": "model",
        "is_quantum": True,
    }

    def __init__(
        self,
        n_qubits=4,
        n_layers=3,
        ansatz=SELAnsatz,
        encoder=AngleEmbedding,
        n_classes=2,
        measure_fn=None,
        measure_wires=None,
        backend="pennylane",
        device="default.qubit",
    ):
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
        self.backend = backend
        self.device = device

        self._ansatz_name = self.ansatz.__name__
        self._encoder_name = self.encoder.__name__

        self.ansatz_obj = self.ansatz(n_qubits=n_qubits, n_layers=n_layers)
        self.embedding_obj = self.encoder(n_qubits=n_qubits)

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
            ansatz_obj=self.ansatz_obj,
            embedding_obj=self.embedding_obj,
            measure_fn=self._measure_fn,
            measure_wires=self._measure_wires,
            device=device,
        )

    def __repr__(self):
        return (
            f"VQCClassifier(n_qubits={self.n_qubits}, n_layers={self.n_layers}, "
            f"n_classes={self.n_classes}, ansatz={self._ansatz_name}, "
            f"encoder={self._encoder_name}, device='{self.device}')"
        )

    def forward(self, X, *custom_weights):
        current_weights = (
            custom_weights
            if custom_weights
            else [self.weights[k] for k in self.weight_keys]
        )

        raw_output = self.qnode(X, *current_weights)

        if self.n_classes == 2:
            return (raw_output + 1.0) / 2.0
        else:
            if raw_output.ndim == 1:
                class_probs = raw_output[: self.n_classes]
            else:
                class_probs = raw_output[:, : self.n_classes]
            import pennylane.math as qml_math

            return class_probs / qml_math.sum(class_probs, axis=-1, keepdims=True)

    def predict_step(self, X):
        current_weights = [self.weights[k] for k in self.weight_keys]
        raw_output = self.qnode(X, *current_weights)

        is_torch = _is_torch(raw_output)

        if self.n_classes == 2:
            if is_torch:
                return (raw_output >= 0.0).int()
            else:
                return (raw_output >= 0.0).astype(int)
        else:
            class_raw = raw_output[: self.n_classes]
            if is_torch:
                return class_raw.argmax(dim=0)
            else:
                return class_raw.argmax(axis=0)

    @classmethod
    def get_test_params(cls):
        from pyqit.core.embeddings import AmplitudeEmbedding, IQPEmbedding

        return [
            {},
            {
                "n_qubits": 3,
                "n_layers": 2,
                "n_classes": 2,
                "ansatz": SELAnsatz,
                "encoder": IQPEmbedding,
                "device": "default.qutrit",
            },
            {
                "n_qubits": 4,
                "n_layers": 3,
                "n_classes": 4,
                "ansatz": SELAnsatz,
                "encoder": AmplitudeEmbedding,
                "device": "default.gaussian",
            },
        ]
