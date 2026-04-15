import inspect

import pennylane as qml
import pennylane.numpy as np

from pyqit.ansatzes.sel import SELAnsatz
from pyqit.core.config import get_backend
from pyqit.core.embeddings import AngleEmbedding
from pyqit.core.measurements import measure_expval_z, measure_probs
from pyqit.models.base.quantum_model import BaseQuantumModel
from pyqit.models.classification.classifier_mixin import ClassifierMixin
from pyqit.utils import _is_torch


class VQCClassifier(BaseQuantumModel, ClassifierMixin):
    def __init__(
        self,
        n_qubits=4,
        n_layers=3,
        ansatz=SELAnsatz,
        encoder=AngleEmbedding,
        n_classes=2,
        measure_fn=None,
        measure_wires=None,
        device="default.qubit",
        shots=None,
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

        super().__init__(
            device=device,
            shots=shots,
        )

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.ansatz = ansatz
        self.encoder = encoder
        self.n_classes = n_classes
        self.measure_fn = measure_fn
        self.measure_wires = measure_wires
        self.backend = get_backend()
        self.device = device
        self.shots = shots

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

        self.weight_keys = list(self.ansatz_obj.get_weight_shapes().keys())

        dev = qml.device(self.device, wires=self.n_qubits, shots=self.shots)
        primary_qnode = qml.QNode(self._circuit, dev, interface=self.get_interface())

        self.register_qnode(
            "main_circuit", primary_qnode, self.ansatz_obj.get_weight_shapes()
        )

    def __repr__(self):
        return (
            f"VQCClassifier(n_qubits={self.n_qubits}, n_layers={self.n_layers}, "
            f"n_classes={self.n_classes}, ansatz={self._ansatz_name}, "
            f"encoder={self._encoder_name}, device='{self.device}')"
        )

    def _circuit(self, inputs, **weights):
        self.embedding_obj.forward(inputs)
        self.ansatz_obj.build_circuit(weights)
        return self._measure_fn(self._measure_wires)

    def forward(self, X, **custom_weights):
        raw_output = self.execute_qnode("main_circuit", X, **custom_weights)

        if self.n_classes == 2:
            return (raw_output + 1.0) / 2.0
        else:
            if raw_output.ndim == 1:
                class_probs = raw_output[: self.n_classes]
            else:
                class_probs = raw_output[:, : self.n_classes]
            import pennylane.math as qml_math

            return class_probs / qml_math.sum(class_probs, axis=-1, keepdims=True)

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
            },
            {
                "n_qubits": 4,
                "n_layers": 3,
                "n_classes": 4,
                "ansatz": SELAnsatz,
                "encoder": AmplitudeEmbedding,
                "trainer_kwargs": {"loss_fn": "cross_entropy"},
            },
        ]
