from abc import abstractmethod

import pennylane as qml
import pennylane.numpy as pnp
from skbase.utils.dependencies import _check_soft_dependencies

from pyqit.core.embeddings import AngleEmbedding
from pyqit.core.measurements import measure_expval_z
from pyqit.models.base.base import BaseModel


class BaseQuantumModel(BaseModel):
    def __init__(
        self,
        ansatz_obj,
        embedding_obj=None,
        measure_fn=None,
        measure_wires=None,
        device="default.qubit",
        shots=None,
        backend="pennylane",
    ):
        self.ansatz_obj = ansatz_obj
        self.embedding_obj = embedding_obj
        self.measure_wires = measure_wires
        self.measure_fn = measure_fn
        self.device = device
        self.shots = shots
        self.backend = backend

        self._device_obj = qml.device(
            self.device, wires=ansatz_obj.n_qubits, shots=self.shots
        )

        self._embedding_obj = (
            self.embedding_obj
            if self.embedding_obj
            else AngleEmbedding(ansatz_obj.n_qubits)
        )
        self._measure_wires = self.measure_wires or [0]
        self._measure_fn = self.measure_fn or measure_expval_z

        self.shapes = ansatz_obj.get_weight_shapes()
        self.weight_keys = list(self.shapes.keys())

        self._compile_backend(self.backend)

    def _compile_backend(self, backend: str):
        self.backend = backend

        interface = "torch" if backend == "torch" else "autograd"
        self.qnode = qml.QNode(self._circuit, self._device_obj, interface=interface)

        self.weights = self._init_weights(self.shapes)

    def _init_weights(self, shapes):
        if getattr(
            self, "backend", "pennylane"
        ) == "torch" and _check_soft_dependencies(["torch"], severity="error"):
            import torch

            return {
                name: torch.rand(shape, dtype=torch.float64, requires_grad=True)
                for name, shape in shapes.items()
            }
        else:
            return {
                name: pnp.random.uniform(0, 2 * pnp.pi, size=shape, requires_grad=True)
                for name, shape in shapes.items()
            }

    def _circuit(self, inputs, *flat_weights):
        weights_dict = dict(zip(self.weight_keys, flat_weights))
        self._embedding_obj.forward(inputs)
        self.ansatz_obj.build_circuit(weights_dict)
        return self._measure_fn(self._measure_wires)

    @abstractmethod
    def forward(self, X):
        pass

    @abstractmethod
    def predict_step(self, X):
        pass

    def update_weights(self, new_weights):
        self.weights = new_weights

    def __call__(self, X):
        return self.forward(X)
