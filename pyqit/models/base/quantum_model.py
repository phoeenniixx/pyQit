from abc import abstractmethod

import pennylane as qml
import pennylane.numpy as pnp
from skbase.utils.dependencies import _check_soft_dependencies

from pyqit.core.config import get_backend
from pyqit.models.base.base import BaseModel


class BaseQuantumModel(BaseModel):
    _tags = {
        "object_type": "model",
        "is_quantum": True,
    }

    def __init__(
        self,
        device="default.qubit",
        shots=None,
    ):
        self.device = device
        self.shots = shots
        self.backend = get_backend()

        self._qnodes = {}

    def get_interface(self):
        return "torch" if self.backend == "torch" else "autograd"

    def register_qnode(self, name: str, qnode: qml.QNode, weight_shapes: dict):
        if self.backend == "torch" and _check_soft_dependencies(
            ["torch"], severity="none"
        ):
            import torch

            torch_layer = qml.qnn.TorchLayer(qnode, weight_shapes)
            setattr(self, name, torch_layer)
            self._qnodes[name] = torch_layer
        else:
            setattr(self, name, qnode)
            self._qnodes[name] = {
                "node": qnode,
                "weights": {
                    w: pnp.random.uniform(0, 2 * pnp.pi, size=s, requires_grad=True)
                    for w, s in weight_shapes.items()
                },
            }

    def execute_qnode(self, name: str, X, **custom_weights):
        if self.backend == "torch":
            return getattr(self, name)(X)
        else:
            node_data = self._qnodes[name]
            if custom_weights:
                prefix = f"{name}."
                weights = {
                    k.replace(prefix, ""): v
                    for k, v in custom_weights.items()
                    if k.startswith(prefix)
                }
            else:
                weights = node_data["weights"]
            return node_data["node"](X, **weights)

    @abstractmethod
    def _circuit(self, inputs, *flat_weights):
        pass

    @abstractmethod
    def forward(self, X):
        pass

    @property
    def weights(self):
        flat_weights = {}
        if self.backend == "torch":
            import torch

            for node_name, node in self._qnodes.items():
                if isinstance(node, torch.nn.Module):
                    for w_name, param in node.named_parameters():
                        flat_weights[f"{node_name}.{w_name}"] = param
        else:
            for node_name, data in self._qnodes.items():
                for w_name, w_val in data["weights"].items():
                    flat_weights[f"{node_name}.{w_name}"] = w_val
        return flat_weights

    def update_weights(self, flat_weights_dict):
        if self.backend == "torch":
            return

        for flat_key, new_val in flat_weights_dict.items():
            node_name, w_name = flat_key.split(".", 1)
            self._qnodes[node_name]["weights"][w_name] = new_val

    def __call__(self, X):
        return self.forward(X)
