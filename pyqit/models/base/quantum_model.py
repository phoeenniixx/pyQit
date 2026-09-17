from abc import abstractmethod

import pennylane as qml
import pennylane.numpy as pnp
from skbase.utils.dependencies import _check_soft_dependencies

from pyqit.core.config import get_backend
from pyqit.models.base.base import BaseModel
from pyqit.models.layers.dense import dense, init_dense_weights


class BaseQuantumModel(BaseModel):
    """Base class wiring a PennyLane QNode into either backend.

    Subclasses build a QNode and call `register_qnode`; this class handles
    the pennylane/torch fork, and exposes weights as a flat dict keyed
    `"<qnode_name>.<weight_name>"` regardless of backend.
    """

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
        """PennyLane QNode interface for the active backend."""
        return "torch" if self.backend == "torch" else "autograd"

    def init_weights(self, weight_shapes: dict) -> dict:
        """Draw uniform ``[0, 1)`` starting weights from numpy's global RNG.

        The range matches Qiskit ML's default ``initial_point`` and PennyLane's
        template examples; uniform ``[0, 2pi)`` is the Haar-like regime where
        gradients vanish (McClean et al. 2018).

        Call this before building the device: a PennyLane device seeded
        ``"global"`` consumes numpy's RNG at construction by a device-dependent
        amount, so weights drawn after it differ per device for the same seed.

        Parameters
        ----------
        weight_shapes : dict
            Weight name to shape, as returned by an ansatz's
            `get_weight_shapes`.

        Returns
        -------
        dict
        """
        return {
            w: pnp.random.uniform(0, 1, size=s, requires_grad=True)
            for w, s in weight_shapes.items()
        }

    def register_qnode(
        self, name: str, qnode: qml.QNode, weight_shapes: dict, weights=None
    ):
        """Wrap `qnode` for the active backend and store it under `name`.

        Parameters
        ----------
        name : str
            Key under which the node's weights appear in `weights`.
        qnode : qml.QNode
        weight_shapes : dict
            Weight name to shape, as returned by an ansatz's
            `get_weight_shapes`.
        weights : dict, optional
            Starting weights from `init_weights`; drawn here when omitted, so
            both backends start from the same point for the same seed.
        """
        if weights is None:
            weights = self.init_weights(weight_shapes)
        if self.backend == "torch" and _check_soft_dependencies(
            ["torch"], severity="none"
        ):
            import torch

            init = {
                w: torch.tensor(pnp.asarray(v), dtype=torch.get_default_dtype())
                for w, v in weights.items()
            }
            torch_layer = qml.qnn.TorchLayer(qnode, weight_shapes, init_method=init)
            setattr(self, name, torch_layer)
            self._qnodes[name] = torch_layer
        else:
            setattr(self, name, qnode)
            self._qnodes[name] = {"node": qnode, "weights": weights}

    def register_dense(self, name: str, n_in: int, n_out: int, weights=None):
        """Register a classical dense layer ``X @ weight.T + bias`` under `name`.

        Lives in the same registry as the QNodes, so ``weights``,
        ``update_weights``, checkpoints and the flat-kwargs routing cover it
        with no further plumbing. Run it with ``execute_qnode``.

        Parameters
        ----------
        name : str
        n_in, n_out : int
        weights : dict, optional
            ``{"weight", "bias"}`` from `init_dense_weights`; drawn when omitted.
        """
        if weights is None:
            weights = init_dense_weights(n_in, n_out)
        if self.backend == "torch" and _check_soft_dependencies(
            ["torch"], severity="none"
        ):
            import torch

            layer = torch.nn.Linear(n_in, n_out)
            with torch.no_grad():
                for w_name, value in weights.items():
                    getattr(layer, w_name).copy_(torch.as_tensor(pnp.asarray(value)))
            setattr(self, name, layer)
            self._qnodes[name] = layer
        else:
            self._qnodes[name] = {"node": dense, "weights": weights}

    @staticmethod
    def _qnode_of(node):
        """The ``qml.QNode`` behind a registry entry, or None for a classical layer."""
        qnode = node["node"] if isinstance(node, dict) else getattr(node, "qnode", None)
        return qnode if isinstance(qnode, qml.QNode) else None

    def execute_qnode(self, name: str, X, **custom_weights):
        """Run the QNode or dense layer registered under `name` on a batch.

        Parameters
        ----------
        name : str
            Name passed to `register_qnode`.
        X : array-like
        **custom_weights
            Flat `"<name>.<weight>"` overrides; unprefixed keys are ignored.
            Falls back to the model's own weights when empty.

        Returns
        -------
        array-like
        """
        if self.backend == "torch":
            layer = getattr(self, name)
            if self._qnode_of(layer) is None:
                return layer(X.to(next(layer.parameters()).dtype))
            if self.shots is None:
                return layer(X)
            import torch

            return layer(X.to(torch.float64)).to(X.dtype)
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
        """Run the model on a batch and return its raw output."""

    def diff_methods(self, X) -> dict:
        """Differentiation method PennyLane resolves ``"best"`` to, per QNode.

        ``backprop`` and ``adjoint`` are simulator-only; a device with shots or
        real hardware resolves to ``parameter-shift``, which costs two circuit
        executions per parameter for every gradient.

        Parameters
        ----------
        X : array-like
            One prescaled batch; only its shape matters.

        Returns
        -------
        dict
            QNode name to method name.
        """
        from pennylane.workflow import get_best_diff_method

        X = qml.math.asarray(X, like=self.get_interface())
        methods = {}
        for name, node in self._qnodes.items():
            qnode = self._qnode_of(node)
            if qnode is None:
                continue
            prefix = f"{name}."
            weights = {
                k.removeprefix(prefix): v
                for k, v in self.weights.items()
                if k.startswith(prefix)
            }
            methods[name] = get_best_diff_method(qnode)(X, **weights)
        return methods

    @property
    def weights(self):
        """Flat ``{"<qnode_name>.<weight_name>": array}`` dict, both backends."""
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
        """Write `flat_weights_dict` into the model's own weights.

        No-op under torch, where autograd owns the `nn.Parameter` objects directly.

        Parameters
        ----------
        flat_weights_dict : dict
            Keyed like `weights`.
        """
        if self.backend == "torch":
            return

        for flat_key, new_val in flat_weights_dict.items():
            node_name, w_name = flat_key.split(".", 1)
            self._qnodes[node_name]["weights"][w_name] = new_val

    def __call__(self, X):
        """Alias for `forward`."""
        return self.forward(X)
