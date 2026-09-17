from abc import abstractmethod

import pennylane as qml
import pennylane.numpy as pnp
from skbase.utils.dependencies import _check_soft_dependencies

from pyqit.base import _PyQitObject
from pyqit.core.config import get_backend


class BaseModel(_PyQitObject):
    """Base class for all trainable models in PyQit.

    Holds the weight registry: layers registered under a name, run with
    `execute_qnode`, and exposed as a flat dict keyed
    `"<layer_name>.<weight_name>"` on both backends. Classical layers register
    here with `register_dense`; `BaseQuantumModel` adds `register_qnode`.
    """

    _tags = {
        "object_type": "model",
        "is_quantum": True,
        "n_qubits": None,
        "differentiable": True,
        "requires_fit": True,
    }

    def __init__(self):
        self.backend = get_backend()
        self._qnodes = {}

    @abstractmethod
    def forward(self, X):
        """Run the model on a batch and return its raw output."""

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
        from pyqit.models.layers.dense import dense, init_dense_weights

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
            if getattr(self, "shots", None) is None:
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

    def is_fitted(self) -> bool:
        """Whether `Trainer.fit` has trained this model."""
        return getattr(self, "_is_fitted", False)

    def _mark_fitted(self):
        self._is_fitted = True
