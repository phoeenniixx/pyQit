from abc import abstractmethod

import pennylane as qml
import pennylane.numpy as pnp
from skbase.utils.dependencies import _check_soft_dependencies

from pyqit.models.base.base import BaseModel


class BaseQuantumModel(BaseModel):
    """Base class wiring a PennyLane QNode into either backend.

    Subclasses build a QNode and call `register_qnode`, which handles the
    pennylane/torch fork. The weight registry itself is `BaseModel`'s.
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
        super().__init__()
        self.device = device
        self.shots = shots

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
