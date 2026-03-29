from collections.abc import Callable

from pyqit.core.losses import cross_entropy_loss, hinge_loss, mse_loss

_LOSS_REGISTRY = {
    "mse": mse_loss,
    "hinge": hinge_loss,
    "cross_entropy": cross_entropy_loss,
}


def get_loss_fn(name: str | Callable, backend: str = "pennylane") -> Callable:
    if callable(name):
        return name

    name = name.lower()

    if backend == "pennylane":
        if name not in _LOSS_REGISTRY:
            raise ValueError(
                f"Unknown PennyLane loss: {name!r}. "
                f"Available: {list(_LOSS_REGISTRY.keys())}"
            )
        return _LOSS_REGISTRY[name]

    if backend == "torch":
        import torch.nn.functional as F

        if name == "mse":
            return F.mse_loss
        elif name == "cross_entropy":
            return F.binary_cross_entropy
        else:
            raise NotImplementedError(f"Torch loss '{name}' not yet mapped.")

    raise ValueError(f"Unknown backend: {backend}")
