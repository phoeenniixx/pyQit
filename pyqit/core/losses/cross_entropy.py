import pennylane as qml
import pennylane.numpy as pnp

from pyqit.core.losses.base import BaseLoss


def cross_entropy_loss(preds, targets):
    """Cross-entropy loss function for binary and multi-class classification.

    Parameters
    ----------
    preds : array-like
        The predicted probabilities from the model. For binary classification,
        this should be a 1D array or a 2D array of shape `(n_samples, 1)`.
        For multi-class classification, this should be a 2D array of shape
        `(n_samples, n_classes)`.
    targets : array-like
        The ground truth labels. For binary classification, values should be
        0 or 1. For multi-class classification, values should be integer
        class indices in the range `[0, n_classes - 1]`.
    Returns
    -------
    float or tensor
        The computed mean cross-entropy loss across the batch."""
    probs = pnp.clip(preds, 1e-9, 1.0 - 1e-9)

    if probs.ndim == 1 or (probs.ndim == 2 and probs.shape[1] == 1):
        probs = probs.flatten()
        targets = qml.math.unwrap(targets)
        log_p = targets * pnp.log(probs) + (1.0 - targets) * pnp.log(1.0 - probs)
        return -pnp.mean(log_p)

    n = len(targets)
    if pnp.max(targets) >= probs.shape[1] or pnp.min(targets) < 0:
        raise ValueError(
            f"Target mismatch: Model output {probs.shape[1]} classes, "
            f"but targets contain class index {pnp.max(targets)}. "
            "Ensure the model's `n_classes` matches the number of "
            "unique classes in your dataset."
        )

    log_p = pnp.log(probs[pnp.arange(n), targets.astype(int)])
    return -pnp.mean(log_p)


class CrossEntropyLoss(BaseLoss):
    """Cross entropy over class probabilities."""

    _tags = {"name": "cross_entropy", "target_dtype": "int"}

    def _pennylane(self, preds, targets):
        return cross_entropy_loss(preds, targets)

    def _torch(self, preds, targets):
        import torch
        import torch.nn.functional as F

        probs = torch.clamp(preds, 1e-9, 1.0 - 1e-9)

        if probs.ndim == 1 or (probs.ndim == 2 and probs.shape[1] == 1):
            probs = probs.flatten()
            t = targets.to(probs.dtype).flatten()
            return -(t * torch.log(probs) + (1.0 - t) * torch.log(1.0 - probs)).mean()

        return F.nll_loss(torch.log(probs), targets.long().flatten())


_REGISTRY_CACHE = None
