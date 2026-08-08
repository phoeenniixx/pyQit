import pennylane.numpy as pnp

from pyqit.core.losses.base import BaseLoss


def hinge_loss(preds, targets):
    """Hinge loss function for binary classification.
    Parameters
    ----------
    preds : array-like
        The predicted raw scores (logits) from the model.
    targets : array-like
        The ground truth binary labels, expected to be encoded as 0 or 1.

    Returns
    -------
    float or tensor
        The computed mean hinge loss across the batch."""
    y_signed = 2.0 * targets - 1.0
    return pnp.mean(pnp.maximum(0, 1 - y_signed * preds))


class HingeLoss(BaseLoss):
    """Hinge loss for binary labels encoded as 0/1."""

    _tags = {"name": "hinge"}

    def _pennylane(self, preds, targets):
        return hinge_loss(preds, targets)

    def _torch(self, preds, targets):
        import torch

        y_signed = 2.0 * targets.to(preds.dtype) - 1.0
        return torch.clamp(1.0 - y_signed * preds, min=0.0).mean()
