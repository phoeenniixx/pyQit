import pennylane.numpy as pnp

from pyqit.core.losses.base import BaseLoss


def hinge_loss(preds, targets):
    """Hinge loss function for binary classification.
    Parameters
    ----------
    preds : array-like
        Class-1 probabilities from the model, mapped back to signed scores
        in ``[-1, 1]`` so a confident correct prediction reaches zero loss.
    targets : array-like
        The ground truth binary labels, expected to be encoded as 0 or 1.

    Returns
    -------
    float or tensor
        The computed mean hinge loss across the batch."""
    y_signed = 2.0 * targets - 1.0
    scores = 2.0 * preds - 1.0
    return pnp.mean(pnp.maximum(0, 1 - scores * y_signed))


class HingeLoss(BaseLoss):
    """Hinge loss for binary labels encoded as 0/1."""

    _tags = {"name": "hinge"}

    def _pennylane(self, preds, targets):
        return hinge_loss(preds, targets)

    def _torch(self, preds, targets):
        import torch

        y_signed = 2.0 * targets.to(preds.dtype) - 1.0
        scores = 2.0 * preds - 1.0
        return torch.clamp(1.0 - y_signed * scores, min=0.0).mean()
