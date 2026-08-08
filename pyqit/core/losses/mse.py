import pennylane.numpy as pnp

from pyqit.core.losses.base import BaseLoss


def mse_loss(preds, targets):
    """Mean Squared Error loss function.
    Parameters
    ----------
    preds : array-like
        The predicted values from the model.
    targets : array-like
        The ground truth target values. Expected to have the same shape
        as `preds`.

    Returns
    -------
    float or tensor
        The computed mean squared error loss across the batch.
    """
    return pnp.mean((preds - targets) ** 2)


class MSELoss(BaseLoss):
    """Mean squared error."""

    _tags = {"name": "mse"}

    def _pennylane(self, preds, targets):
        return mse_loss(preds, targets)

    def _torch(self, preds, targets):
        import torch.nn.functional as F

        return F.mse_loss(preds, targets.to(preds.dtype))
