"""A module for losses."""

from pyqit.core.losses._registry import get_loss_fn, loss_registry
from pyqit.core.losses.base import BaseLoss
from pyqit.core.losses.cross_entropy import CrossEntropyLoss, cross_entropy_loss
from pyqit.core.losses.hinge import HingeLoss, hinge_loss
from pyqit.core.losses.mse import MSELoss, mse_loss

__all__ = [
    "BaseLoss",
    "MSELoss",
    "HingeLoss",
    "CrossEntropyLoss",
    "mse_loss",
    "hinge_loss",
    "cross_entropy_loss",
    "get_loss_fn",
    "loss_registry",
]
