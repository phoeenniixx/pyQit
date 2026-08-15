"""A module for losses."""

from pyqit.core.losses._registry import get_loss_fn
from pyqit.core.losses.base import BaseLoss
from pyqit.core.losses.cross_entropy import CrossEntropyLoss
from pyqit.core.losses.hinge import HingeLoss
from pyqit.core.losses.mse import MSELoss

__all__ = [
    "BaseLoss",
    "MSELoss",
    "HingeLoss",
    "CrossEntropyLoss",
    "get_loss_fn",
]
