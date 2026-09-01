"""Backend training loops."""

from pyqit.core.trainer.loops._registry import get_training_loop, loop_registry
from pyqit.core.trainer.loops.base import BaseTrainingLoop
from pyqit.core.trainer.loops.lightning_loop import LightningLoop
from pyqit.core.trainer.loops.pennylane_loop import PennyLaneLoop

__all__ = [
    "BaseTrainingLoop",
    "PennyLaneLoop",
    "LightningLoop",
    "loop_registry",
    "get_training_loop",
]
