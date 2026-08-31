"""Training orchestration.

``Trainer`` is the entry point; the backend loops under ``loops/`` do the
actual fitting. ``HAS_RICH`` and ``console`` are re-exported because
``core/pipeline.py`` and user code import them from here.
"""

from pyqit.core.trainer._reporting import HAS_RICH, Reporter, console
from pyqit.core.trainer.history import TrainingHistory
from pyqit.core.trainer.trainer import Trainer

__all__ = [
    "Trainer",
    "TrainingHistory",
    "Reporter",
    "HAS_RICH",
    "console",
]
