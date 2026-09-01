"""Backend-neutral callback protocol."""

from dataclasses import dataclass, field
from typing import Any

from pyqit.base.base_object import _PyQitObject


@dataclass
class LoopState:
    """Everything a callback may read, and the one flag it may write.

    Attributes
    ----------
    model : BaseModel
        The model being trained.
    datamodule : DataModule
        The data it is training on, already set up.
    history : TrainingHistory
        Metrics recorded so far this run.
    reporter : Reporter
        Console output, for callbacks that announce something.
    max_epochs : int
        Epoch budget for the run.
    epoch : int
        Zero-based index of the epoch just finished; ``-1`` before the first.
    metrics : dict of {str: float}
        This epoch's metrics, keyed ``train_loss``, ``val_loss``, ``train_acc``,
        ``val_acc``, ``epoch_time``.
    stop : bool
        Set by a callback to end training after this epoch. Both loops check it;
        the Lightning loop forwards it to ``trainer.should_stop``.
    """

    model: Any
    datamodule: Any
    history: Any
    reporter: Any
    max_epochs: int
    epoch: int = -1
    metrics: dict[str, float] = field(default_factory=dict)
    stop: bool = False


class BaseCallback(_PyQitObject):
    """Base class for pyqit callbacks."""

    _tags = {
        "object_type": "callback",
    }

    def on_fit_start(self, state: LoopState) -> None:
        """Called once, after setup and before the first epoch."""

    def on_epoch_end(self, state: LoopState) -> None:
        """Called once per epoch, with ``state.metrics`` filled for that epoch."""

    def on_fit_end(self, state: LoopState) -> None:
        """Called once, after the last epoch, including after an early stop."""

    @classmethod
    def get_test_params(cls):
        return [{}]
