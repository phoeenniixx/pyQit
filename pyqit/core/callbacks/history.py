"""Callback recording per-epoch metrics into a ``TrainingHistory``."""

from pyqit.core.callbacks.base import BaseCallback, LoopState


class HistoryCallback(BaseCallback):
    """Copy each epoch's metrics into a ``TrainingHistory``.

    Both loops install this, so ``history`` is filled by the same code path on
    either backend.

    Parameters
    ----------
    history_obj : TrainingHistory
        The object to append to. Supplied by the Trainer, which returns it
        from ``fit``.
    """

    def __init__(self, history_obj):
        self.history_obj = history_obj
        super().__init__()

    @property
    def history(self):
        """The ``TrainingHistory`` being filled."""
        return self.history_obj

    def on_epoch_end(self, state: LoopState) -> None:
        """Record this epoch's metrics."""
        m = state.metrics
        self.history_obj.record(
            epoch=state.epoch,
            train_loss=m.get("train_loss", float("nan")),
            val_loss=m.get("val_loss", float("nan")),
            train_acc=m.get("train_acc", float("nan")),
            val_acc=m.get("val_acc", float("nan")),
            epoch_time=m.get("epoch_time", 0.0),
        )

    @classmethod
    def get_test_params(cls):
        from pyqit.core.trainer import TrainingHistory

        return [{"history_obj": TrainingHistory()}]
