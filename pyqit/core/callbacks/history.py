import time

from skbase.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("lightning", severity="none"):
    from lightning.pytorch import Callback
else:

    class Callback:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Lightning is not installed. "
                "Please install it to use the PyTorch backend."
            )


class HistoryCallback(Callback):
    def __init__(self, history_obj):
        self.history = history_obj
        self.epoch_start_time = 0

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics

        t_loss = metrics.get("train_loss", float("nan"))
        v_loss = metrics.get("val_loss", float("nan"))
        t_acc = metrics.get("train_acc", float("nan"))
        v_acc = metrics.get("val_acc", float("nan"))

        elapsed = time.time() - self.epoch_start_time

        self.history.record(
            epoch=trainer.current_epoch,
            train_loss=float(t_loss),
            val_loss=float(v_loss),
            train_acc=float(t_acc),
            val_acc=float(v_acc),
            epoch_time=elapsed,
        )
