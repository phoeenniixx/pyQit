import time

from pytorch_lightning.callbacks import Callback


class HistoryCallback(Callback):
    def __init__(self, history_obj):
        self.history = history_obj
        self.epoch_start_time = 0

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics

        t_loss = metrics.get("train_loss")
        v_loss = metrics.get("val_loss")

        train_loss = t_loss.item() if t_loss is not None else float("nan")
        val_loss = v_loss.item() if v_loss is not None else float("nan")

        elapsed = time.time() - self.epoch_start_time

        self.history.record(
            epoch=trainer.current_epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_acc=0.0,
            val_acc=0.0,
            epoch_time=elapsed,
        )
