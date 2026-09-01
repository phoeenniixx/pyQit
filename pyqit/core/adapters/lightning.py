"""Adapters wrapping pyqit objects in the Lightning classes."""

from skbase.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies(["lightning", "torch"], severity="none"):
    from lightning.pytorch import Callback, LightningDataModule, LightningModule
else:
    _MESSAGE = (
        "The PyTorch backend requires both lightning and torch, and at least "
        "one is not installed. Install them with `pip install pyqit[all_extras]`."
    )

    class Callback:
        def __init__(self, *args, **kwargs):
            raise ImportError(_MESSAGE)

    class LightningModule:
        def __init__(self, *args, **kwargs):
            raise ImportError(_MESSAGE)

    class LightningDataModule:
        def __init__(self, *args, **kwargs):
            raise ImportError(_MESSAGE)


class _LightningModelAdapter(LightningModule):
    """Wrap a pyqit model as a ``LightningModule``.

    Registers the model's ``TorchLayer`` qnodes as submodules so Lightning can
    see their parameters, and implements the training, validation and optimizer
    hooks in terms of the pyqit loss.

    Parameters
    ----------
    pyqit_model : torch.nn.Module or callable
        The model to wrap. Any ``torch.nn.Module`` in its ``_qnodes`` dict is
        registered as a submodule.
    lr : float
        Learning rate for the optimizer.
    optimizer_name : str
        ``"sgd"`` selects SGD, case-insensitively; anything else selects Adam.
    loss_fn : callable
        Takes ``(preds, y)``. Its ``target_dtype`` tag decides whether targets
        are cast to class indices.
    """

    def __init__(self, pyqit_model, lr, optimizer_name, loss_fn):
        super().__init__()
        self.pyqit_model = pyqit_model
        self.lr = lr
        self.optimizer_name = optimizer_name
        self.loss_fn = loss_fn
        get_tag = getattr(loss_fn, "get_tag", None)
        self.target_dtype = (
            get_tag("target_dtype", "float")
            if get_tag is not None
            else (
                "int"
                if getattr(loss_fn, "__name__", "") == "cross_entropy"
                else "float"
            )
        )
        if hasattr(pyqit_model, "_qnodes"):
            import torch

            for name, node in pyqit_model._qnodes.items():
                if isinstance(node, torch.nn.Module):
                    self.add_module(name, node)

    def forward(self, x):
        """Model predictions for a batch."""
        return self.pyqit_model(x)

    @staticmethod
    def _accuracy(preds, y):
        """Hard-label accuracy, using the shared cross-backend labelling rule."""
        from pyqit.utils.utils import _hard_labels

        labels = _hard_labels(preds)
        return (labels == y.long().flatten()).to(preds.dtype).mean()

    def _prepare_target(self, preds, y):
        """Cast targets to what the configured loss expects."""
        if self.target_dtype == "int" and preds.ndim > 1 and preds.shape[1] > 1:
            return y.long()
        return y.to(preds.dtype)

    def training_step(self, batch, batch_idx):
        """Loss for one training batch, logging ``train_loss`` and ``train_acc``."""
        X, y = batch
        preds = self(X)
        y = self._prepare_target(preds, y)

        loss = self.loss_fn(preds, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "train_acc",
            self._accuracy(preds, y),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """Log ``val_loss`` and ``val_acc`` for one validation batch."""
        X, y = batch
        preds = self(X)
        y = self._prepare_target(preds, y)
        loss = self.loss_fn(preds, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", self._accuracy(preds, y), prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        import torch

        parameters = list(self.parameters())

        if not parameters:
            raise ValueError("No parameters found! TorchLayers were not registered.")

        name = self.optimizer_name
        name = name.lower() if isinstance(name, str) else name

        if name == "sgd":
            return torch.optim.SGD(parameters, lr=self.lr)
        return torch.optim.Adam(parameters, lr=self.lr)


class _LightningDataAdapter(LightningDataModule):
    """Wrap a pyqit ``DataModule`` as a ``LightningDataModule``.

    Parameters
    ----------
    pyqit_dm : DataModule
        Supplies the split arrays and every loader setting -- ``batch_size``,
        ``num_workers``, ``shuffle``, ``drop_last`` -- so it is the only place
        to configure them.
    """

    def __init__(self, pyqit_dm):
        super().__init__()
        self.dm = pyqit_dm

    def setup(self, stage=None):
        """Check the wrapped DataModule is ready."""
        if not self.dm._is_setup:
            raise RuntimeError(
                "The wrapped DataModule is not set up. Call pyqit's "
                "Trainer.fit, or DataModule.setup(n_qubits=..., encoder=...) "
                "with the values your model implies, before handing the "
                "adapter to a Lightning Trainer."
            )

    def _build_loader(self, X, y, shuffle=False):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if X is None or y is None:
            return None

        dataset = TensorDataset(
            torch.as_tensor(X, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.float32),
        )
        return DataLoader(
            dataset,
            batch_size=self.dm.batch_size,
            num_workers=self.dm.num_workers,
            shuffle=shuffle,
            drop_last=self.dm.drop_last,
        )

    def train_dataloader(self):
        """Loader over the training split."""
        return self._build_loader(
            self.dm._X_train, self.dm._y_train, shuffle=self.dm.shuffle
        )

    def val_dataloader(self):
        """Loader over the validation split, or None if there is none."""
        if self.dm._X_val is not None:
            return self._build_loader(self.dm._X_val, self.dm._y_val)
        return None

    def test_dataloader(self):
        """Loader over the test split."""
        return self._build_loader(self.dm._X_test, self.dm._y_test)


class _PyQitCallbackShim(Callback):
    """Run pyqit callbacks from inside a Lightning training loop.

    Reads Lightning's ``callback_metrics`` into the metric names the pennylane
    loop produces, fires the pyqit hooks, and forwards a callback's stop request
    onto ``trainer.should_stop``. This is what lets one pyqit callback work on
    both backends.

    Parameters
    ----------
    callbacks : list of BaseCallback
        The pyqit callbacks to drive.
    state : LoopState
        Shared state; the shim fills ``epoch`` and ``metrics`` each epoch.
    """

    METRIC_KEYS = ("train_loss", "val_loss", "train_acc", "val_acc")

    def __init__(self, callbacks, state):
        self.callbacks = list(callbacks)
        self.state = state
        self._epoch_start = 0.0

    def _fire(self, hook):
        for callback in self.callbacks:
            getattr(callback, hook)(self.state)

    def on_fit_start(self, trainer, pl_module):
        """Fire ``on_fit_start`` on every pyqit callback."""
        self._fire("on_fit_start")

    def on_train_epoch_start(self, trainer, pl_module):
        """Start this epoch's timer."""
        import time

        self._epoch_start = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        """Fill ``state.metrics``, fire ``on_epoch_end``, honour a stop request."""
        import time

        metrics = trainer.callback_metrics
        self.state.epoch = trainer.current_epoch
        self.state.metrics = {
            key: float(metrics.get(key, float("nan"))) for key in self.METRIC_KEYS
        }
        self.state.metrics["epoch_time"] = time.time() - self._epoch_start

        self._fire("on_epoch_end")

        if self.state.stop:
            trainer.should_stop = True

    def on_fit_end(self, trainer, pl_module):
        """Fire ``on_fit_end`` on every pyqit callback."""
        self._fire("on_fit_end")
