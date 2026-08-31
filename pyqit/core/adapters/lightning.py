from skbase.utils.dependencies import _check_soft_dependencies, _safe_import

torch = _safe_import("torch")
if _check_soft_dependencies("lightning", severity="none"):
    from lightning.pytorch import Callback, LightningDataModule, LightningModule
else:

    class Callback:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Lightning is not installed. "
                "Please install it to use the PyTorch backend."
            )

    class LightningModule:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Lightning is not installed. "
                "Please install it to use the PyTorch backend."
            )

    class LightningDataModule:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Lightning is not installed. "
                "Please install it to use the PyTorch backend."
            )


class _LightningModelAdapter(LightningModule):
    """
    Adapter to wrap a model into a PyTorch Lightning Module.

    This class handles the training loop, validation loop, and optimizer
    configuration for the underlying PyQIT model, integrating seamlessly
    with the PyTorch Lightning Trainer.

    Parameters
    ----------
    pyqit_model : torch.nn.Module or callable
        The core model to be wrapped. If it contains a `_qnodes` dictionary
        with `torch.nn.Module` objects, they will be registered as submodules.
    lr : float
        The learning rate for the optimizer.
    optimizer_name : str
        The name of the optimizer to use. Supports "sgd"; defaults to "Adam"
        for all other values.
    loss_fn : callable
        The loss function used for training and validation. It must accept
        `preds` and `y` as arguments.
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
            for name, node in pyqit_model._qnodes.items():
                if isinstance(node, torch.nn.Module):
                    self.add_module(name, node)

    def forward(self, x):
        return self.pyqit_model(x)

    @staticmethod
    def _accuracy(preds, y):
        """Hard-label accuracy, using the same rule as the pennylane path."""
        if preds.ndim > 1 and preds.shape[1] > 1:
            labels = preds.argmax(dim=1)
        else:
            labels = (preds >= 0.5).long().flatten()
        return (labels == y.long().flatten()).to(preds.dtype).mean()

    def _prepare_target(self, preds, y):
        """Cast targets to what the configured loss expects."""
        if self.target_dtype == "int" and preds.ndim > 1 and preds.shape[1] > 1:
            return y.long()
        return y.to(preds.dtype)

    def training_step(self, batch, batch_idx):
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
        X, y = batch
        preds = self(X)
        y = self._prepare_target(preds, y)
        loss = self.loss_fn(preds, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", self._accuracy(preds, y), prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        parameters = list(self.parameters())

        if not parameters:
            raise ValueError("No parameters found! TorchLayers were not registered.")

        # Case-folded here rather than in Trainer.__init__, which must store its
        # arguments verbatim to satisfy skbase's get_params/clone contract. A
        # bare `== "sgd"` would hand Trainer(optimizer="SGD") an Adam optimizer
        # without saying so.
        name = self.optimizer_name
        name = name.lower() if isinstance(name, str) else name

        if name == "sgd":
            return torch.optim.SGD(parameters, lr=self.lr)
        return torch.optim.Adam(parameters, lr=self.lr)


class _LightningDataAdapter(LightningDataModule):
    """
    Adapter to wrap a PyQIT data module into a PyTorch Lightning DataModule.

    This class extracts the internal training, validation, and testing arrays
    from the PyQIT data module and converts them into standard PyTorch
    DataLoaders compatible with the PyTorch Lightning Trainer.

    Parameters
    ----------
    pyqit_dm : object
        The internal PyQIT data module containing the raw data attributes
        (`_X_train`, `_y_train`, etc.) and configuration parameters. Every
        loader setting — `batch_size`, `num_workers`, `shuffle`, `drop_last` —
        is read from it, so it is the only place to configure them.
    """

    def __init__(self, pyqit_dm):
        super().__init__()
        self.dm = pyqit_dm

    def _build_loader(self, X, y, shuffle=False):
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
        return self._build_loader(
            self.dm._X_train, self.dm._y_train, shuffle=self.dm.shuffle
        )

    def val_dataloader(self):
        if self.dm._X_val is not None:
            return self._build_loader(self.dm._X_val, self.dm._y_val)
        return None

    def test_dataloader(self):
        return self._build_loader(self.dm._X_test, self.dm._y_test)


class _PyQitCallbackShim(Callback):
    """Run pyqit callbacks from inside a Lightning training loop.

    This is the whole reason a pyqit callback works on both backends. It reads
    Lightning's ``callback_metrics`` into the same metric names the pennylane
    loop produces, fires the pyqit hooks, and forwards a callback's stop request
    onto ``trainer.should_stop`` so ``EarlyStopping`` ends a Lightning run the
    same way it ends a pennylane one.

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
        self._fire("on_fit_start")

    def on_train_epoch_start(self, trainer, pl_module):
        import time

        self._epoch_start = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
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
        self._fire("on_fit_end")
