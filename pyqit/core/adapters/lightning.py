from skbase.utils.dependencies import _check_soft_dependencies, _safe_import

torch = _safe_import("torch")
if _check_soft_dependencies("lightning", severity="none"):
    from lightning.pytorch import LightningDataModule, LightningModule
else:

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
        if hasattr(pyqit_model, "_qnodes"):
            for name, node in pyqit_model._qnodes.items():
                if isinstance(node, torch.nn.Module):
                    self.add_module(name, node)

    def forward(self, x):
        return self.pyqit_model(x)

    def training_step(self, batch, batch_idx):
        X, y = batch
        preds = self(X)

        if (
            self.loss_fn.__name__ == "cross_entropy"
            and preds.ndim > 1
            and preds.shape[1] > 1
        ):
            y = y.long()
        else:
            y = y.to(preds.dtype)

        loss = self.loss_fn(preds, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        preds = self(X)
        y = y.to(preds.dtype)
        loss = self.loss_fn(preds, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        parameters = list(self.parameters())

        if not parameters:
            raise ValueError("No parameters found! TorchLayers were not registered.")
        if self.optimizer_name == "sgd":
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
        (`_X_train`, `_y_train`, etc.) and configuration parameters.
    num_workers : int, default=0
        The number of workers for data loading. (Note: Currently overridden
        by `pyqit_dm.num_workers` in the loader configuration).
    train_loader_kwargs : dict or None, optional
        Additional keyword arguments to pass to the training DataLoader.
    eval_loader_kwargs : dict or None, optional
        Additional keyword arguments to pass to the evaluation DataLoader.
    """

    def __init__(
        self,
        pyqit_dm,
        num_workers: int = 0,
        train_loader_kwargs: dict | None = None,
        eval_loader_kwargs: dict | None = None,
    ):
        super().__init__()
        self.dm = pyqit_dm

    def _build_loader(self, X, y):
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
            shuffle=self.dm.shuffle,
            drop_last=self.dm.drop_last,
        )

    def train_dataloader(self):
        return self._build_loader(self.dm._X_train, self.dm._y_train)

    def val_dataloader(self):
        if self.dm._X_val is not None:
            return self._build_loader(self.dm._X_val, self.dm._y_val)
        return None

    def test_dataloader(self):
        return self._build_loader(self.dm._X_test, self.dm._y_test)
