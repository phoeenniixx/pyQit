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
    def __init__(self, pyqit_model, lr, optimizer_name, loss_fn):
        super().__init__()
        self.pyqit_model = pyqit_model
        self.lr = lr
        self.optimizer_name = optimizer_name
        self.loss_fn = loss_fn

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
        parameters = list(self.pyqit_model.weights.values())
        if self.optimizer_name == "sgd":
            return torch.optim.SGD(parameters, lr=self.lr)
        return torch.optim.Adam(parameters, lr=self.lr)


class _LightningDataAdapter(LightningDataModule):
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
