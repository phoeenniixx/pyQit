from lightning.pytorch import LightningDataModule, LightningModule
import torch


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
    def __init__(self, pyqit_dm):
        super().__init__()
        self.dm = pyqit_dm

    def train_dataloader(self):
        return self.dm.train_loader(shuffle=True)

    def val_dataloader(self):
        if self.dm.X_val is not None:
            return self.dm.val_loader(shuffle=False)
        return None
