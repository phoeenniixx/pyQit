from collections.abc import Callable
import time

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
from skbase.base import BaseMetaObject
from skbase.utils.dependencies import _check_soft_dependencies, _safe_import

from pyqit.core._loss_mapping import get_loss_fn
from pyqit.core.config import get_backend
from pyqit.data.datamodule import DataModule
from pyqit.models.base.base import BaseModel

HAS_RICH = _check_soft_dependencies("rich", severity="none")
if HAS_RICH:
    from rich import box
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        Progress,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    from rich.table import Table

    console = Console()


class TrainingHistory:
    def __init__(self):
        self.train_loss: list[float] = []
        self.val_loss: list[float] = []
        self.train_acc: list[float] = []
        self.val_acc: list[float] = []
        self.epoch_times: list[float] = []
        self.best_epoch: int = 0
        self.best_val_loss: float = float("inf")

    def record(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float = float("nan"),
        train_acc: float = 0.0,
        val_acc: float = 0.0,
        epoch_time: float = 0.0,
    ) -> None:
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.train_acc.append(train_acc)
        self.val_acc.append(val_acc)
        self.epoch_times.append(epoch_time)
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_acc": self.train_acc,
            "val_acc": self.val_acc,
            "epoch_times": self.epoch_times,
        }

    def __repr__(self) -> str:
        if not self.train_loss:
            return "TrainingHistory(empty)"
        return (
            f"TrainingHistory("
            f"epochs={len(self.train_loss)}, "
            f"best_val_loss={self.best_val_loss:.4f} @ epoch {self.best_epoch})"
        )


class Trainer:
    def __init__(
        self,
        max_epochs: int = 30,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        optimizer: str = "adam",
        loss_fn: str | Callable = "mse",
        verbose: int = 2,
        seed: int | None = 42,
        num_workers: int = 2,
        enable_checkpointing: bool = False,
        logger: bool | object = False,
        lightning_accelerator: str = "cpu",
        check_bp: bool = False,
        bp_samples: int = 200,
    ):
        self.max_epochs = max_epochs
        self.lr = learning_rate
        self.batch_size = batch_size
        self.optimizer = optimizer.lower() if isinstance(optimizer, str) else optimizer
        self.loss_fn = loss_fn
        self.backend = get_backend()
        self.verbose = verbose
        self.seed = seed
        self.num_workers = num_workers
        self.enable_checkpointing = enable_checkpointing
        self.logger = logger
        self.lightning_accelerator = lightning_accelerator
        self.check_bp = check_bp
        self.bp_samples = bp_samples

    def fit(
        self,
        model: BaseModel,
        datamodule: DataModule,
    ) -> TrainingHistory:
        backend = self.backend
        model_encoder_class = None
        if hasattr(model, "_embedding_obj"):
            model_encoder_class = type(model._embedding_obj)

        n_qubits = getattr(model, "n_qubits", None)
        datamodule.setup(
            stage="fit",
            batch_size=self.batch_size,
            n_qubits=n_qubits,
            encoder=model_encoder_class,
        )

        if self.verbose >= 2:
            self._print_model_summary(model, datamodule, backend)
        elif self.verbose == 1:
            if HAS_RICH:
                console.print(
                    f"[bold cyan][Trainer][/bold cyan] \
                        Starting [green]{backend}[/green] backend |\
                              {self.max_epochs} epochs | lr={self.lr}\n"
                )
            else:
                print(
                    f"[Trainer] Starting {backend} backend | {self.max_epochs} epochs\
                          | lr={self.lr}"
                )

        if self.check_bp:
            self._run_bp_diagnostic(model, datamodule)

        if backend == "torch":
            return self._fit_torch(model, datamodule)
        else:
            return self._fit_pennylane(model, datamodule)

    def _run_bp_diagnostic(self, model, datamodule):
        """Runs the pre-flight gradient variance check."""
        import logging

        from pyqit.utils.diagnostic import check_barren_plateau

        logger = logging.getLogger("pyqit.trainer")
        result = check_barren_plateau(
            model=model,
            datamodule_or_X=datamodule,
            num_samples=self.bp_samples,
            loss_name=self.loss_fn,
            plot=False,
        )
        if self.verbose >= 1:
            print(result)

        if self.verbose == 0 and result.is_barren:
            logger.warning(
                "Barren Plateau detected! \
                           Gradient variance is critically low."
            )

    def _print_model_summary(self, model: BaseModel, dm: DataModule, backend: str):
        """Displays a professional table of the model architecture and data splits."""
        if not HAS_RICH:
            print(
                f"\n[Trainer] Starting {backend} backend | {self.max_epochs} epochs \
                    | lr={self.lr}"
            )
            return

        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("Parameter", style="dim", width=20)
        table.add_column("Value", style="bold")

        table.add_row("Model Name", type(model).__name__)
        table.add_row("Backend", backend.capitalize())
        table.add_row("Qubits", str(getattr(model, "n_qubits", "N/A")))
        table.add_row(
            "Ansatz",
            type(getattr(model, "ansatz_obj", None)).__name__
            if hasattr(model, "ansatz_obj")
            else "N/A",
        )
        table.add_row(
            "Encoder",
            type(getattr(model, "_embedding_obj", None)).__name__
            if hasattr(model, "_embedding_obj")
            else "N/A",
        )
        table.add_row("Optimizer", self.optimizer.upper())
        table.add_row("Learning Rate", str(self.lr))

        train_samples = len(dm.X_train) if dm.X_train is not None else 0
        val_samples = len(dm.X_val) if dm.X_val is not None else 0
        table.add_row("Train / Val Samples", f"{train_samples} / {val_samples}")

        console.print(table)
        console.print()

    def _get_progress_context(self):
        if HAS_RICH and self.verbose:
            return Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(complete_style="cyan", finished_style="bold green"),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console,
            )
        else:
            import contextlib

            return contextlib.nullcontext()

    def _native_progress(self, epoch, train_loss, val_loss, elapsed):
        import sys

        val_str = f" | val_loss={val_loss:.4f}" if not np.isnan(val_loss) else ""

        percent = (epoch + 1) / self.max_epochs
        bar_len = 30
        filled = int(round(bar_len * percent))

        bar = "=" * max(0, filled - 1) + ">" * min(1, filled) + "." * (bar_len - filled)

        msg = (
            f"\rEpoch {epoch+1:>{len(str(self.max_epochs))}}/{self.max_epochs} "
            f"[{bar}] {percent:.0%} | loss={train_loss:.4f}{val_str} [{elapsed:.1f}s]  "
        )

        sys.stdout.write(msg)
        sys.stdout.flush()

        if epoch + 1 == self.max_epochs:
            sys.stdout.write("\n")

    def _fit_pennylane(
        self,
        model: BaseModel,
        datamodule: DataModule,
    ) -> TrainingHistory:
        history = TrainingHistory()
        weight_keys = list(model.weights.keys())
        current_weights = [
            pnp.array(model.weights[k], requires_grad=True) for k in weight_keys
        ]

        loss_fn = get_loss_fn(self.loss_fn, backend="pennylane")
        train_loader = datamodule.train_loader(shuffle=True)
        val_loader = datamodule.val_loader(shuffle=False)
        if self.logger and self.logger is not True:
            import warnings

            warnings.warn(
                f"You passed a custom logger ({type(self.logger).__name__}) "
                "to the Trainer, "
                "but the 'pennylane' backend does not support Lightning loggers. "
                "Metrics will be stored in the TrainingHistory object instead.",
                UserWarning,
                stacklevel=2,
            )

        best_native_val_loss = float("inf")

        def batch_cost(X_b, y_b, *weight_tensors):
            flat_kwargs = dict(zip(weight_keys, weight_tensors))
            model.update_weights(flat_kwargs)
            preds = model.forward(X_b, **flat_kwargs)
            if preds.ndim == 0:
                preds = pnp.expand_dims(preds, axis=0)
            return loss_fn(preds, y_b)

        if self.optimizer == "adam":
            opt = qml.AdamOptimizer(stepsize=self.lr)
        else:
            opt = qml.GradientDescentOptimizer(stepsize=self.lr)

        with self._get_progress_context() as progress:
            if progress:
                task_id = progress.add_task("[cyan]Training...", total=self.max_epochs)

            for epoch in range(self.max_epochs):
                t0 = time.time()
                batch_losses = []

                for X_batch, y_batch in train_loader:
                    X_b = pnp.array(X_batch, requires_grad=False)
                    y_b = pnp.array(y_batch, requires_grad=False)

                    args_out, batch_loss = opt.step_and_cost(
                        batch_cost,
                        X_b,
                        y_b,
                        *current_weights,
                    )
                    current_weights = list(args_out[2:])
                    batch_losses.append(float(batch_loss))

                model.update_weights(dict(zip(weight_keys, current_weights)))

                train_loader_eval = datamodule.train_loader(shuffle=False)

                train_loss = float(np.mean(batch_losses))
                train_acc = self._accuracy_pl(model, train_loader_eval)

                val_loss, val_acc = float("nan"), float("nan")
                if val_loader is not None:
                    val_loss = self._eval_loss_pl(model, val_loader, loss_fn)
                    val_acc = self._accuracy_pl(model, val_loader)

                elapsed = time.time() - t0
                history.record(epoch, train_loss, val_loss, train_acc, val_acc, elapsed)

                if self.enable_checkpointing and not np.isnan(val_loss):
                    if val_loss < best_native_val_loss:
                        best_native_val_loss = val_loss

                        checkpoint_path = "pyqit_pennylane_checkpoint.npz"
                        numpy_weights = {
                            k: np.array(v) for k, v in zip(weight_keys, current_weights)
                        }
                        np.savez(checkpoint_path, **numpy_weights)

                        if self.verbose >= 1 and not HAS_RICH:
                            print(
                                "[Checkpoint] Saved new best model "
                                f"(val_loss: {val_loss:.4f})"
                            )

                if self.verbose >= 0:
                    if progress:
                        val_str = (
                            f"| Val Loss: {val_loss:.4f}"
                            if not np.isnan(val_loss)
                            else ""
                        )
                        progress.update(
                            task_id,
                            completed=epoch + 1,
                            description=f"[cyan]Epoch {epoch+1}/{self.max_epochs}\
                                  | Loss: {train_loss:.4f} {val_str}",
                        )
                        if epoch + 1 == self.max_epochs:
                            progress.refresh()
                            time.sleep(0.05)
                    else:
                        self._native_progress(epoch, train_loss, val_loss, elapsed)

        if self.verbose >= 1:
            if HAS_RICH:
                console.print("[bold green][Trainer] Training complete.[/bold green]")
            else:
                print("[Trainer] Training complete.")
        return history

    def _accuracy_pl(self, model, dataloader) -> float:
        if dataloader is None:
            return float("nan")

        correct, total = 0, 0
        for X_b, y_b in dataloader:
            preds = pnp.array(model.forward(X_b))

            if preds.ndim > 1 and preds.shape[1] > 1:
                preds_labels = preds.argmax(axis=1)
            else:
                preds_labels = (preds >= 0.5).astype(int)
                preds_labels = preds_labels.flatten()

            y_tens = y_b.astype(int).flatten()
            correct += np.sum(preds_labels == y_tens)
            total += len(y_tens)

        return float(correct / total) if total > 0 else float("nan")

    def _eval_loss_pl(self, model, dataloader, loss_fn) -> float:
        if dataloader is None:
            return float("nan")

        losses = []
        for X_b, y_b in dataloader:
            preds = model.forward(X_b)
            y_target = pnp.array(y_b, requires_grad=False)

            losses.append(float(loss_fn(preds, y_target)))

        return float(np.mean(losses))

    def _fit_torch(
        self,
        model: BaseModel,
        datamodule: DataModule,
    ) -> TrainingHistory:
        if _check_soft_dependencies("lightning", severity="none"):
            from lightning.pytorch import Trainer as LightningTrainer
            from lightning.pytorch.callbacks import ModelCheckpoint
        else:
            raise ImportError(
                "Lightning is not installed. "
                "Please install it to use the PyTorch backend."
            )

        from pyqit.core._loss_mapping import get_loss_fn
        from pyqit.core.adapters.lightning import (
            _LightningModelAdapter,
        )
        from pyqit.core.callbacks.history import HistoryCallback

        loss_func = get_loss_fn(self.loss_fn, backend="torch")
        pl_model = _LightningModelAdapter(model, self.lr, self.optimizer, loss_func)
        pl_data = datamodule.to_lightning()

        history = TrainingHistory()
        callbacks = [HistoryCallback(history)]
        if self.enable_checkpointing:
            callbacks.append(ModelCheckpoint(save_weights_only=True))

        pl_trainer = LightningTrainer(
            max_epochs=self.max_epochs,
            enable_progress_bar=(self.verbose >= 0),
            callbacks=callbacks,
            logger=self.logger,
            enable_model_summary=(self.verbose >= 2),
            enable_checkpointing=self.enable_checkpointing,
            accelerator=self.lightning_accelerator,
        )

        pl_trainer.fit(pl_model, datamodule=pl_data)

        return history

    def _accuracy_torch(self, model, dataloader) -> float:
        import torch

        if dataloader is None:
            return float("nan")

        correct, total = 0, 0
        with torch.no_grad():
            for X_b, y_b in dataloader:
                preds = model.forward(torch.as_tensor(X_b))
                y_tens = torch.as_tensor(y_b).squeeze()

                if preds.ndim > 1 and preds.shape[1] > 1:
                    preds_labels = preds.argmax(dim=1)
                else:
                    preds_labels = (preds >= 0.5).to(torch.int)

                correct += (preds_labels == y_tens).sum().item()
                total += len(y_tens)

        return float(correct / total) if total > 0 else float("nan")

    def _eval_loss_torch(self, model, dataloader, loss_fn) -> float:
        import torch

        if dataloader is None:
            return float("nan")

        losses = []
        with torch.no_grad():
            for X_b, y_b in dataloader:
                preds = model.forward(torch.as_tensor(X_b))
                y_tensor = torch.as_tensor(y_b)

                if preds.ndim == 2 and preds.shape[1] > 1 and y_tensor.ndim == 1:
                    y_target = y_tensor.to(torch.long)
                else:
                    y_target = y_tensor.to(dtype=preds.dtype)

                losses.append(float(loss_fn(preds, y_target).item()))

        return float(np.mean(losses))

    def predict(
        self,
        model: BaseModel | BaseMetaObject,
        datamodule: DataModule,
        return_format: str = "auto",
    ) -> np.ndarray:
        if not datamodule._is_setup:
            n_qubits = getattr(model, "n_qubits", None)
            datamodule.setup(
                stage="predict",
                batch_size=self.batch_size,
                n_qubits=n_qubits,
            )

        if datamodule.X_test is not None:
            loader = datamodule.test_loader(shuffle=False)
        elif datamodule.X_val is not None:
            loader = datamodule.val_loader(shuffle=False)
        else:
            loader = datamodule.train_loader(shuffle=False)

        all_preds = []

        if self.backend == "torch":
            import torch

            context = torch.no_grad()
        else:
            import contextlib

            context = contextlib.nullcontext()

        with context:
            for X_batch, _ in loader:
                if self.backend == "pennylane":
                    X_b = pnp.array(X_batch, requires_grad=False)
                else:
                    X_b = X_batch

                batch_preds = model.predict_step(X_b)
                all_preds.append(batch_preds)

        if not all_preds:
            return np.array([])

        is_torch_output = type(all_preds[0]).__module__.startswith("torch")

        target_format = return_format
        if return_format == "auto":
            target_format = "torch" if is_torch_output else "numpy"

        if target_format == "torch":
            import torch

            tensor_preds = [
                p if type(p).__module__.startswith("torch") else torch.as_tensor(p)
                for p in all_preds
            ]
            return torch.cat(tensor_preds, dim=0)

        else:
            numpy_preds = [
                p.detach().cpu().numpy()
                if type(p).__module__.startswith("torch")
                else np.asarray(p)
                for p in all_preds
            ]
            return np.concatenate(numpy_preds, axis=0)

    @staticmethod
    def _print_epoch(epoch, train_loss, val_loss, train_acc, val_acc, elapsed):
        val_str = (
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}"
            if not np.isnan(val_loss)
            else "no validation"
        )
        print(
            f"  epoch {epoch:>4}  "
            f"loss={train_loss:.4f}  acc={train_acc:.3f}  "
            f"{val_str}  [{elapsed:.1f}s]"
        )

    def __repr__(self) -> str:
        return (
            f"Trainer(backend={self.backend!r}, "
            f"max_epochs={self.max_epochs}, lr={self.lr})"
        )
