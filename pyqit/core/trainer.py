from collections.abc import Callable
import time

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
from skbase.base import BaseMetaObject
from skbase.utils.dependencies import _check_soft_dependencies

from pyqit.core._loss_mapping import get_loss_fn
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
        backend_type: str = "auto",
        verbose: int = 2,
        seed: int | None = 42,
        num_workers: int = 0,
    ):
        self.max_epochs = max_epochs
        self.lr = learning_rate
        self.batch_size = batch_size
        self.optimizer = optimizer.lower() if isinstance(optimizer, str) else optimizer
        self.loss_fn = loss_fn
        self.backend_type = backend_type
        self.verbose = verbose
        self.seed = seed
        self.num_workers = num_workers

    def fit(
        self,
        model: BaseModel,
        datamodule: DataModule,
    ) -> TrainingHistory:
        backend = self._resolve_backend(model)
        model_encoder_class = None
        if hasattr(model, "_embedding_obj"):
            model_encoder_class = type(model._embedding_obj)

        n_qubits = getattr(model, "n_qubits", None)
        datamodule.setup(
            stage="fit",
            backend=backend,
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

        if backend == "torch":
            return self._fit_torch(model, datamodule)
        else:
            return self._fit_pennylane(model, datamodule)

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

    def _resolve_backend(self, model: BaseModel) -> str:
        if self.backend_type != "auto":
            if self.backend_type not in ("pennylane", "torch"):
                raise ValueError(
                    f"backend_type must be 'pennylane', 'torch', or 'auto'. "
                    f"Got {self.backend_type!r}."
                )
            return self.backend_type

        model_type = getattr(model, "metadata", {}).get("model_type", "pure_qml")
        return "torch" if model_type == "hybrid" else "pennylane"

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

        with self._get_progress_context() as progress:
            if progress:
                task_id = progress.add_task("[cyan]Training...", total=self.max_epochs)

            for epoch in range(self.max_epochs):
                t0 = time.time()
                batch_losses = []

                for X_batch, y_batch in train_loader:
                    X_b = pnp.array(X_batch, requires_grad=False)
                    y_b = pnp.array(y_batch, requires_grad=False)

                    def batch_cost(*weight_tensors):
                        model.update_weights(dict(zip(weight_keys, weight_tensors)))
                        preds = model.forward(X_b, *weight_tensors)
                        if preds.ndim == 0:
                            preds = pnp.expand_dims(preds, axis=0)
                        return loss_fn(preds, y_b)

                    grads = qml.grad(batch_cost)(*current_weights)
                    batch_loss = float(batch_cost(*current_weights))
                    batch_losses.append(batch_loss)

                    current_weights = self._apply_gradients_pl(
                        current_weights, grads, epoch
                    )

                model.update_weights(dict(zip(weight_keys, current_weights)))

                train_loss = float(np.mean(batch_losses))
                train_acc = self._accuracy_pl(
                    model, datamodule.X_train, datamodule.y_train
                )

                val_loss, val_acc = float("nan"), float("nan")
                if val_loader is not None:
                    val_loss = self._eval_loss_pl(
                        model, datamodule.X_val, datamodule.y_val, loss_fn
                    )
                    val_acc = self._accuracy_pl(
                        model, datamodule.X_val, datamodule.y_val
                    )

                elapsed = time.time() - t0
                history.record(epoch, train_loss, val_loss, train_acc, val_acc, elapsed)

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

    def _apply_gradients_pl(
        self,
        weights: list,
        grads: tuple,
        epoch: int,
    ) -> list:
        if self.optimizer == "sgd":
            return [
                pnp.array(w - self.lr * g, requires_grad=True)
                for w, g in zip(weights, grads)
            ]

        if not hasattr(self, "_adam_m"):
            self._adam_m = [np.zeros_like(w) for w in weights]
            self._adam_v = [np.zeros_like(w) for w in weights]
            self._adam_t = 0

        b1, b2, eps = 0.9, 0.999, 1e-8
        self._adam_t += 1
        t = self._adam_t
        new_weights = []

        for i, (w, g) in enumerate(zip(weights, grads)):
            self._adam_m[i] = b1 * self._adam_m[i] + (1 - b1) * g
            self._adam_v[i] = b2 * self._adam_v[i] + (1 - b2) * g**2
            m_hat = self._adam_m[i] / (1 - b1**t)
            v_hat = self._adam_v[i] / (1 - b2**t)
            w_new = w - self.lr * m_hat / (np.sqrt(v_hat) + eps)
            new_weights.append(pnp.array(w_new, requires_grad=True))

        return new_weights

    def _accuracy_pl(self, model, X, y) -> float:
        if X is None:
            return float("nan")
        preds = np.array([model.forward(pnp.atleast_1d(x)) for x in X])

        if preds.ndim > 1:
            preds = preds.argmax(axis=1)
        else:
            preds = (preds >= 0.5).astype(int)

        return float(np.mean(preds == y.astype(int)))

    def _eval_loss_pl(self, model, X, y, loss_fn) -> float:
        if X is None:
            return float("nan")
        preds = pnp.array([model.forward(pnp.atleast_1d(x)) for x in X])
        return float(loss_fn(preds, pnp.array(y, requires_grad=False)))

    def _fit_torch(
        self,
        model: BaseModel,
        datamodule: DataModule,
    ) -> TrainingHistory:
        from lightning.pytorch import Trainer
        from lightning.pytorch.callbacks import RichProgressBar
        import torch

        from pyqit.core._loss_mapping import get_loss_fn
        from pyqit.core.adapters.lightning import (
            _LightningDataAdapter,
            _LightningModelAdapter,
        )
        from pyqit.core.callbacks.history import HistoryCallback

        torch_weights = {
            k: (
                torch.tensor(np.array(w), dtype=torch.float64, requires_grad=True)
                if not isinstance(w, torch.Tensor)
                else w.detach().requires_grad_(True)
            )
            for k, w in model.weights.items()
        }
        model.update_weights(torch_weights)

        loss_func = get_loss_fn(self.loss_fn, backend="torch")
        pl_model = _LightningModelAdapter(model, self.lr, self.optimizer, loss_func)
        pl_data = _LightningDataAdapter(datamodule)

        history = TrainingHistory()
        callbacks = [HistoryCallback(history)]
        if self.verbose >= 0:
            callbacks.append(RichProgressBar())

        pl_trainer = Trainer(
            max_epochs=self.max_epochs,
            enable_progress_bar=(self.verbose >= 0),
            callbacks=callbacks,
            logger=False,
            enable_model_summary=(self.verbose >= 2),
        )

        pl_trainer.fit(pl_model, datamodule=pl_data)

        return history

    def _accuracy_torch(self, model, X, y) -> float:
        import torch

        if X is None:
            return float("nan")

        with torch.no_grad():
            X_tens = torch.as_tensor(X)
            preds = model.forward(X_tens)
            y_tens = torch.as_tensor(y).squeeze()

            if preds.ndim > 1 and preds.shape[1] > 1:
                preds_labels = preds.argmax(dim=1)
            else:
                preds_labels = (preds >= 0.5).to(torch.int)

            return float((preds_labels == y_tens).float().mean().item())

    def _eval_loss_torch(self, model, X, y, loss_fn) -> float:
        import torch

        if X is None:
            return float("nan")

        with torch.no_grad():
            preds = torch.stack([model.forward(torch.as_tensor(x)) for x in X])
            loss = loss_fn(preds, torch.as_tensor(y).to(dtype=preds.dtype))

        return float(loss.item())

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
                backend=self.backend_type,
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

        if self.backend_type == "torch":
            import torch

            context = torch.no_grad()
        else:
            import contextlib

            context = contextlib.nullcontext()

        with context:
            for X_batch, _ in loader:
                if self.backend_type == "pennylane":
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
            f"Trainer(backend={self.backend_type!r}, "
            f"max_epochs={self.max_epochs}, lr={self.lr})"
        )
