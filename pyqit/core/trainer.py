import time
import numpy as np
from typing import Optional, Callable, List, Dict, Union

import pennylane as qml
import pennylane.numpy as pnp

from pyqit.data.datamodule import DataModule
from pyqit.base import BaseModel

class TrainingHistory:
    def __init__(self):
        self.train_loss:  List[float] = []
        self.val_loss:    List[float] = []
        self.train_acc:   List[float] = []
        self.val_acc:     List[float] = []
        self.epoch_times: List[float] = []
        self.best_epoch:     int   = 0
        self.best_val_loss:  float = float("inf")

    def record(
        self,
        epoch:      int,
        train_loss: float,
        val_loss:   float  = float("nan"),
        train_acc:  float  = 0.0,
        val_acc:    float  = 0.0,
        epoch_time: float  = 0.0,
    ) -> None:
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.train_acc.append(train_acc)
        self.val_acc.append(val_acc)
        self.epoch_times.append(epoch_time)
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch    = epoch

    def as_dict(self) -> Dict[str, List[float]]:
        return {
            "train_loss":  self.train_loss,
            "val_loss":    self.val_loss,
            "train_acc":   self.train_acc,
            "val_acc":     self.val_acc,
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
        max_epochs:    int   = 30,
        learning_rate: float = 0.01,
        batch_size:    int   = 32,
        optimizer:     str   = "adam",
        loss_fn:       Union[str, Callable] = "mse",
        backend_type:  str   = "auto",
        verbose:       int   = 5,
        seed:          Optional[int] = 42,
        num_workers:   int   = 0,
    ):
        self.max_epochs    = max_epochs
        self.lr            = learning_rate
        self.batch_size    = batch_size
        self.optimizer     = optimizer.lower() if isinstance(optimizer, str) else optimizer
        self.loss_fn       = loss_fn
        self.backend_type  = backend_type
        self.verbose       = verbose
        self.seed          = seed
        self.num_workers   = num_workers


    def fit(
        self,
        model:      BaseModel,
        datamodule: DataModule,
    ) -> TrainingHistory:
        backend = self._resolve_backend(model)

        n_qubits = getattr(model, "n_qubits", None)
        datamodule.setup(
            backend    = backend,
            batch_size = self.batch_size,
            n_qubits   = n_qubits,
            num_workers= self.num_workers,
        )

        if backend == "torch":
            return self._fit_torch(model, datamodule)
        else:
            return self._fit_pennylane(model, datamodule)


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
        model:      BaseModel,
        datamodule: DataModule,
    ) -> TrainingHistory:
        history = TrainingHistory()

        weight_keys = list(model.weights.keys())
        current_weights = [
            pnp.array(model.weights[k], requires_grad=True)
            for k in weight_keys
        ]

        loss_fn = self._resolve_loss_fn()
        train_loader = datamodule.train_loader(shuffle=True)
        val_loader   = datamodule.val_loader(shuffle=False)

        print(f"[Trainer] PennyLane backend | {self.max_epochs} epochs | lr={self.lr}")

        for epoch in range(self.max_epochs):
            t0 = time.time()
            batch_losses = []

            for X_batch, y_batch in train_loader:
                X_b = pnp.array(X_batch, requires_grad=False)
                y_b = pnp.array(y_batch, requires_grad=False)

                def batch_cost(*weight_tensors):
                    preds = pnp.array([
                        model.forward_from_tensors(pnp.atleast_1d(x), *weight_tensors)
                        for x in X_b
                    ])
                    return loss_fn(preds, y_b)

                grads      = qml.grad(batch_cost)(*current_weights)
                batch_loss = float(batch_cost(*current_weights))
                batch_losses.append(batch_loss)

                current_weights = self._apply_gradients(current_weights, grads, epoch)

            train_loss = float(np.mean(batch_losses))
            train_acc  = self._accuracy_pennylane(model, current_weights,
                                                   datamodule.X_train, datamodule.y_train)

            val_loss, val_acc = float("nan"), float("nan")
            if val_loader is not None:
                val_loss = self._eval_loss_pennylane(
                    model, current_weights, datamodule.X_val, datamodule.y_val, loss_fn
                )
                val_acc = self._accuracy_pennylane(
                    model, current_weights, datamodule.X_val, datamodule.y_val
                )

            elapsed = time.time() - t0
            history.record(epoch, train_loss, val_loss, train_acc, val_acc, elapsed)

            if self.verbose and (epoch % self.verbose == 0 or epoch == self.max_epochs - 1):
                self._print_epoch(epoch, train_loss, val_loss, train_acc, val_acc, elapsed)
        model.update_weights(dict(zip(weight_keys, current_weights)))
        print("[Trainer] Training complete.")
        return history

    def _apply_gradients(
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
            self._adam_v[i] = b2 * self._adam_v[i] + (1 - b2) * g ** 2
            m_hat = self._adam_m[i] / (1 - b1 ** t)
            v_hat = self._adam_v[i] / (1 - b2 ** t)
            w_new = w - self.lr * m_hat / (np.sqrt(v_hat) + eps)
            new_weights.append(pnp.array(w_new, requires_grad=True))

        return new_weights

    def _accuracy_pennylane(self, model, weights, X, y) -> float:
        if X is None:
            return float("nan")
        weight_keys = list(model.weights.keys())
        original = {k: model.weights[k] for k in weight_keys}
        model.update_weights(dict(zip(weight_keys, weights)))
        preds = np.array([
            model.forward(pnp.atleast_1d(x)) for x in X
        ])
        model.update_weights(original)
        if preds.ndim > 1:
            preds = preds.argmax(axis=1)
        else:
            preds = (preds >= 0.5).astype(int)
        return float(np.mean(preds == y.astype(int)))

    def _eval_loss_pennylane(self, model, weights, X, y, loss_fn) -> float:
        if X is None:
            return float("nan")
        weight_keys = list(model.weights.keys())
        preds = pnp.array([
            model.forward_from_tensors(pnp.atleast_1d(x), *weights)
            for x in X
        ])
        return float(loss_fn(preds, pnp.array(y, requires_grad=False)))



    def _fit_torch(
        self,
        model:      BaseModel,
        datamodule: DataModule,
    ) -> TrainingHistory:
        pass  # Placeholder for future PyTorch implementation

    def _resolve_loss_fn(self) -> Callable:
        """Return a callable loss function."""
        if callable(self.loss_fn):
            return self.loss_fn

        name = self.loss_fn
        if name == "mse":
            return lambda preds, targets: pnp.mean((preds - targets) ** 2)

        elif name == "hinge":
            def _hinge(preds, targets):
                y_signed = 2.0 * targets - 1.0   # {0,1} → {-1,+1}
                return pnp.mean(pnp.maximum(0, 1 - y_signed * preds))
            return _hinge

        elif name == "cross_entropy":
            def _ce(preds, targets):
                # preds: (n, n_classes) probabilities
                n = len(targets)
                probs = pnp.clip(preds, 1e-9, 1.0)
                log_p = pnp.log(probs[pnp.arange(n), targets.astype(int)])
                return -pnp.mean(log_p)
            return _ce

        raise ValueError(
            f"Unknown loss function: {name!r}. "
            f"Choose 'mse', 'hinge', 'cross_entropy', or pass a callable."
        )

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