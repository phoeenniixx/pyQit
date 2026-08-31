"""Hand-rolled training loop for the pennylane backend."""

import time

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp

from pyqit.core.losses import get_loss_fn
from pyqit.core.trainer.loops.base import BaseTrainingLoop
from pyqit.utils.utils import _hard_labels


class PennyLaneLoop(BaseTrainingLoop):
    """Functional training loop driven by ``qml`` optimizers.

    Weights are threaded through ``opt.step_and_cost`` as explicit arguments
    rather than mutated in place, which is what lets the same flat-kwargs
    routing serve both this loop and the barren-plateau sampler.
    """

    _tags = {
        "object_type": "training_loop",
        "backend": "pennylane",
        "rejects": {
            "backend_kwargs": (
                "backend_kwargs is forwarded verbatim to lightning.Trainer, and "
                "this backend has no Lightning trainer to forward it to."
            ),
        },
        "warns": {
            "logger": (
                "Lightning loggers are not used here; per-epoch metrics are "
                "returned in the TrainingHistory from Trainer.fit instead."
            ),
        },
        "reserved_backend_kwargs": (),
    }

    def fit(self, model, datamodule, state, callbacks) -> None:
        trainer = self.trainer
        loss_fn = get_loss_fn(trainer.loss_fn, backend="pennylane")

        weight_keys = list(model.weights.keys())
        current_weights = [
            pnp.array(model.weights[k], requires_grad=True) for k in weight_keys
        ]

        train_loader = datamodule.train_loader(shuffle=True)
        val_loader = datamodule.val_loader(shuffle=False)
        # Built once: the loader is re-iterable and shuffle=False makes it
        # identical every epoch, so rebuilding it per epoch bought nothing.
        train_eval_loader = (
            datamodule.train_loader(shuffle=False) if trainer.eval_train_acc else None
        )

        def batch_cost(X_b, y_b, *weight_tensors):
            flat_kwargs = dict(zip(weight_keys, weight_tensors))
            model.update_weights(flat_kwargs)
            preds = model.forward(X_b, **flat_kwargs)
            if preds.ndim == 0:
                preds = pnp.expand_dims(preds, axis=0)
            return loss_fn(preds, y_b)

        opt = self._make_optimizer(trainer)

        self._emit(callbacks, "on_fit_start", state)

        with self.reporter.progress() as progress:
            for epoch in range(trainer.max_epochs):
                t0 = time.time()
                batch_losses = []

                for X_batch, y_batch in train_loader:
                    X_b = pnp.array(X_batch, requires_grad=False)
                    y_b = pnp.array(y_batch, requires_grad=False)

                    args_out, batch_loss = opt.step_and_cost(
                        batch_cost, X_b, y_b, *current_weights
                    )
                    current_weights = list(args_out[2:])
                    batch_losses.append(float(batch_loss))

                model.update_weights(dict(zip(weight_keys, current_weights)))

                train_loss = float(np.mean(batch_losses))
                _, train_acc = self._evaluate(model, train_eval_loader)
                val_loss, val_acc = self._evaluate(model, val_loader, loss_fn)

                state.epoch = epoch
                state.metrics = {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "epoch_time": time.time() - t0,
                }
                self._emit(callbacks, "on_epoch_end", state)

                progress.update(
                    epoch, train_loss, val_loss, state.metrics["epoch_time"]
                )

                if state.stop:
                    break

            progress.finish()

        self._emit(callbacks, "on_fit_end", state)
        self.reporter.success("Training complete.")

    @staticmethod
    def _make_optimizer(trainer):
        name = trainer.optimizer
        name = name.lower() if isinstance(name, str) else name
        if name == "adam":
            return qml.AdamOptimizer(stepsize=trainer.learning_rate)
        return qml.GradientDescentOptimizer(stepsize=trainer.learning_rate)

    @staticmethod
    def _evaluate(model, dataloader, loss_fn=None) -> tuple[float, float]:
        """Mean loss and hard-label accuracy over ``dataloader`` in one pass.

        Loss and accuracy used to be two methods that each iterated the loader
        and ran their own forward pass over it. On a backend where the circuit
        evaluation dominates, that doubled the cost of every evaluated split to
        produce two numbers from the same predictions.
        """
        if dataloader is None:
            return float("nan"), float("nan")

        losses = []
        correct, total = 0, 0

        for X_b, y_b in dataloader:
            preds = model.forward(X_b)

            if loss_fn is not None:
                y_target = pnp.array(y_b, requires_grad=False)
                losses.append(float(loss_fn(preds, y_target)))

            labels = _hard_labels(preds)
            y_true = y_b.astype(int).flatten()
            correct += np.sum(labels == y_true)
            total += len(y_true)

        loss = float(np.mean(losses)) if losses else float("nan")
        accuracy = float(correct / total) if total > 0 else float("nan")
        return loss, accuracy
