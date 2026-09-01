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

    _tags = {"backend": "pennylane"}

    # backend_kwargs targets lightning.pytorch.Trainer; there is none here.
    rejects = ("backend_kwargs",)
    # Metrics are returned in the TrainingHistory instead.
    warns = ("logger",)

    def fit(self, model, datamodule, state, callbacks: list) -> None:
        """Train ``model`` with a ``qml`` optimizer. See ``BaseTrainingLoop.fit``."""
        trainer = self.trainer
        loss_fn = get_loss_fn(trainer.loss_fn, backend="pennylane")

        weight_keys = list(model.weights.keys())
        current_weights = [
            pnp.array(model.weights[k], requires_grad=True) for k in weight_keys
        ]

        train_loader = datamodule.train_loader(shuffle=True)
        val_loader = datamodule.val_loader(shuffle=False)

        captured = {}

        def batch_cost(X_b, y_b, *weight_tensors):
            flat_kwargs = dict(zip(weight_keys, weight_tensors))
            preds = model.forward(X_b, **flat_kwargs)
            if preds.ndim == 0:
                preds = pnp.expand_dims(preds, axis=0)
            captured["preds"] = qml.math.unwrap(preds)
            return loss_fn(preds, y_b)

        opt = self._make_optimizer(trainer)

        self._emit(callbacks, "on_fit_start", state)

        with self.reporter.progress() as progress:
            for epoch in range(trainer.max_epochs):
                t0 = time.time()
                batch_losses = []
                correct = total = 0

                for X_batch, y_batch in train_loader:
                    X_b = pnp.array(X_batch, requires_grad=False)
                    y_b = pnp.array(y_batch, requires_grad=False)

                    args_out, batch_loss = opt.step_and_cost(
                        batch_cost, X_b, y_b, *current_weights
                    )
                    current_weights = list(args_out[2:])
                    batch_losses.append(float(batch_loss))

                    y_true = np.asarray(y_batch).astype(int).flatten()
                    correct += np.sum(_hard_labels(captured["preds"]) == y_true)
                    total += len(y_true)

                model.update_weights(dict(zip(weight_keys, current_weights)))

                train_loss = float(np.mean(batch_losses))
                train_acc = float(correct / total) if total else float("nan")
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
        """The ``qml`` optimizer named by ``trainer.optimizer``.

        The name is case-folded here rather than in ``Trainer.__init__``, which
        must store its arguments verbatim for skbase's ``get_params``/``clone``.
        """
        name = trainer.optimizer
        name = name.lower() if isinstance(name, str) else name
        if name == "adam":
            return qml.AdamOptimizer(stepsize=trainer.learning_rate)
        return qml.GradientDescentOptimizer(stepsize=trainer.learning_rate)

    @staticmethod
    def _evaluate(model, dataloader, loss_fn) -> tuple[float, float]:
        """Mean loss and hard-label accuracy over ``dataloader`` in one pass.

        Parameters
        ----------
        model : BaseModel
            Evaluated with its current weights.
        dataloader : iterable or None
            Yields ``(X, y)`` batches. ``None`` returns ``(nan, nan)``.
        loss_fn : callable
            Takes ``(preds, y)``.

        Returns
        -------
        tuple of (float, float)
            Mean loss and accuracy.
        """
        if dataloader is None:
            return float("nan"), float("nan")

        losses = []
        correct, total = 0, 0

        for X_b, y_b in dataloader:
            preds = model.forward(X_b)
            losses.append(float(loss_fn(preds, pnp.array(y_b, requires_grad=False))))

            y_true = y_b.astype(int).flatten()
            correct += np.sum(_hard_labels(preds) == y_true)
            total += len(y_true)

        loss = float(np.mean(losses)) if losses else float("nan")
        accuracy = float(correct / total) if total > 0 else float("nan")
        return loss, accuracy
