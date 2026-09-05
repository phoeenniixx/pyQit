"""Callback saving model weights during and after training."""

import os

import numpy as np

from pyqit.core.callbacks.base import BaseCallback, LoopState
from pyqit.utils.utils import _is_torch, _restore_weights, _snapshot_weights


class ModelCheckpoint(BaseCallback):
    """Save model weights, and optionally restore the best epoch's.

    Three things can be written, independently: the best epoch (``save_best``),
    the final epoch (``save_last``), and a periodic snapshot
    (``every_n_epochs``). The policy is backend-neutral; only serialization
    forks -- a ``.ckpt`` holding a ``state_dict`` on torch, an ``.npz`` on
    pennylane. Array keys are ``model.weights`` keys on both.

    Parameters
    ----------
    dirpath : str, optional
        Directory to write into. Defaults to ``"checkpoints"``.
    filename : str, default "best"
        Stem of the best-epoch file. The other two have fixed stems, ``last``
        and ``epoch<n>``, so the three never collide.
    monitor : str, optional
        Metric deciding which epoch is best. ``None`` picks ``"val_loss"`` when
        a validation split produced a finite value and ``"train_loss"``
        otherwise, so a run without a validation split still checkpoints.
    mode : {"min", "max"}, default "min"
        Whether a lower or higher value of ``monitor`` is better.
    save_best : bool, default True
        Write the best epoch's weights.
    save_last : bool, default False
        Write the final epoch's weights. Written before any restore, so the
        file holds the last epoch even when ``restore_best`` is on.
    every_n_epochs : int, optional
        Also write a snapshot every N epochs, named by the zero-based epoch
        index to match ``best_epoch``. With ``every_n_epochs=5`` that is
        ``epoch4``, ``epoch9``, and so on.
    save_on_improve : bool, default False
        Write the best file on every improvement rather than once after
        training. Costs extra I/O but survives a crash mid-run. Ignored when
        ``save_best`` is False.
    restore_best : bool, optional
        Load the best weights back into the model when training ends. Defaults
        to ``save_best``, so asking only for the last epoch does not silently
        hand back the best one.

    Attributes
    ----------
    best_score : float
        Best value of ``monitor`` seen.
    best_epoch : int
        Zero-based epoch it was seen on, or ``-1``.
    best_path, last_path : str or None
        Paths written, once anything has been.
    periodic_paths : list of str
        Paths written by ``every_n_epochs``, in order.
    """

    def __init__(
        self,
        dirpath: str | None = None,
        filename: str = "best",
        monitor: str | None = None,
        mode: str = "min",
        save_best: bool = True,
        save_last: bool = False,
        every_n_epochs: int | None = None,
        save_on_improve: bool = False,
        restore_best: bool | None = None,
    ):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}.")
        if every_n_epochs is not None and every_n_epochs < 1:
            raise ValueError(
                f"every_n_epochs must be >= 1 or None, got {every_n_epochs}."
            )
        self.dirpath = dirpath
        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.save_best = save_best
        self.save_last = save_last
        self.every_n_epochs = every_n_epochs
        self.save_on_improve = save_on_improve
        self.restore_best = restore_best
        super().__init__()

        self._restore_best = save_best if restore_best is None else restore_best

        if not (save_best or save_last or every_n_epochs) and not self._restore_best:
            raise ValueError(
                "ModelCheckpoint would do nothing: it writes no file and does "
                "not restore. Set save_best, save_last, every_n_epochs or "
                "restore_best."
            )

        self.best_score: float = float("inf") if mode == "min" else float("-inf")
        self.best_epoch: int = -1
        self.best_path: str | None = None
        self.last_path: str | None = None
        self.periodic_paths: list[str] = []
        self._monitor: str | None = monitor
        self._best_weights: dict | None = None

    def _is_better(self, score: float) -> bool:
        if score != score:  # NaN never improves on anything
            return False
        return (
            score < self.best_score if self.mode == "min" else score > self.best_score
        )

    def _resolve_monitor(self, metrics: dict) -> str:
        """The metric to track, chosen once on the first epoch."""
        if self._monitor is not None:
            return self._monitor
        val = metrics.get("val_loss", float("nan"))
        self._monitor = "val_loss" if val == val else "train_loss"
        return self._monitor

    def _tracks_best(self) -> bool:
        return self.save_best or self._restore_best

    def on_epoch_end(self, state: LoopState) -> None:
        """Track the best epoch and write any periodic snapshot."""
        if self.every_n_epochs and (state.epoch + 1) % self.every_n_epochs == 0:
            path = self._write(
                state.model, _snapshot_weights(state.model), f"epoch{state.epoch}"
            )
            self.periodic_paths.append(path)
            state.reporter.success(f"Epoch {state.epoch} -> {path}", tag="Checkpoint")

        if not self._tracks_best():
            return

        monitor = self._resolve_monitor(state.metrics)
        if monitor not in state.metrics:
            raise KeyError(
                f"ModelCheckpoint monitors {monitor!r}, which this run does not "
                f"produce. Available metrics: {sorted(state.metrics)}."
            )

        score = state.metrics[monitor]
        if not self._is_better(score):
            return

        self.best_score = score
        self.best_epoch = state.epoch
        self._best_weights = _snapshot_weights(state.model)

        if self.save_best and self.save_on_improve:
            self.best_path = self._write(state.model, self._best_weights, self.filename)
            state.reporter.success(
                f"New best ({monitor}: {score:.4f}) -> {self.best_path}",
                tag="Checkpoint",
            )

    def on_fit_end(self, state: LoopState) -> None:
        """Write the requested files, then restore the best weights."""
        if self.save_last:
            self.last_path = self._write(
                state.model, _snapshot_weights(state.model), "last"
            )
            state.reporter.success(f"Last epoch -> {self.last_path}", tag="Checkpoint")

        if self.save_best and not self.save_on_improve:
            weights = self._best_weights or _snapshot_weights(state.model)
            self.best_path = self._write(state.model, weights, self.filename)

        if self._restore_best and self._best_weights is not None:
            _restore_weights(state.model, self._best_weights)
            state.reporter.success(
                f"Restored best weights from epoch {self.best_epoch} "
                f"({self._monitor}: {self.best_score:.4f})",
                tag="Checkpoint",
            )

    def _write(self, model, weights: dict, stem: str) -> str:
        """Serialize ``weights`` in the active backend's format; return the path."""
        directory = self.dirpath or "checkpoints"
        os.makedirs(directory, exist_ok=True)
        torch_backend = any(_is_torch(v) for v in model.weights.values())

        if torch_backend:
            import torch

            path = os.path.join(directory, f"{stem}.ckpt")
            torch.save(
                {"state_dict": {k: torch.as_tensor(v) for k, v in weights.items()}},
                path,
            )
            return path

        path = os.path.join(directory, f"{stem}.npz")
        np.savez(path, **{k: np.asarray(v) for k, v in weights.items()})
        return path

    @classmethod
    def get_test_params(cls):
        """List constructor kwargs used to parametrize this class in the test suite."""
        return [{}, {"save_best": False, "save_last": True}]
