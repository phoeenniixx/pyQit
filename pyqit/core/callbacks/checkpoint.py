"""Callback tracking, saving and restoring the best epoch's weights."""

import os

import numpy as np

from pyqit.core.callbacks.base import BaseCallback, LoopState
from pyqit.utils.utils import _is_torch, _restore_weights, _snapshot_weights


class ModelCheckpoint(BaseCallback):
    """Track the best epoch, save its weights, and optionally restore them.

    Parameters
    ----------
    dirpath : str, optional
        Directory to write into. Defaults to ``"checkpoints"``.
    filename : str, default "best"
        Stem of the written file; the extension follows the backend.
    monitor : str, optional
        Metric to track. ``None`` picks ``"val_loss"`` when a validation split
        produced a finite value and ``"train_loss"`` otherwise, so a run without
        a validation split still checkpoints.
    mode : {"min", "max"}, default "min"
        Whether a lower or higher value of ``monitor`` is better.
    save_on_improve : bool, default False
        When False the file is written once, after training. Set True to write
        on every improvement, which costs extra I/O but survives a crash.
    restore_best : bool, default True
        Load the best weights back into the model when training ends.

    Attributes
    ----------
    best_score : float
        Best value of ``monitor`` seen.
    best_epoch : int
        Epoch it was seen on, or ``-1``.
    best_path : str or None
        Path written to, once anything has been written.
    """

    def __init__(
        self,
        dirpath: str | None = None,
        filename: str = "best",
        monitor: str | None = None,
        mode: str = "min",
        save_on_improve: bool = False,
        restore_best: bool = True,
    ):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}.")
        self.dirpath = dirpath
        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.save_on_improve = save_on_improve
        self.restore_best = restore_best
        super().__init__()

        self.best_score: float = float("inf") if mode == "min" else float("-inf")
        self.best_epoch: int = -1
        self.best_path: str | None = None
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

    def on_epoch_end(self, state: LoopState) -> None:
        """Snapshot the weights if this epoch improved on the best so far."""
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

        if self.save_on_improve:
            self.best_path = self._write(state.model, self._best_weights)
            state.reporter.success(
                f"New best ({monitor}: {score:.4f}) -> {self.best_path}",
                tag="Checkpoint",
            )

    def on_fit_end(self, state: LoopState) -> None:
        """Write the best weights out and restore them into the model."""
        weights = self._best_weights or _snapshot_weights(state.model)

        if not self.save_on_improve:
            self.best_path = self._write(state.model, weights)

        if self.restore_best and self._best_weights is not None:
            _restore_weights(state.model, self._best_weights)
            state.reporter.success(
                f"Restored best weights from epoch {self.best_epoch} "
                f"({self._monitor}: {self.best_score:.4f})",
                tag="Checkpoint",
            )

    def _write(self, model, weights: dict) -> str:
        """Serialize ``weights`` in the active backend's format; return the path."""
        directory = self.dirpath or "checkpoints"
        os.makedirs(directory, exist_ok=True)
        torch_backend = any(_is_torch(v) for v in model.weights.values())

        if torch_backend:
            import torch

            path = os.path.join(directory, f"{self.filename}.ckpt")
            torch.save(
                {"state_dict": {k: torch.as_tensor(v) for k, v in weights.items()}},
                path,
            )
            return path

        path = os.path.join(directory, f"{self.filename}.npz")
        np.savez(path, **{k: np.asarray(v) for k, v in weights.items()})
        return path
