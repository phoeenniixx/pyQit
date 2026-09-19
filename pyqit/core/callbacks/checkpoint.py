"""Callback saving model checkpoints during and after training.

A checkpoint file holds what other frameworks keep together with the weights:
the optimizer's state and the run's history to that epoch, so ``resume_from``
picks a run up where the file left it. What a run depends on beyond the model,
such as the fitted preprocessing in a ``DataModule``, is saved on request by
its own object (``DataModule.save``).
"""

import copy
import os
import warnings

import numpy as np

from pyqit.core.callbacks.base import BaseCallback, LoopState
from pyqit.utils.utils import (
    _is_torch,
    _restore_weights,
    _snapshot_weights,
    _to_numpy,
)

_HISTORY_KEYS = ("train_loss", "val_loss", "train_acc", "val_acc", "epoch_times")


def _optimizer_state(optimizer):
    """A detached copy of what the optimizer accumulates, on either backend.

    Torch gives one ``state_dict`` over its parameter groups; pennylane gives
    one ``qml`` optimizer per weight group, so its state is keyed by group.
    """
    if hasattr(optimizer, "state_dict"):
        return copy.deepcopy(optimizer.state_dict())
    state = {
        g: getattr(opt, "accumulation", None) for g, opt in (optimizer or {}).items()
    }
    state = {g: acc for g, acc in state.items() if acc is not None}
    return copy.deepcopy(state) if state else None


def _snapshot(state: LoopState, n_epochs: int | None = None) -> dict:
    """Weights, optimizer state and the first ``n_epochs`` rows of the history."""
    history = state.history.as_dict()
    return {
        "weights": _snapshot_weights(state.model),
        "optimizer": _optimizer_state(state.optimizer),
        "history": {k: list(v[:n_epochs]) for k, v in history.items()},
    }


def _write_checkpoint(path: str, snapshot: dict) -> None:
    if path.endswith(".ckpt"):
        import torch

        weights = {k: torch.as_tensor(v) for k, v in snapshot["weights"].items()}
        torch.save(
            {
                "state_dict": weights,
                "optimizer": snapshot["optimizer"],
                "history": snapshot["history"],
            },
            path,
        )
        return

    arrays = {f"weights/{k}": np.asarray(v) for k, v in snapshot["weights"].items()}
    for key, series in snapshot["history"].items():
        arrays[f"history/{key}"] = np.asarray(series, dtype=float)
    for group, accumulation in (snapshot["optimizer"] or {}).items():
        arrays[f"optimizer/{group}/t"] = accumulation["t"]
        for moment in ("fm", "sm"):
            for i, value in enumerate(accumulation[moment]):
                arrays[f"optimizer/{group}/{moment}{i}"] = np.asarray(value)
    np.savez(path, **arrays)


def _read_checkpoint(path: str) -> dict:
    """The ``weights``, ``optimizer`` and ``history`` a checkpoint file holds."""
    if path.endswith(".ckpt"):
        import torch

        data = torch.load(path, weights_only=False)
        return {
            "weights": {k: _to_numpy(v) for k, v in data["state_dict"].items()},
            "optimizer": data["optimizer"],
            "history": data["history"],
        }

    data = np.load(path)
    optimizer = {}
    for key in data.files:
        if key.startswith("optimizer/") and key.endswith("/t"):
            group = key[len("optimizer/") : -len("/t")]
            n = sum(k.startswith(f"optimizer/{group}/fm") for k in data.files)
            optimizer[group] = {
                "t": int(data[key]),
                "fm": [data[f"optimizer/{group}/fm{i}"] for i in range(n)],
                "sm": [data[f"optimizer/{group}/sm{i}"] for i in range(n)],
            }
    return {
        "weights": {
            k[len("weights/") :]: data[k]
            for k in data.files
            if k.startswith("weights/")
        },
        "optimizer": optimizer or None,
        "history": {
            k[len("history/") :]: data[k].tolist()
            for k in data.files
            if k.startswith("history/")
        },
    }


class ModelCheckpoint(BaseCallback):
    """Save checkpoints, restore the best epoch's weights, or resume from one.

    Three files can be written, independently: the best epoch (``save_best``),
    the final epoch (``save_last``), and a periodic snapshot
    (``every_n_epochs``). Each holds the weights, the optimizer's state and the
    history up to that epoch, so any of them resumes the run. The policy is
    backend-neutral; only serialization forks -- a ``.ckpt`` on torch, an
    ``.npz`` on pennylane. Weights are keyed by ``model.weights`` keys on both.

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
        Write the best epoch's checkpoint.
    save_last : bool, default False
        Write the final epoch's checkpoint. Written before any restore, so the
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
    resume_from : str, optional
        A checkpoint written by this callback. Before the first epoch its
        weights are loaded into the model, its history into the run's, so
        training continues from the next epoch within ``max_epochs``, and its
        optimizer state into the optimizer the loop builds. A file from the
        other backend restores weights and history only, with a warning, since
        optimizer state does not transfer.

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

    Notes
    -----
    A file from the other backend restores the weights and history and warns
    that the optimizer starts fresh, since its state does not transfer.
    Callbacks that track improvement, `EarlyStopping` and the best epoch here,
    start their count over on a resumed run. The fitted preprocessing is not in
    the file; it is the DataModule's own artifact, saved with
    `DataModule.save` when wanted.

    Examples
    --------
    Save the last epoch, then pick the run up from it two epochs later:

    >>> import pyqit
    >>> from pyqit.core import ModelCheckpoint
    >>> saving = ModelCheckpoint(dirpath="ckpts", save_best=False, save_last=True)
    >>> pyqit.Trainer(max_epochs=2, callbacks=[saving]).fit(model, dm)  # doctest: +SKIP
    >>> resuming = ModelCheckpoint(save_best=False, resume_from="ckpts/last.npz")
    >>> history = pyqit.Trainer(max_epochs=4, callbacks=[resuming]).fit(
    ...     model, dm
    ... )  # doctest: +SKIP

    The history's length is the next epoch, so this trains epochs 2 and 3, and
    Adam keeps its moment estimates.
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
        resume_from: str | None = None,
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
        self.resume_from = resume_from
        super().__init__()

        self._restore_best = save_best if restore_best is None else restore_best

        does_nothing = not (save_best or save_last or every_n_epochs or resume_from)
        if does_nothing and not self._restore_best:
            raise ValueError(
                "ModelCheckpoint would do nothing: it writes no file and does "
                "not restore. Set save_best, save_last, every_n_epochs, "
                "restore_best or resume_from."
            )

        self.best_score: float = float("inf") if mode == "min" else float("-inf")
        self.best_epoch: int = -1
        self.best_path: str | None = None
        self.last_path: str | None = None
        self.periodic_paths: list[str] = []
        self._monitor: str | None = monitor
        self._best: dict | None = None

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

    def on_fit_start(self, state: LoopState) -> None:
        """Load ``resume_from`` into the model, the history and the optimizer."""
        if self.resume_from is None:
            return
        checkpoint = _read_checkpoint(self.resume_from)
        if not checkpoint["weights"]:
            raise ValueError(
                f"{self.resume_from} holds no weights; it was not written by "
                "ModelCheckpoint, or predates its current format."
            )
        _restore_weights(state.model, checkpoint["weights"])
        for epoch, row in enumerate(
            zip(*(checkpoint["history"][k] for k in _HISTORY_KEYS))
        ):
            state.history.record(epoch, *row)

        torch_file = self.resume_from.endswith(".ckpt")
        torch_model = any(_is_torch(v) for v in state.model.weights.values())
        if torch_file != torch_model:
            warnings.warn(
                f"{self.resume_from} was written by the other backend: its "
                "weights and history are restored, but optimizer state does not "
                "transfer, so the optimizer starts fresh.",
                UserWarning,
                stacklevel=2,
            )
        else:
            state.optimizer_state = checkpoint["optimizer"]
        state.reporter.success(
            f"Resumed from {self.resume_from} at epoch "
            f"{len(state.history.train_loss)}",
            tag="Checkpoint",
        )

    def on_epoch_end(self, state: LoopState) -> None:
        """Track the best epoch and write any periodic snapshot."""
        if self.every_n_epochs and (state.epoch + 1) % self.every_n_epochs == 0:
            path = self._write(
                state.model, _snapshot(state, state.epoch + 1), f"epoch{state.epoch}"
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
        self._best = _snapshot(state, state.epoch + 1)

        if self.save_best and self.save_on_improve:
            self.best_path = self._write(state.model, self._best, self.filename)
            state.reporter.success(
                f"New best ({monitor}: {score:.4f}) -> {self.best_path}",
                tag="Checkpoint",
            )

    def on_fit_end(self, state: LoopState) -> None:
        """Write the requested files, then restore the best weights."""
        if self.save_last:
            self.last_path = self._write(state.model, _snapshot(state), "last")
            state.reporter.success(f"Last epoch -> {self.last_path}", tag="Checkpoint")

        if self.save_best and not self.save_on_improve:
            snapshot = self._best or _snapshot(state)
            self.best_path = self._write(state.model, snapshot, self.filename)

        if self._restore_best and self._best is not None:
            _restore_weights(state.model, self._best["weights"])
            state.reporter.success(
                f"Restored best weights from epoch {self.best_epoch} "
                f"({self._monitor}: {self.best_score:.4f})",
                tag="Checkpoint",
            )

    def _write(self, model, snapshot: dict, stem: str) -> str:
        """Serialize ``snapshot`` in the active backend's format; return the path."""
        directory = self.dirpath or "checkpoints"
        os.makedirs(directory, exist_ok=True)
        torch_backend = any(_is_torch(v) for v in model.weights.values())
        path = os.path.join(
            directory, f"{stem}.ckpt" if torch_backend else f"{stem}.npz"
        )
        _write_checkpoint(path, snapshot)
        return path

    @classmethod
    def get_test_params(cls):
        """List constructor kwargs used to parametrize this class in the test suite."""
        return [{}, {"save_best": False, "save_last": True}]
