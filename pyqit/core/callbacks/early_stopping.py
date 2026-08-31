"""Stop training once a monitored metric stops improving."""

import numpy as np

from pyqit.core.callbacks.base import BaseCallback, LoopState


class EarlyStopping(BaseCallback):
    """Request a stop after ``patience`` epochs without improvement.

    Parameters
    ----------
    monitor : str, optional
        Metric to watch. ``None`` picks ``"val_loss"`` when a validation split
        produced a finite value and ``"train_loss"`` otherwise.
    patience : int, default 5
        Epochs without improvement to tolerate before stopping. Counted the way
        ``lightning.pytorch.callbacks.EarlyStopping`` counts it (stop once the
        wait reaches ``patience``), so a given ``patience`` means the same
        number of epochs here as it does in Lightning.
    min_delta : float, default 0.0
        Improvement smaller than this does not count as an improvement.
    mode : {"min", "max"}, default "min"
    check_finite : bool, default True
        Stop as soon as the monitored metric is NaN or infinite. Matches
        Lightning's default, and matters on a quantum model because a diverged
        circuit yields NaN rather than a large loss.
    verbose : bool, default True
        Announce the stop through the run's reporter.

    Notes
    -----
    The stop takes effect at the end of an epoch, never mid-epoch, so both
    backends end on the same boundary. ``on_fit_end`` still runs, which is what
    lets a ``ModelCheckpoint`` installed alongside restore the best weights.

    This is a deliberate reimplementation rather than a wrapper: pennylane ships
    no callback or early-stopping machinery of any kind, so Lightning's version
    could only ever have covered one of the two backends.
    """

    def __init__(
        self,
        monitor: str | None = None,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
        check_finite: bool = True,
        verbose: bool = True,
    ):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}.")
        if patience < 0:
            raise ValueError(f"patience must be >= 0, got {patience}.")
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.check_finite = check_finite
        self.verbose = verbose
        super().__init__()

        self.stopped_epoch: int | None = None
        self.stopping_reason: str | None = None
        self.best_score: float = float("inf") if mode == "min" else float("-inf")
        self._monitor: str | None = monitor
        self._wait = 0

    def _resolve_monitor(self, metrics: dict) -> str:
        if self._monitor is not None:
            return self._monitor
        val = metrics.get("val_loss", float("nan"))
        self._monitor = "val_loss" if val == val else "train_loss"
        return self._monitor

    def _is_better(self, score: float) -> bool:
        if score != score:
            return False
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        return score > self.best_score + self.min_delta

    def on_epoch_end(self, state: LoopState) -> None:
        monitor = self._resolve_monitor(state.metrics)
        if monitor not in state.metrics:
            raise KeyError(
                f"EarlyStopping monitors {monitor!r}, which this run does not "
                f"produce. Available metrics: {sorted(state.metrics)}."
            )

        score = state.metrics[monitor]

        if self.check_finite and not np.isfinite(score):
            self._stop(state, f"{monitor} is {score}, which is not finite")
            return

        if self._is_better(score):
            self.best_score = score
            self._wait = 0
            return

        self._wait += 1
        # ``>=``, not ``>``: Lightning stops once the wait *reaches* patience, and
        # a patience meaning a different number of epochs in each framework is a
        # difference nobody would think to check for.
        if self._wait >= self.patience:
            self._stop(
                state,
                f"{monitor} did not improve for {self._wait} epoch(s) "
                f"(best {monitor}: {self.best_score:.4f})",
            )

    def _stop(self, state: LoopState, reason: str) -> None:
        self.stopped_epoch = state.epoch
        self.stopping_reason = reason
        state.stop = True
        if self.verbose:
            state.reporter.warn(
                f"Stopped at epoch {state.epoch} -- {reason}", tag="EarlyStopping"
            )
