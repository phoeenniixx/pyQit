"""Stop training once a monitored metric stops improving."""

from pyqit.core.callbacks.base import BaseCallback, LoopState


class EarlyStopping(BaseCallback):
    """Request a stop after ``patience`` epochs without improvement.

    Parameters
    ----------
    monitor : str, optional
        Metric to watch. ``None`` picks ``"val_loss"`` when a validation split
        produced a finite value and ``"train_loss"`` otherwise.
    patience : int, default 5
        Epochs to wait after the last improvement before stopping.
    min_delta : float, default 0.0
        Improvement smaller than this does not count as an improvement.
    mode : {"min", "max"}, default "min"
    verbose : bool, default True
        Announce the stop through the run's reporter.

    Notes
    -----
    The stop takes effect at the end of an epoch, never mid-epoch, so both
    backends end on the same boundary. ``on_fit_end`` still runs, which is what
    lets a ``ModelCheckpoint`` installed alongside restore the best weights.
    """

    def __init__(
        self,
        monitor: str | None = None,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
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
        self.verbose = verbose
        super().__init__()

        self.stopped_epoch: int | None = None
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
        if self._is_better(score):
            self.best_score = score
            self._wait = 0
            return

        self._wait += 1
        if self._wait > self.patience:
            self.stopped_epoch = state.epoch
            state.stop = True
            if self.verbose:
                state.reporter.warn(
                    f"Stopped at epoch {state.epoch} -- {monitor} did not improve "
                    f"for {self.patience} epoch(s) "
                    f"(best {monitor}: {self.best_score:.4f})",
                    tag="EarlyStopping",
                )
