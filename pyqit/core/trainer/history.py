"""Per-epoch metric record, shared by both backends."""


class TrainingHistory:
    """Metrics recorded once per epoch, backend-agnostic.

    Attributes
    ----------
    train_loss, val_loss, train_acc, val_acc : list of float
        One entry per completed epoch. ``val_*`` are NaN without a validation
        split.
    epoch_times : list of float
        Wall-clock seconds per epoch.
    best_epoch : int
        Epoch with the best score.
    best_score : float
        That epoch's value of ``best_metric``.
    best_metric : {"val_loss", "train_loss"}
        Which metric the best epoch was chosen on: ``val_loss`` when a
        validation split exists, ``train_loss`` otherwise. NaN loses every
        comparison, so monitoring ``val_loss`` unconditionally left a run
        without a validation split reporting ``inf @ epoch 0``.
    """

    def __init__(self):
        self.train_loss: list[float] = []
        self.val_loss: list[float] = []
        self.train_acc: list[float] = []
        self.val_acc: list[float] = []
        self.epoch_times: list[float] = []
        self.best_epoch: int = 0
        self.best_score: float = float("inf")
        self.best_metric: str = "val_loss"

    def record(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float = float("nan"),
        train_acc: float = 0.0,
        val_acc: float = 0.0,
        epoch_time: float = 0.0,
    ) -> None:
        """Append one epoch's metrics and update the running best.

        Parameters
        ----------
        epoch : int
            Zero-based epoch index.
        train_loss : float
            Mean training loss over the epoch.
        val_loss : float, default NaN
            Validation loss; NaN means the run has no validation split.
        train_acc, val_acc : float, default 0.0
            Accuracies for the epoch.
        epoch_time : float, default 0.0
            Wall-clock seconds the epoch took.
        """
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.train_acc.append(train_acc)
        self.val_acc.append(val_acc)
        self.epoch_times.append(epoch_time)

        score, metric = (
            (val_loss, "val_loss")
            if val_loss == val_loss  # NaN is the only value failing this
            else (train_loss, "train_loss")
        )
        if metric != self.best_metric:
            # A split appearing or vanishing mid-run makes the running best
            # incomparable; restart it against the metric now in use.
            self.best_metric = metric
            self.best_score = float("inf")
        if score < self.best_score:
            self.best_score = score
            self.best_epoch = epoch

    def as_dict(self) -> dict[str, list[float]]:
        """The recorded series, keyed by metric name."""
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
            f"best_{self.best_metric}={self.best_score:.4f} "
            f"@ epoch {self.best_epoch})"
        )
