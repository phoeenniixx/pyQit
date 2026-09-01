"""Console output for training runs."""

from collections.abc import Iterator
from contextlib import contextmanager
import logging
import sys

import numpy as np
from skbase.utils.dependencies import _check_soft_dependencies

from pyqit.utils.utils import _count_params

_CONSOLE = None


def has_rich() -> bool:
    """Whether ``rich`` is installed."""
    return _check_soft_dependencies("rich", severity="none")


def console():
    """The shared ``rich.Console``, or None when rich is not installed."""
    global _CONSOLE
    if _CONSOLE is None and has_rich():
        from rich.console import Console

        _CONSOLE = Console()
    return _CONSOLE


@contextmanager
def lightning_log_level(level: int) -> Iterator[None]:
    """Raise Lightning's logger level for the duration of the block.

    Parameters
    ----------
    level : int
        A ``logging`` level applied to the ``lightning.pytorch`` logger.
    """
    lit_logger = logging.getLogger("lightning.pytorch")
    previous = lit_logger.level
    lit_logger.setLevel(level)
    try:
        yield
    finally:
        lit_logger.setLevel(previous)


class _NullProgress:
    """Progress handle used when there is nothing to draw."""

    def update(self, epoch, train_loss, val_loss, elapsed):
        pass

    def finish(self):
        pass


class _RichProgress:
    """Progress handle backed by a ``rich.progress.Progress`` task."""

    def __init__(self, progress, task_id, max_epochs):
        self._progress = progress
        self._task_id = task_id
        self._max_epochs = max_epochs

    def update(self, epoch, train_loss, val_loss, elapsed):
        val_str = f"| Val Loss: {val_loss:.4f}" if not np.isnan(val_loss) else ""
        self._progress.update(
            self._task_id,
            completed=epoch + 1,
            description=(
                f"[cyan]Epoch {epoch + 1}/{self._max_epochs} "
                f"| Loss: {train_loss:.4f} {val_str}"
            ),
        )

    def finish(self):
        self._progress.refresh()


class _PlainProgress:
    """ASCII progress bar used when rich is not installed."""

    BAR_LEN = 30

    def __init__(self, max_epochs):
        self._max_epochs = max_epochs

    def update(self, epoch, train_loss, val_loss, elapsed):
        val_str = f" | val_loss={val_loss:.4f}" if not np.isnan(val_loss) else ""
        percent = (epoch + 1) / self._max_epochs
        filled = int(round(self.BAR_LEN * percent))
        bar = (
            "=" * max(0, filled - 1)
            + ">" * min(1, filled)
            + "." * (self.BAR_LEN - filled)
        )
        width = len(str(self._max_epochs))
        sys.stdout.write(
            f"\rEpoch {epoch + 1:>{width}}/{self._max_epochs} "
            f"[{bar}] {percent:.0%} | loss={train_loss:.4f}{val_str} "
            f"[{elapsed:.1f}s]  "
        )
        sys.stdout.flush()

    def finish(self):
        sys.stdout.write("\n")
        sys.stdout.flush()


class Reporter:
    """All user-facing output for one training run.

    Parameters
    ----------
    verbose : int
        ``0`` silent, ``1`` progress only, ``2`` progress plus the model table.
    max_epochs : int
        Sizes the progress bar.
    show_summary : bool, default True
        Whether the per-run model table may be printed. ``QuantumPipeline``
        clears it and prints one summary covering every stage.
    """

    def __init__(self, verbose: int, max_epochs: int, show_summary: bool = True):
        self.verbose = verbose
        self.max_epochs = max_epochs
        self.show_summary = show_summary

    @property
    def quiet(self) -> bool:
        """Whether all output is suppressed."""
        return self.verbose < 1

    @property
    def summary_suppressed(self) -> bool:
        """Whether the run summary is withheld, for any reason."""
        return not self.show_summary or self.verbose < 2

    def _emit(self, rich_markup: str, plain: str) -> None:
        if has_rich():
            console().print(rich_markup)
        else:
            print(plain)

    def info(self, msg: str, tag: str = "Trainer") -> None:
        """Print a neutral message."""
        if self.verbose < 1:
            return
        self._emit(f"[bold cyan][{tag}][/bold cyan] {msg}", f"[{tag}] {msg}")

    def success(self, msg: str, tag: str = "Trainer") -> None:
        """Print a success message."""
        if self.verbose < 1:
            return
        self._emit(f"[bold green][{tag}][/bold green] {msg}", f"[{tag}] {msg}")

    def warn(self, msg: str, tag: str = "Trainer") -> None:
        """Print a warning message."""
        if self.verbose < 1:
            return
        self._emit(f"[bold yellow][{tag}][/bold yellow] {msg}", f"[{tag}] {msg}")

    def banner(self, backend: str, learning_rate: float) -> None:
        """Print a one-line run header, used in place of the full table."""
        if self.verbose < 1:
            return
        if has_rich():
            console().print(
                f"[bold cyan][Trainer][/bold cyan] Starting "
                f"[green]{backend}[/green] backend | "
                f"{self.max_epochs} epochs | lr={learning_rate}\n"
            )
        else:
            print(
                f"[Trainer] Starting {backend} backend | "
                f"{self.max_epochs} epochs | lr={learning_rate}"
            )

    def model_summary(self, model, dm, backend: str, optimizer: str, lr: float) -> None:
        """Print the architecture and split sizes, once per fit.

        Parameters
        ----------
        model : BaseModel
            Supplies the qubit count, ansatz, encoder and parameter count.
        dm : DataModule
            Must already be set up; supplies the split sizes.
        backend : str
            Active backend name.
        optimizer : str
            Optimizer name, as passed to the Trainer.
        lr : float
            Learning rate.
        """
        if self.summary_suppressed:
            return
        if not has_rich():
            self.banner(backend, lr)
            return

        from rich.table import Table

        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("Parameter", style="dim", width=20)
        table.add_column("Value", style="bold")

        table.add_row("Model Name", type(model).__name__)
        table.add_row("Backend", backend.capitalize())
        table.add_row("Qubits", str(getattr(model, "n_qubits", "N/A")))
        table.add_row("Ansatz", _obj_name(model, "ansatz_obj"))
        table.add_row("Encoder", _obj_name(model, "embedding_obj"))

        n_params = _count_params(model)
        table.add_row("Trainable Params", str(n_params) if n_params else "0")
        table.add_row("Optimizer", optimizer.upper())
        table.add_row("Learning Rate", str(lr))

        train_samples = len(dm.X_train) if dm.X_train is not None else 0
        val_samples = len(dm.X_val) if dm.X_val is not None else 0
        table.add_row("Train / Val Samples", f"{train_samples} / {val_samples}")

        console().print(table)
        console().print()

    @contextmanager
    def progress(self) -> Iterator[object]:
        """Yield a handle with ``update(epoch, train_loss, val_loss, elapsed)``."""
        if self.verbose < 1:
            yield _NullProgress()
        elif has_rich():
            from rich.progress import (
                BarColumn,
                Progress,
                TaskProgressColumn,
                TextColumn,
                TimeElapsedColumn,
            )

            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(complete_style="cyan", finished_style="bold green"),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console(),
            ) as progress:
                task_id = progress.add_task("[cyan]Training...", total=self.max_epochs)
                yield _RichProgress(progress, task_id, self.max_epochs)
        else:
            yield _PlainProgress(self.max_epochs)


def _obj_name(model, attr: str) -> str:
    obj = getattr(model, attr, None)
    return type(obj).__name__ if obj is not None else "N/A"
