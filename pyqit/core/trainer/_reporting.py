"""Console output for training runs.

Every rich / plaintext fork in the package lives here.  The loops call a
``Reporter`` and never test ``HAS_RICH`` themselves, so the two backends cannot
drift into printing different things.
"""

from contextlib import contextmanager
import logging
import sys

import numpy as np
from skbase.utils.dependencies import _check_soft_dependencies

from pyqit.utils.utils import _count_params

HAS_RICH = _check_soft_dependencies("rich", severity="none")
if HAS_RICH:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        Progress,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    from rich.table import Table

    console = Console()
else:  # keep the name importable so callers need no second guard
    console = None


@contextmanager
def lightning_log_level(level):
    """Temporarily raise Lightning's logger level, restoring it afterwards."""
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
        Used to size the progress bar.
    show_summary : bool, default True
        Cleared by ``QuantumPipeline``, which prints one summary naming every
        stage in place of one table per stage.
    """

    def __init__(self, verbose: int, max_epochs: int, show_summary: bool = True):
        self.verbose = verbose
        self.max_epochs = max_epochs
        self.show_summary = show_summary

    @property
    def quiet(self) -> bool:
        return self.verbose < 1

    @property
    def summary_suppressed(self) -> bool:
        """Whether the run summary is being withheld, for any reason.

        Lightning's own banner is silenced on exactly this condition, so the two
        cannot disagree about whether a run is meant to be quiet.
        """
        return not self.show_summary or self.verbose < 2

    def _emit(self, rich_markup: str, plain: str) -> None:
        if HAS_RICH:
            console.print(rich_markup)
        else:
            print(plain)

    def info(self, msg: str, tag: str = "Trainer") -> None:
        if self.verbose < 1:
            return
        self._emit(f"[bold cyan][{tag}][/bold cyan] {msg}", f"[{tag}] {msg}")

    def success(self, msg: str, tag: str = "Trainer") -> None:
        if self.verbose < 1:
            return
        self._emit(f"[bold green][{tag}][/bold green] {msg}", f"[{tag}] {msg}")

    def warn(self, msg: str, tag: str = "Trainer") -> None:
        if self.verbose < 1:
            return
        self._emit(f"[bold yellow][{tag}][/bold yellow] {msg}", f"[{tag}] {msg}")

    def banner(self, backend: str, learning_rate: float) -> None:
        """One-line run header, used when the full table is not wanted."""
        if self.verbose < 1:
            return
        if HAS_RICH:
            console.print(
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
        """Architecture and split sizes, once per fit."""
        if self.summary_suppressed:
            return
        if not HAS_RICH:
            self.banner(backend, lr)
            return

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

        console.print(table)
        console.print()

    @contextmanager
    def progress(self):
        """Yield an object with ``update(epoch, train_loss, val_loss, elapsed)``."""
        if self.verbose < 1:
            yield _NullProgress()
        elif HAS_RICH:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(complete_style="cyan", finished_style="bold green"),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task_id = progress.add_task("[cyan]Training...", total=self.max_epochs)
                yield _RichProgress(progress, task_id, self.max_epochs)
        else:
            yield _PlainProgress(self.max_epochs)


def _obj_name(model, attr: str) -> str:
    obj = getattr(model, attr, None)
    return type(obj).__name__ if obj is not None else "N/A"
