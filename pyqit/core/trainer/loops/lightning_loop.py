"""Torch backend loop: a thin delegation to a real Lightning Trainer.

Nothing about the training mechanics is reimplemented here. The loop's whole
job is translating pyqit's Trainer settings into Lightning's constructor,
forwarding anything else the user asked for verbatim, and bridging pyqit
callbacks onto Lightning's hooks.
"""

import logging

from skbase.utils.dependencies import _check_soft_dependencies

from pyqit.core.losses import get_loss_fn
from pyqit.core.trainer._reporting import lightning_log_level
from pyqit.core.trainer.loops.base import BaseTrainingLoop


class LightningLoop(BaseTrainingLoop):
    """Delegate fitting to ``lightning.pytorch.Trainer``.

    Every Lightning setting pyqit does not derive itself is reachable through
    ``Trainer(backend_kwargs=...)``, which is forwarded unchanged. Mirroring
    Lightning's forty-odd constructor parameters onto pyqit's Trainer would
    have meant maintaining a copy that goes stale on the next Lightning
    release, and would still not cover settings added after this was written.
    """

    _tags = {
        "object_type": "training_loop",
        "backend": "torch",
        "rejects": {},
        "warns": {},
        "reserved_backend_kwargs": (
            "max_epochs",
            "callbacks",
            "logger",
            "enable_checkpointing",
            "enable_progress_bar",
            "enable_model_summary",
            "limit_val_batches",
        ),
    }

    #: Lightning defaults to "auto" and would silently start using a GPU for
    #: users who never asked; pyqit's own default has always been CPU.
    DEFAULT_ACCELERATOR = "cpu"

    def fit(self, model, datamodule, state, callbacks) -> None:
        if not _check_soft_dependencies("lightning", severity="none"):
            raise ImportError(
                "Lightning is not installed. "
                "Please install it to use the PyTorch backend."
            )

        from lightning.pytorch import Trainer as LightningTrainer

        from pyqit.core.adapters.lightning import (
            _LightningModelAdapter,
            _PyQitCallbackShim,
        )

        trainer = self.trainer
        extra = self._resolve_backend_kwargs()
        extra.setdefault("accelerator", self.DEFAULT_ACCELERATOR)

        loss_func = get_loss_fn(trainer.loss_fn, backend="torch")
        pl_model = _LightningModelAdapter(
            model, trainer.learning_rate, trainer.optimizer, loss_func
        )
        pl_data = datamodule.to_lightning()
        has_val = datamodule.X_val is not None

        quiet = self.reporter.summary_suppressed
        log_ctx = lightning_log_level(logging.WARNING) if quiet else _null_ctx()

        with log_ctx:
            pl_trainer = LightningTrainer(
                max_epochs=trainer.max_epochs,
                callbacks=[_PyQitCallbackShim(callbacks, state)],
                logger=trainer.logger,
                enable_progress_bar=(self.reporter.verbose >= 1),
                enable_model_summary=False,
                # pyqit's own ModelCheckpoint callback owns checkpointing on both
                # backends; leaving this on would add Lightning's default
                # checkpointer alongside it and write the run out twice.
                enable_checkpointing=False,
                # val_dataloader() returns None without a validation split, which
                # Lightning rejects outright; this is its supported way of saying
                # the run has no validation loop.
                limit_val_batches=0 if not has_val else 1.0,
                **extra,
            )
            pl_trainer.fit(pl_model, datamodule=pl_data)

        self.reporter.success("Training complete.")


def _null_ctx():
    from contextlib import nullcontext

    return nullcontext()
