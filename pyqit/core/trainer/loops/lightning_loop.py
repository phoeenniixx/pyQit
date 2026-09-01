"""Torch backend loop: delegates to ``lightning.pytorch.Trainer``."""

import logging

from pyqit.core.trainer.loops.base import BaseTrainingLoop


class LightningLoop(BaseTrainingLoop):
    """Run the fit through a real Lightning Trainer.

    None of the training mechanics are reimplemented; the loop translates
    pyqit's Trainer settings into Lightning's constructor, forwards
    ``backend_kwargs`` unchanged, and bridges pyqit callbacks onto Lightning's
    hooks.
    """

    _tags = {
        "object_type": "training_loop",
        "backend": "torch",
        "rejects": (),
        "warns": (),
        # Derived from Trainer settings below.
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

    DEFAULT_ACCELERATOR = "cpu"

    def fit(self, model, datamodule, state, callbacks: list) -> None:
        """Fit through Lightning. See ``BaseTrainingLoop.fit``."""
        from contextlib import nullcontext

        from skbase.utils.dependencies import _check_soft_dependencies

        if not _check_soft_dependencies(["lightning", "torch"], severity="none"):
            raise ImportError(
                "The PyTorch backend requires both lightning and torch, and at "
                "least one is not installed. Install them with "
                "`pip install pyqit[all_extras]`."
            )

        from lightning.pytorch import Trainer as LightningTrainer

        from pyqit.core.adapters.lightning import (
            _LightningModelAdapter,
            _PyQitCallbackShim,
        )
        from pyqit.core.losses import get_loss_fn
        from pyqit.core.trainer._reporting import lightning_log_level

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
        log_ctx = lightning_log_level(logging.WARNING) if quiet else nullcontext()

        with log_ctx:
            pl_trainer = LightningTrainer(
                max_epochs=trainer.max_epochs,
                callbacks=[_PyQitCallbackShim(callbacks, state)],
                logger=trainer.logger,
                enable_progress_bar=(self.reporter.verbose >= 1),
                enable_model_summary=False,
                # pyqit's ModelCheckpoint owns checkpointing on both backends;
                # leaving this on would write the run out twice.
                enable_checkpointing=False,
                # val_dataloader() returns None without a validation split,
                # which Lightning rejects outright.
                limit_val_batches=0 if not has_val else 1.0,
                **extra,
            )
            pl_trainer.fit(pl_model, datamodule=pl_data)

        self.reporter.success("Training complete.")
