"""The Trainer: configuration, orchestration, and backend dispatch."""

from collections.abc import Callable
import logging

import numpy as np
import pennylane.numpy as pnp
from skbase.base import BaseMetaObject

from pyqit.base.base_object import _PyQitObject
from pyqit.core.callbacks import HistoryCallback, LoopState, ModelCheckpoint
from pyqit.core.config import get_backend, set_seed
from pyqit.core.trainer._reporting import Reporter
from pyqit.core.trainer.history import TrainingHistory
from pyqit.core.trainer.loops import get_training_loop
from pyqit.data.datamodule import DataModule
from pyqit.models.base.base import BaseModel


class Trainer(_PyQitObject):
    """Orchestrates a training run and dispatches it to a backend loop.

    The Trainer itself trains nothing. It seeds, sets the DataModule up, prints
    the run summary, optionally runs the barren-plateau pre-flight, assembles
    the callback list, and hands off to the loop registered for the active
    backend.

    Its parameters are the ones that mean the same thing on every backend.
    Anything backend-specific goes through ``backend_kwargs``, which the torch
    loop forwards verbatim to ``lightning.pytorch.Trainer``. Mirroring
    Lightning's own constructor here was considered and rejected: Lightning
    carries roughly forty parameters of which a pennylane optimizer loop can
    honour under a quarter, and it deliberately holds none of ``learning_rate``,
    ``optimizer``, ``loss_fn`` or ``batch_size``, which live on its
    LightningModule and DataModule instead.

    Parameters
    ----------
    max_epochs : int, default 30
    learning_rate : float, default 0.01
    batch_size : int, default 32
        Applied to the DataModule at ``setup``, overriding its own setting.
    optimizer : {"adam", "sgd"}, default "adam"
    loss_fn : str or callable, default "mse"
        A name from ``loss_registry()``, or a callable taking ``(preds, y)``.
    callbacks : list of BaseCallback, optional
        Run on both backends. Lightning callbacks are not accepted here; see
        ``pyqit.core.callbacks``.
    verbose : {0, 1, 2}, default 2
        ``0`` silent, ``1`` progress only, ``2`` progress plus the model table.
    seed : int, optional
        Seeded before anything stochastic runs. Weights are drawn at model
        construction, so reproducing them needs ``pyqit.set_seed`` before the
        model is built; this covers training and diagnostics only.
    enable_checkpointing : bool, default False
        Installs a default ``ModelCheckpoint``. For any non-default policy,
        pass your own through ``callbacks`` instead.
    checkpoint_dir : str, optional
        Directory for the default checkpoint. Defaults to ``"checkpoints"``.
    logger : bool or object, default False
        Forwarded to Lightning on the torch backend. The pennylane backend has
        no logger and warns if this is set.
    check_bp : bool, default False
        Run the barren-plateau gradient-variance check before training.
    bp_samples : int, default 200
    eval_train_acc : bool, default True
        Evaluate training accuracy each epoch on the pennylane backend. This
        costs a full extra forward pass over the training split per epoch, which
        on a circuit-bound backend is a large fraction of the epoch; set False
        to trade the ``train_acc`` history for the speed. Ignored on torch,
        where training accuracy comes free from the training batches.
    backend_kwargs : dict, optional
        Forwarded verbatim to the backend's underlying trainer. On torch that is
        ``lightning.pytorch.Trainer`` (so ``{"accelerator": "gpu"}``,
        ``{"gradient_clip_val": 0.5}``, and anything else Lightning accepts).
        The pennylane backend has no underlying trainer and raises if this is set.
    """

    _tags = {
        "object_type": "trainer",
    }

    def __init__(
        self,
        max_epochs: int = 30,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        optimizer: str = "adam",
        loss_fn: str | Callable = "mse",
        callbacks: list | None = None,
        verbose: int = 2,
        seed: int | None = 42,
        enable_checkpointing: bool = False,
        checkpoint_dir: str | None = None,
        logger: bool | object = False,
        check_bp: bool = False,
        bp_samples: int = 200,
        eval_train_acc: bool = True,
        backend_kwargs: dict | None = None,
    ):
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.callbacks = callbacks
        self.verbose = verbose
        self.seed = seed
        self.enable_checkpointing = enable_checkpointing
        self.checkpoint_dir = checkpoint_dir
        self.logger = logger
        self.check_bp = check_bp
        self.bp_samples = bp_samples
        self.eval_train_acc = eval_train_acc
        self.backend_kwargs = backend_kwargs
        super().__init__()

        # Cached at construction, matching every other pyqit object: setting the
        # backend after building a Trainer does not retarget it.
        self.backend = get_backend()
        self._print_summary = True

    def fit(self, model: BaseModel, datamodule: DataModule) -> TrainingHistory:
        """Train ``model`` on ``datamodule`` and return the epoch history."""
        if self.seed is not None:
            set_seed(self.seed)

        reporter = Reporter(
            verbose=self.verbose,
            max_epochs=self.max_epochs,
            show_summary=self._print_summary,
        )
        # Built before any data is touched: constructing the loop validates this
        # Trainer's settings against what the backend can honour, and an
        # unsupported setting should fail before a split or a fit is run.
        loop = get_training_loop(self.backend, trainer=self, reporter=reporter)

        datamodule.setup(
            stage="fit",
            batch_size=self.batch_size,
            n_qubits=getattr(model, "n_qubits", None),
            encoder=self._encoder_class(model),
        )

        if self._print_summary:
            if self.verbose >= 2:
                reporter.model_summary(
                    model, datamodule, self.backend, self.optimizer, self.learning_rate
                )
            else:
                reporter.banner(self.backend, self.learning_rate)

        if self.check_bp:
            self._run_bp_diagnostic(model, datamodule)

        history = TrainingHistory()
        callbacks = self._build_callbacks(history)
        state = LoopState(
            model=model,
            datamodule=datamodule,
            history=history,
            reporter=reporter,
            max_epochs=self.max_epochs,
        )

        loop.fit(model, datamodule, state, callbacks)
        return history

    def _build_callbacks(self, history: TrainingHistory) -> list:
        """Built-ins first, then the user's.

        ``HistoryCallback`` leads so that anything downstream reading
        ``state.history`` sees the current epoch already recorded, and the
        checkpoint precedes user callbacks so a user callback observing
        ``on_fit_end`` sees the restored weights rather than the last ones.
        """
        callbacks = [HistoryCallback(history)]
        if self.enable_checkpointing:
            callbacks.append(ModelCheckpoint(dirpath=self.checkpoint_dir))
        callbacks.extend(self.callbacks or [])
        return callbacks

    @staticmethod
    def _encoder_class(model):
        """The embedding class that drives prescaling, or None.

        Reads the public ``embedding_obj``; a mismatch here silently disables
        prescaling rather than raising, so the name must stay in step with
        ``QuantumPipeline`` and the model classes.
        """
        embedding = getattr(model, "embedding_obj", None)
        return type(embedding) if embedding is not None else None

    def _run_bp_diagnostic(self, model, datamodule) -> None:
        """Pre-flight gradient-variance check."""
        from pyqit.utils.diagnostic import check_barren_plateau

        result = check_barren_plateau(
            model=model,
            datamodule_or_X=datamodule,
            num_samples=self.bp_samples,
            loss_name=self.loss_fn,
            plot=False,
        )
        if self.verbose >= 1:
            print(result)
        elif result.is_barren:
            logging.getLogger("pyqit.trainer").warning(
                "Barren Plateau detected! Gradient variance is critically low."
            )

    def predict(
        self,
        model: BaseModel | BaseMetaObject,
        datamodule: DataModule,
        return_format: str = "auto",
    ) -> np.ndarray:
        """Run ``model`` over the most specific split ``datamodule`` holds."""
        if not datamodule._is_setup:
            datamodule.setup(
                stage="predict",
                batch_size=self.batch_size,
                n_qubits=getattr(model, "n_qubits", None),
            )

        loader = self._predict_loader(datamodule)
        all_preds = []

        with self._inference_context():
            for X_batch, _ in loader:
                if self.backend == "pennylane":
                    X_batch = pnp.array(X_batch, requires_grad=False)
                all_preds.append(model.predict_step(X_batch))

        if not all_preds:
            return np.array([])

        return self._collect(all_preds, return_format)

    @staticmethod
    def _predict_loader(datamodule: DataModule):
        if datamodule.X_test is not None:
            return datamodule.test_loader(shuffle=False)
        if datamodule.X_val is not None:
            return datamodule.val_loader(shuffle=False)
        return datamodule.train_loader(shuffle=False)

    def _inference_context(self):
        if self.backend == "torch":
            import torch

            return torch.no_grad()

        from contextlib import nullcontext

        return nullcontext()

    @staticmethod
    def _collect(all_preds: list, return_format: str):
        from pyqit.utils.utils import _is_torch

        target = return_format
        if return_format == "auto":
            target = "torch" if _is_torch(all_preds[0]) else "numpy"

        if target == "torch":
            import torch

            return torch.cat(
                [p if _is_torch(p) else torch.as_tensor(p) for p in all_preds], dim=0
            )

        return np.concatenate(
            [
                p.detach().cpu().numpy() if _is_torch(p) else np.asarray(p)
                for p in all_preds
            ],
            axis=0,
        )

    def __repr__(self) -> str:
        return (
            f"Trainer(backend={self.backend!r}, "
            f"max_epochs={self.max_epochs}, learning_rate={self.learning_rate})"
        )

    @classmethod
    def get_test_params(cls):
        return [{"max_epochs": 1, "verbose": 0}]
