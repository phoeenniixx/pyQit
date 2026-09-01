"""The Trainer: configuration, orchestration and backend dispatch."""

from collections.abc import Callable
import logging

import numpy as np
from skbase.base import BaseMetaObject
from skbase.utils.dependencies import _check_soft_dependencies

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

    The Trainer trains nothing itself. It seeds, sets the DataModule up, prints
    the run summary, optionally runs the barren-plateau pre-flight, assembles
    the callback list, and hands off to the loop registered for the active
    backend.

    Its parameters are the ones that mean the same thing on every backend;
    anything backend-specific goes through ``backend_kwargs``.

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
        Run on both backends. Lightning callbacks are not accepted.
    verbose : {0, 1, 2}, default 2
        ``0`` silent, ``1`` progress only, ``2`` progress plus the model table.
    seed : int, optional
        Seeded before anything stochastic runs. Weights are drawn at model
        construction, so reproducing them needs ``pyqit.set_seed`` before the
        model is built; this covers training and diagnostics only.
    enable_checkpointing : bool, default False
        Installs a default ``ModelCheckpoint``. For any other policy, pass your
        own through ``callbacks``.
    checkpoint_dir : str, optional
        Directory for the default checkpoint. Defaults to ``"checkpoints"``.
    logger : bool or object, default False
        Forwarded to Lightning on the torch backend; ignored on pennylane.
    check_bp : bool, default False
        Run the barren-plateau gradient-variance check before training.
    bp_samples : int, default 200
        Gradient samples drawn by that check.
    backend_kwargs : dict, optional
        Forwarded verbatim to ``lightning.pytorch.Trainer`` on the torch
        backend. Rejected on pennylane, which has no underlying trainer.
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
        self.backend_kwargs = backend_kwargs
        super().__init__()

        self.backend = get_backend()
        self._print_summary = True

    def fit(self, model: BaseModel, datamodule: DataModule) -> TrainingHistory:
        """Train ``model`` on ``datamodule``.

        Parameters
        ----------
        model : BaseModel
            Trained in place.
        datamodule : DataModule
            Set up here if it is not already.

        Returns
        -------
        TrainingHistory
            Per-epoch losses, accuracies and timings.
        """
        if self.seed is not None:
            set_seed(self.seed)

        reporter = Reporter(
            verbose=self.verbose,
            max_epochs=self.max_epochs,
            show_summary=self._print_summary,
        )

        loop = get_training_loop(self.backend, trainer=self, reporter=reporter)

        self._setup_data(model, datamodule, stage="fit")

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

    def predict(
        self,
        model: BaseModel | BaseMetaObject,
        datamodule: DataModule,
        return_format: str = "auto",
    ) -> np.ndarray:
        """Run ``model`` over the most specific split ``datamodule`` holds.

        Parameters
        ----------
        model : BaseModel or BaseMetaObject
            Used for inference only; weights are not touched.
        datamodule : DataModule
            Test split if present, else validation, else train.
        return_format : {"auto", "numpy", "torch"}, default "auto"
            ``"auto"`` follows whatever the model emits.

        Returns
        -------
        numpy.ndarray or torch.Tensor
            Predictions for every sample in the chosen split.
        """
        import pennylane.numpy as pnp

        self._setup_data(model, datamodule, stage="predict")

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

    def _setup_data(self, model, datamodule: DataModule, stage: str) -> None:
        """Set the DataModule up with the shaping ``model`` implies.

        ``fit`` and ``predict`` share this so they cannot disagree about the
        encoder, which drives quantum prescaling: omitting it leaves inputs
        unscaled without raising. ``setup`` is idempotent, so calling this on an
        already-set-up DataModule only re-applies ``batch_size``.
        """
        datamodule.setup(
            stage=stage,
            batch_size=self.batch_size,
            n_qubits=getattr(model, "n_qubits", None),
            encoder=self._encoder_class(model),
        )

    def _build_callbacks(self, history: TrainingHistory) -> list:
        """Built-ins first, then the user's.

        ``HistoryCallback`` leads so anything downstream reading
        ``state.history`` sees the current epoch already recorded, and the
        checkpoint precedes user callbacks so a user callback observing
        ``on_fit_end`` sees the restored weights.
        """
        callbacks = [HistoryCallback(history)]
        if self.enable_checkpointing:
            callbacks.append(ModelCheckpoint(dirpath=self.checkpoint_dir))
        callbacks.extend(self.callbacks or [])
        return callbacks

    @staticmethod
    def _encoder_class(model) -> type | None:
        """The embedding class that drives prescaling, or None.

        Reads the public ``embedding_obj``; a mismatch silently disables
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

    @staticmethod
    def _predict_loader(datamodule: DataModule):
        """The most specific split's loader: test, else val, else train."""
        if datamodule.X_test is not None:
            return datamodule.test_loader(shuffle=False)
        if datamodule.X_val is not None:
            return datamodule.val_loader(shuffle=False)
        return datamodule.train_loader(shuffle=False)

    def _inference_context(self):
        """A no-grad context on torch, a null context elsewhere."""
        from contextlib import nullcontext

        if self.backend == "torch":
            import torch

            return torch.no_grad()

        return nullcontext()

    @staticmethod
    def _collect(all_preds: list, return_format: str):
        """Concatenate per-batch predictions into one array of ``return_format``."""
        from pyqit.utils.utils import _is_torch

        target = return_format
        if return_format == "auto":
            target = "torch" if _is_torch(all_preds[0]) else "numpy"

        if target == "torch":
            if not _check_soft_dependencies("torch", severity="none"):
                raise ImportError(
                    "return_format='torch' requires torch, which is not "
                    "installed. Use return_format='numpy' or 'auto'."
                )

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
