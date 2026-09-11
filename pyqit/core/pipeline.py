from contextlib import contextmanager
import copy

import numpy as np
from skbase.base import BaseMetaObject

from pyqit.core.trainer import Trainer, console, has_rich
from pyqit.data.datamodule import DataModule, _apply_prescale
from pyqit.models.base.base import BaseModel
from pyqit.utils.utils import (
    _cat,
    _count_params,
    _ensure_col,
    _is_torch,
    _mean,
    _stack,
    _to_numpy,
)


class PipelineStage:
    """One model in a `QuantumPipeline`.

    Parameters
    ----------
    model : BaseModel
    name : str, optional
        Defaults to the model's class name.
    passthrough : bool, default False
        Concatenate this stage's input onto its output.
    trainable : bool, default True
        `frozen_backbone` fit mode requires this False on every non-final stage.
    input_slice : slice or array-like of int, optional
        Feed only these input columns to this stage.
    """

    def __init__(
        self, model, name=None, passthrough=False, trainable=True, input_slice=None
    ):
        self.model = model
        self.name = name or type(model).__name__
        self.passthrough = passthrough
        self.trainable = trainable
        self.input_slice = input_slice

    def _flags(self):
        return [
            flag
            for flag, on in (
                ("frozen", not self.trainable),
                ("passthrough", self.passthrough),
                (f"slice={self.input_slice}", self.input_slice is not None),
            )
            if on
        ]

    def __repr__(self):
        flags = self._flags()
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        return f"Stage({self.name}{flag_str})"


class QuantumPipeline(BaseMetaObject):
    """Compose `PipelineStage` objects sequentially or as an ensemble.

    Parameters
    ----------
    steps : list of PipelineStage, or list of (name, model)
    mode : {"sequential", "ensemble"}, default "sequential"
        Sequential feeds each stage's output to the next. Ensemble runs every
        stage on the same input and combines the outputs.
    aggregation : {"mean", "vote"} or callable, default "mean"
        How ensemble outputs are combined. Ignored in sequential mode.

    Examples
    --------
    >>> from pyqit.core import PipelineStage, QuantumPipeline
    >>> pipe = QuantumPipeline(
    ...     [PipelineStage(backbone, trainable=False), PipelineStage(head)],
    ...     mode="sequential",
    ... )
    >>> pipe.fit(dm, fit_mode="frozen_backbone")  # doctest: +SKIP
    """

    _tags = {
        "mode": "sequential",
        "n_stages": 0,
        "has_quantum": False,
    }

    def __init__(self, steps, mode="sequential", aggregation="mean"):
        if mode not in ("sequential", "ensemble"):
            raise ValueError(f"mode must be 'sequential' or 'ensemble', got {mode}.")
        self.steps = self._to_named_steps(steps)
        self.mode = mode
        self.aggregation = aggregation
        super().__init__()

        has_quantum = any(
            s.model.get_tag("is_quantum", tag_value_default=False)
            for _, s in self.steps
        )
        self.set_tags(
            mode=mode,
            n_stages=len(self.steps),
            has_quantum=has_quantum,
        )

    def set_params(self, **kwargs):
        """Set stage or nested-stage parameters. See `sklearn`'s convention."""
        return self._set_params("steps", **kwargs)

    def get_params(self, deep: bool = True):
        """Get stage and nested-stage parameters. See `sklearn`'s convention."""
        return self._get_params("steps", deep=deep)

    @property
    def named_stages(self) -> dict[str, PipelineStage]:
        """Stages keyed by name."""
        return dict(self.steps)

    def __getitem__(self, key: str | int) -> PipelineStage:
        if isinstance(key, int):
            return self.steps[key][1]
        return self.named_stages[key]

    def __len__(self) -> int:
        return len(self.steps)

    @staticmethod
    def _slice_input(X, input_slice):
        """Select ``input_slice`` columns from ``X``, always returning 2-D."""
        if X is None or input_slice is None:
            return X
        if isinstance(input_slice, (int, np.integer)):
            return X[:, [int(input_slice)]]
        return X[:, input_slice]

    @staticmethod
    def _prescale_of(model):
        return getattr(type(getattr(model, "embedding_obj", None)), "PRESCALE", None)

    def _prescale_for(self, stage, X):
        """Shape ``X`` for ``stage``'s embedding, as ``DataModule.setup`` does."""
        n_qubits = getattr(stage.model, "n_qubits", None)
        prescale = self._prescale_of(stage.model)
        if X is None or n_qubits is None or prescale in (None, "none"):
            return X

        if _is_torch(X):
            import torch

            out = _apply_prescale(_to_numpy(X), prescale, n_qubits)
            return torch.as_tensor(out, dtype=X.dtype, device=X.device)
        return _apply_prescale(np.asarray(X), prescale, n_qubits)

    def _prepare_stage_input(self, X, stage):
        """Slice, then prescale, the input a stage is about to consume.

        Every stage is prescaled here, the first included: the pipeline sets its
        DataModule up without an encoder, so slicing and passthrough see the
        unprescaled features and each embedding's prescaling runs exactly once.
        """
        return self._prescale_for(stage, self._slice_input(X, stage.input_slice))

    def _run_stage(self, stage, X, final=False):
        """Run one sequential stage on ``X``; return what the next one receives.

        Passthrough concatenates the unprescaled slice, not the prescaled input,
        so the next stage's prescaling is applied to it once rather than twice.
        """
        raw = self._slice_input(X, stage.input_slice)
        inp = self._prescale_for(stage, raw)
        self._check_stage_width(stage, inp.shape[-1])
        out = stage.model.predict_step(inp) if final else stage.model.forward(inp)
        out = _ensure_col(out)
        return _cat(raw, out) if stage.passthrough else out

    def _run_sequential(self, X, labels=False):
        current = X if _is_torch(X) else np.asarray(X)
        last = len(self.steps) - 1
        for i, (_, stage) in enumerate(self.steps):
            current = self._run_stage(stage, current, final=labels and i == last)
        return current

    def _expected_width(self, model):
        """Feature width ``model`` expects."""
        n_qubits = getattr(model, "n_qubits", None)
        if n_qubits is None:
            return None
        return 2**n_qubits if self._prescale_of(model) == "amplitude" else n_qubits

    def _check_stage_width(self, stage, n_features):
        """Raise if ``stage.model`` cannot consume ``n_features`` columns."""
        expected = self._expected_width(stage.model)
        if expected is None or n_features is None or expected == n_features:
            return
        raise ValueError(
            f"Stage '{stage.name}' expects {expected} features "
            f"but received {n_features}."
        )

    def _check_ensemble_consistent(self):
        """Ensemble stages all receive the same X, so they must agree on shape."""
        sliced = [n for n, s in self.steps if s.input_slice is not None]
        if sliced:
            raise ValueError(
                f"Stages {sliced} set input_slice, but ensemble mode feeds every "
                "stage the same X and aggregates the results, so per-stage "
                "column views are not applied. Use mode='sequential'."
            )

        specs = []
        for name, stage in self.steps:
            embedding = getattr(stage.model, "embedding_obj", None)
            encoder = type(embedding).__name__ if embedding is not None else None
            specs.append((name, self._expected_width(stage.model), encoder))

        if len({s[1:] for s in specs}) > 1:
            detail = "; ".join(f"{n}: width={w}, encoder={e}" for n, w, e in specs)
            raise ValueError(
                "Ensemble stages are all fed the same input, but they disagree "
                f"on what they accept: {detail}. Give every stage the same "
                "n_qubits and encoder, or use mode='sequential'."
            )

    def forward(self, X):
        """Run every stage on `X` and return the pipeline's raw output.

        `X` is split and normalized but not prescaled: each stage's embedding
        prescaling is applied here, as `predict` and `fit` do.
        """
        if self.mode == "sequential":
            return self._run_sequential(X)
        return self._forward_ensemble(X)

    def _forward_ensemble(self, X):
        inputs = [self._prepare_stage_input(X, stage) for _, stage in self.steps]

        if self.aggregation == "vote":
            labels = np.stack(
                [
                    _to_numpy(stage.model.predict_step(x)).ravel().astype(int)
                    for (_, stage), x in zip(self.steps, inputs)
                ]
            )
            return np.apply_along_axis(lambda v: np.bincount(v).argmax(), 0, labels)

        raw = [stage.model.forward(x) for (_, stage), x in zip(self.steps, inputs)]
        if callable(self.aggregation):
            return self.aggregation(raw)
        if self.aggregation == "mean":
            return _mean(_stack(raw), axis=0)

        raise ValueError(f"Unknown aggregation: {self.aggregation}")

    def fit(
        self, datamodule: DataModule, trainers=None, fit_mode: str = "sequential_greedy"
    ):
        """Fit every trainable stage.

        Parameters
        ----------
        datamodule : DataModule
            Split and normalized here, never prescaled: the pipeline prescales
            every stage's input itself, so a DataModule already set up for a
            single model's encoder is rejected.
        trainers : Trainer, list of Trainer, or dict
            One `Trainer` for every stage, or one per stage by position or
            name. Every trainable stage needs one; a missing Trainer raises.
        fit_mode : {"sequential_greedy", "frozen_backbone"}, default "sequential_greedy"
            Ignored in ensemble mode. `frozen_backbone` requires every
            non-final stage to have `trainable=False`.

        Returns
        -------
        QuantumPipeline
            `self`, fitted in place.

        Examples
        --------
        >>> pipe.fit(dm, trainers=pyqit.Trainer(max_epochs=20))  # doctest: +SKIP
        """
        if datamodule.encoder is not None:
            raise ValueError(
                f"This DataModule is already prescaled for "
                f"{datamodule.encoder.__name__}, but a QuantumPipeline prescales "
                "each stage's input itself, so it would be scaled twice. Pass a "
                "DataModule that has not been set up for a single model."
            )
        if self.mode == "ensemble":
            self._check_ensemble_consistent()
        elif fit_mode not in ("sequential_greedy", "frozen_backbone"):
            raise ValueError(
                f"fit_mode {fit_mode!r} not valid for sequential pipeline."
            )
        elif fit_mode == "frozen_backbone":
            trainable = [name for name, s in self.steps[:-1] if s.trainable]
            if trainable:
                raise ValueError(
                    "frozen_backbone mode requires all stages except the last to "
                    f"be frozen. Stage '{trainable[0]}' has trainable=True."
                )

        if not datamodule._is_setup:
            first_trainer = self._get_trainer(trainers, 0, self.steps[0][0])
            datamodule.setup(batch_size=getattr(first_trainer, "batch_size", None))

        verbose = self._pipeline_verbose(trainers)
        sequential = self.mode == "sequential"
        self._print_pipeline_summary(verbose, fit_mode if sequential else None)

        dm = datamodule
        for i, (_, stage) in enumerate(self.steps):
            if stage.trainable:
                self._fit_stage(i, dm, trainers, verbose)
            if sequential and i < len(self.steps) - 1:
                dm = self._transform_datamodule(dm, stage)

        self._fit_normalizer = datamodule.normalizer
        return self

    def _fit_stage(self, idx, dm, trainers, verbose):
        name, stage = self.steps[idx]
        trainer = self._get_trainer(trainers, idx, name)
        if not trainer:
            raise ValueError(f"Missing Trainer for stage '{name}'")
        stage_dm = self._stage_datamodule(dm, stage)
        self._check_stage_width(stage, stage_dm._X_train.shape[-1])
        self._announce_stage(verbose, idx, name)
        with self._quiet_stage_summary(trainer):
            trainer.fit(stage.model, stage_dm)

    def _stage_datamodule(self, dm: DataModule, stage: PipelineStage) -> DataModule:
        """Slice and prescale ``dm`` into the view ``stage`` actually consumes."""
        return dm._map_features(lambda X: self._prepare_stage_input(X, stage))

    def _transform_datamodule(self, dm: DataModule, stage: PipelineStage) -> DataModule:
        """Run ``stage`` over every split of the unprescaled ``dm``."""
        return dm._map_features(lambda X: self._transform_split(stage, X, dm._backend))

    def _transform_split(self, stage, X, backend):
        if backend != "torch" and not _is_torch(X):
            return self._run_stage(stage, X)

        import torch

        if not _is_torch(X):
            X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        with torch.no_grad():
            out = self._run_stage(stage, X)
        return out.detach() if _is_torch(out) else out

    @staticmethod
    def _to_named_steps(steps: list) -> list[tuple[str, PipelineStage]]:
        named, seen = [], set()
        for i, step in enumerate(steps):
            name, obj = (
                step if isinstance(step, tuple) and len(step) == 2 else (None, step)
            )
            if isinstance(obj, BaseModel):
                obj = PipelineStage(obj, name=name)
            if not isinstance(obj, PipelineStage) or not isinstance(
                obj.model, BaseModel
            ):
                raise TypeError(
                    f"Each stage must be a BaseModel, PipelineStage, or "
                    f"(name, model/stage) tuple. Got {type(step).__name__} at "
                    f"index {i}."
                )

            base = name = name or obj.name
            counter = 1
            while name in seen:
                name = f"{base}_{counter}"
                counter += 1
            seen.add(name)
            obj.name = name
            named.append((name, obj))
        return named

    def predict_step(self, X):
        """Run every stage on `X`, hard-labeling the final stage's output."""
        if self.mode == "sequential":
            return self._run_sequential(X, labels=True)

        out = self._forward_ensemble(X)
        if self.aggregation == "vote":
            return out
        if out.ndim > 1 and out.shape[1] > 1:
            return out.argmax(1)
        labels = out >= 0.5
        return labels.int() if _is_torch(labels) else labels.astype(int)

    def predict(
        self,
        X,
        batch_size: int = 32,
        return_format: str = "auto",
        trainer_kwargs: dict | None = None,
    ):
        """Predict on raw `X`, not a `DataModule`.

        Reapplies the normalization fitted during `fit`, then prescales every
        stage, so `X` should be the same kind of raw input `fit` was given.

        Parameters
        ----------
        X : array-like
        batch_size : int, default 32
        return_format : {"auto", "numpy", "torch"}, default "auto"

        Returns
        -------
        array-like

        Examples
        --------
        >>> preds = pipe.predict(X_test)  # doctest: +SKIP
        """
        normalizer = getattr(self, "_fit_normalizer", None)
        dm = DataModule(
            X,
            np.zeros(len(X)),
            normalize=normalizer.method if normalizer is not None else None,
            split=(0.0, 0.0, 1.0),
            batch_size=batch_size,
        )
        dm._normalizer = normalizer
        dm.setup(stage="predict")

        trainer = Trainer(batch_size=batch_size, **(trainer_kwargs or {}))
        return trainer.predict(model=self, datamodule=dm, return_format=return_format)

    def validate(self, datamodule: DataModule, trainer_kwargs: dict | None = None):
        """``val_loss`` and ``val_acc`` on ``datamodule``'s validation split.

        Parameters
        ----------
        datamodule : DataModule
            The one `fit` was given, or one set up the same way: split and
            normalized but not prescaled for any single encoder.
        trainer_kwargs : dict, optional
            Forwarded to `Trainer`; `loss_fn` picks the loss reported.
        """
        return Trainer(**(trainer_kwargs or {})).validate(self, datamodule)

    def test(self, datamodule: DataModule, trainer_kwargs: dict | None = None):
        """``test_loss`` and ``test_acc`` on ``datamodule``'s test split.

        Parameters
        ----------
        datamodule : DataModule
            The one `fit` was given, or one set up the same way: split and
            normalized but not prescaled for any single encoder.
        trainer_kwargs : dict, optional
            Forwarded to `Trainer`; `loss_fn` picks the loss reported.
        """
        return Trainer(**(trainer_kwargs or {})).test(self, datamodule)

    def _pipeline_verbose(self, trainers) -> int:
        """Loudest verbosity among the stage trainers.

        The pipeline has no `verbose` of its own, so it takes its cue from the
        trainers it was handed rather than introducing a second knob to keep in
        sync with them.
        """
        levels = [
            getattr(self._get_trainer(trainers, i, name), "verbose", None)
            for i, (name, _) in enumerate(self.steps)
        ]
        levels = [v for v in levels if v is not None]
        return max(levels) if levels else 0

    @staticmethod
    @contextmanager
    def _quiet_stage_summary(trainer):
        """Suppress a stage trainer's per-model summary for the duration of a fit.

        The pipeline prints one summary that names every stage; the trainer's own
        table would repeat it once per stage without saying which stage it is.
        Restored afterwards so a trainer reused outside the pipeline is unchanged.
        """
        previous = getattr(trainer, "_print_summary", True)
        trainer._print_summary = False
        try:
            yield
        finally:
            trainer._print_summary = previous

    def _print_pipeline_summary(self, verbose: int, fit_mode: str | None = None):
        """Print the pipeline structure once, in place of N per-model tables."""
        if verbose < 2:
            return

        header = f"QuantumPipeline | mode={self.mode} | stages={len(self.steps)}"
        if self.mode == "ensemble":
            header += f" | aggregation={self.aggregation}"
        elif fit_mode is not None:
            header += f" | fit_mode={fit_mode}"

        rows = [
            (
                str(i + 1),
                name,
                type(stage.model).__name__,
                str(getattr(stage.model, "n_qubits", "N/A")),
                str(_count_params(stage.model) or 0),
                ", ".join(stage._flags()) or "trainable",
            )
            for i, (name, stage) in enumerate(self.steps)
        ]
        columns = ("#", "Stage", "Model", "Qubits", "Params", "Status")

        if not has_rich():
            print(f"\n[Pipeline] {header}")
            widths = [
                max(len(c), *(len(r[j]) for r in rows)) for j, c in enumerate(columns)
            ]
            fmt = "  ".join(f"{{:<{w}}}" for w in widths)
            print("  " + fmt.format(*columns))
            for row in rows:
                print("  " + fmt.format(*row))
            print()
            return

        from rich.table import Table

        table = Table(show_header=True, header_style="bold cyan", box=None, title=None)
        for column in columns:
            table.add_column(column, style="bold" if column == "Stage" else None)
        for row in rows:
            table.add_row(*row)

        console().print(f"[bold cyan][Pipeline][/bold cyan] {header}")
        console().print(table)
        console().print()

    def _announce_stage(self, verbose: int, idx: int, name: str):
        """Name the stage about to train, since its trainer no longer does."""
        if verbose < 1:
            return
        label = f"[{idx + 1}/{len(self.steps)}] Fitting stage {name!r}"
        if has_rich():
            console().print(f"[bold cyan]{label}[/bold cyan]")
        else:
            print(label)

    def _get_trainer(self, trainers, idx, name):
        if trainers is None:
            return None
        if isinstance(trainers, dict):
            return trainers.get(name) or trainers.get(idx)
        if isinstance(trainers, list):
            return trainers[idx] if 0 <= idx < len(trainers) else None
        return trainers

    def clone(self) -> "QuantumPipeline":
        """Return an independent copy: stages deep-copied, weights included."""
        return type(self)(
            steps=[(name, copy.deepcopy(stage)) for name, stage in self.steps],
            mode=self.mode,
            aggregation=self.aggregation,
        )

    def __call__(self, X):
        """Alias for `forward`."""
        return self.forward(X)

    def __repr__(self):
        stage_str = "\n  ".join(str(s) for s in self.steps)
        return f"QuantumPipeline(mode={self.mode})\n  {stage_str}"
