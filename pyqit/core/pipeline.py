import numpy as np
from skbase.base import BaseMetaObject

from pyqit.core.trainer import Trainer
from pyqit.data.datamodule import DataModule
from pyqit.models.base.base import BaseModel
from pyqit.utils import _cat, _ensure_col, _is_torch, _mean, _round, _stack, _to_numpy


class PipelineStage:
    def __init__(
        self, model, name=None, passthrough=False, trainable=True, input_slice=None
    ):
        self.model = model
        self.name = name or type(model).__name__
        self.passthrough = passthrough
        self.trainable = trainable
        self.input_slice = input_slice

    def __repr__(self):
        flags = []
        if self.passthrough:
            flags.append("passthrough")
        if not self.trainable:
            flags.append("frozen")
        if self.input_slice:
            flags.append(f"slice={self.input_slice}")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        return f"Stage({self.name}{flag_str})"


class QuantumPipeline(BaseMetaObject):
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
        return self._set_params("steps", **kwargs)

    def get_params(self, deep: bool = True):
        return self._get_params("steps", deep=deep)

    @property
    def named_stages(self) -> dict[str, PipelineStage]:
        return dict(self.steps)

    def __getitem__(self, key: str | int) -> PipelineStage:
        if isinstance(key, int):
            return self.steps[key][1]
        return dict(self.steps)[key]

    def __len__(self) -> int:
        return len(self.steps)

    def forward(self, X):
        if self.mode == "sequential":
            return self._forward_sequential(X)
        elif self.mode == "ensemble":
            return self._forward_ensemble(X)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _forward_sequential(self, X):
        current = X if _is_torch(X) else np.asarray(X)
        for _, stage in self.steps:
            inp = current[:, stage.input_slice] if stage.input_slice else current
            n_features = inp.shape[-1]

            model = stage.model
            if hasattr(model, "n_qubits") and model.n_qubits != n_features:
                raise ValueError(
                    f"Stage '{stage.name}' expects {model.n_qubits} features "
                    f"but received {n_features}."
                )
            out = model.forward(inp)
            out = _ensure_col(out)
            current = _cat(inp, out) if stage.passthrough else out

        return current

    def _forward_ensemble(self, X):
        raw = [stage.model.forward(X) for _, stage in self.steps]

        if callable(self.aggregation):
            return self.aggregation(raw)

        outputs = _stack(raw)
        if self.aggregation == "mean":
            return _mean(outputs, axis=0)
        if self.aggregation == "vote":
            preds = _round(outputs)
            return np.apply_along_axis(
                lambda x: np.bincount(x.astype(int)).argmax(), 0, preds
            )

        raise ValueError(f"Unknown aggregation: {self.aggregation}")

    def fit(
        self, datamodule: DataModule, trainers=None, fit_mode: str = "sequential_greedy"
    ):
        if self.mode == "ensemble":
            self._fit_independent(datamodule, trainers)

        elif self.mode == "sequential":
            if not datamodule._is_setup:
                self._setup_dm(datamodule, trainers)

            if fit_mode == "sequential_greedy":
                self._fit_sequential_greedy(datamodule, trainers)
            elif fit_mode == "frozen_backbone":
                self._fit_frozen_backbone(datamodule, trainers)
            else:
                raise ValueError(
                    f"fit_mode {fit_mode!r} not valid for sequential pipeline."
                )
        return self

    def _setup_dm(self, datamodule: DataModule, trainers: Trainer) -> DataModule:
        first_model = self.steps[0][1].model
        n_qubits = getattr(first_model, "n_qubits", None)

        encoder_class = None
        if hasattr(first_model, "_embedding_obj"):
            encoder_class = type(first_model._embedding_obj)

        first_trainer = self._get_trainer(trainers, 0, self.steps[0][0])
        target_batch_size = getattr(
            first_trainer, "batch_size", getattr(datamodule, "batch_size", 32)
        )

        datamodule.setup(
            backend=getattr(first_model, "backend", "pennylane"),
            batch_size=target_batch_size,
            n_qubits=n_qubits,
            encoder=encoder_class,
        )

    def _fit_independent(self, datamodule: DataModule, trainers):
        for i, (name, stage) in enumerate(self.steps):
            if stage.trainable:
                trainer = self._get_trainer(trainers, i, name)
                if not trainer:
                    raise ValueError(f"Missing Trainer for stage '{name}'")
                trainer.fit(stage.model, datamodule)

    def _fit_sequential_greedy(self, datamodule: DataModule, trainers):
        current_dm = datamodule

        for i, (name, stage) in enumerate(self.steps):
            if stage.trainable:
                trainer = self._get_trainer(trainers, i, name)
                if not trainer:
                    raise ValueError(f"Missing Trainer for stage '{name}'")
                trainer.fit(stage.model, current_dm)

            if i < len(self.steps) - 1:
                current_dm = self._transform_datamodule(current_dm, stage)

    def _fit_frozen_backbone(self, datamodule: DataModule, trainers):
        current_dm = datamodule

        for _, stage in self.steps[:-1]:
            if stage.trainable:
                raise ValueError(
                    f"frozen_backbone mode requires all stages except the last to "
                    f"be frozen. Stage '{stage.name}' has trainable=True."
                )
            current_dm = self._transform_datamodule(current_dm, stage)

        last_name, last_stage = self.steps[-1]
        trainer = self._get_trainer(trainers, -1, last_name)
        if not trainer:
            raise ValueError(f"Missing Trainer for final stage '{last_name}'")

        trainer.fit(last_stage.model, current_dm)

    def _transform_datamodule(self, dm: DataModule, stage: PipelineStage) -> DataModule:
        def _process_split(X):
            if X is None:
                return None
            inp = X[:, stage.input_slice] if stage.input_slice else X

            out = stage.model.forward(inp)
            out = _ensure_col(out)

            return _cat(inp, out) if stage.passthrough else out

        new_dm = dm.clone_empty()

        new_dm._is_setup = True
        new_dm._backend = dm._backend
        new_dm._X_train = _process_split(dm._X_train)
        new_dm._X_val = _process_split(dm._X_val)
        new_dm._X_test = _process_split(dm._X_test)

        new_dm._y_train = dm._y_train
        new_dm._y_val = dm._y_val
        new_dm._y_test = dm._y_test

        return new_dm

    def _to_named_steps(self, steps: list) -> list[tuple[str, PipelineStage]]:
        if steps and all(
            isinstance(s, tuple)
            and len(s) == 2
            and isinstance(s[0], str)
            and isinstance(s[1], PipelineStage)
            for s in steps
        ):
            return list(steps)

        result = []
        seen_names: set[str] = set()

        for i, s in enumerate(steps):
            if isinstance(s, tuple) and len(s) == 2:
                name, obj = s
            elif isinstance(s, (PipelineStage, BaseModel)):
                name, obj = None, s
            else:
                raise TypeError(
                    f"Each stage must be a BaseModel, PipelineStage, or "
                    f"(name, model/stage) tuple. Got {type(s).__name__} at index {i}."
                )

            if isinstance(obj, BaseModel):
                obj = PipelineStage(obj, name=name)

            if not isinstance(obj, PipelineStage):
                raise TypeError(
                    f"Stage {i}: expected PipelineStage or BaseModel, "
                    f"got {type(obj).__name__}."
                )

            if name is None:
                name = obj.name
            base, counter = name, 1
            while name in seen_names:
                name = f"{base}_{counter}"
                counter += 1
            seen_names.add(name)
            obj.name = name
            result.append((name, obj))

        return result

    def predict_step(self, X):
        if self.mode == "sequential":
            current = X if _is_torch(X) else np.asarray(X)

            for i, (_, stage) in enumerate(self.steps):
                inp = current[:, stage.input_slice] if stage.input_slice else current

                if i < len(self.steps) - 1:
                    out = stage.model.forward(inp)
                else:
                    out = stage.model.predict_step(inp)

                out = _ensure_col(out)
                current = _cat(inp, out) if stage.passthrough else out

            return current

        elif self.mode == "ensemble":
            if self.aggregation == "vote":
                return self._forward_ensemble(X)

            raw_preds = self._forward_ensemble(X)

            if raw_preds.ndim == 1 or (raw_preds.ndim == 2 and raw_preds.shape[1] == 1):
                if _is_torch(raw_preds):
                    import torch

                    return (raw_preds >= 0.5).to(torch.int)
                return (raw_preds >= 0.5).astype(int)
            else:
                if _is_torch(raw_preds):
                    return raw_preds.argmax(dim=1)
                return raw_preds.argmax(axis=1)

        raise ValueError(f"Unknown mode: {self.mode}")

    def predict(
        self,
        X,
        batch_size: int = 32,
        backend: str = "pennylane",
        return_format: str = "auto",
    ):
        dummy_y = np.zeros(len(X))
        dm = DataModule(X=X, y=dummy_y, batch_size=batch_size, split=(0.0, 0.0, 1.0))
        first_stage_model = self.steps[0][1].model
        n_qubits = getattr(first_stage_model, "n_qubits", None)

        encoder_class = None
        if hasattr(first_stage_model, "_embedding_obj"):
            encoder_class = type(first_stage_model._embedding_obj)

        dm.setup(
            stage="predict",
            backend=backend,
            batch_size=batch_size,
            n_qubits=n_qubits,
            encoder=encoder_class,
        )

        temp_trainer = Trainer(backend_type=backend, batch_size=batch_size)

        return temp_trainer.predict(
            model=self, datamodule=dm, return_format=return_format
        )

    def _get_trainer(self, trainers, idx, name):
        if trainers is None:
            return None
        if isinstance(trainers, dict):
            return trainers.get(name) or trainers.get(idx)
        if isinstance(trainers, list):
            return trainers[idx] if 0 <= idx < len(trainers) else None
        return trainers

    def clone(self) -> "QuantumPipeline":
        import copy

        cloned_steps = [(name, copy.deepcopy(stage)) for name, stage in self.steps]
        return type(self)(
            steps=cloned_steps,
            mode=self.mode,
            aggregation=self.aggregation,
        )

    def __call__(self, X):
        return self.forward(X)

    def __repr__(self):
        stage_str = "\n  ".join(str(s) for s in self.steps)
        return f"QuantumPipeline(mode={self.mode})\n  {stage_str}"

    def summary(self):
        pass
