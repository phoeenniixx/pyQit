from pyqit.base import BaseModel
import pennylane.numpy as np

class PipelineStage:
    def __init__(self, model, name=None, passthrough=False, 
                 trainable=True, input_slice=None):
        self.model = model
        self.name = name or type(model).__name__
        self.passthrough = passthrough
        self.trainable = trainable
        self.input_slice = input_slice

    def __repr__(self):
        flags = []
        if self.passthrough: flags.append("passthrough")
        if not self.trainable: flags.append("frozen")
        if self.input_slice: flags.append(f"slice={self.input_slice}")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        return f"Stage({self.name}{flag_str})"
    

class QuantumPipeline:
    def __init__(self, stages, mode="sequential", aggregation="mean"):
        self.stages = [self._wrap(s) for s in stages]
        self.mode = mode
        self.aggregation = aggregation

    def _wrap(self, stage):
        if isinstance(stage, PipelineStage):
            return stage
        if isinstance(stage, tuple) and len(stage) == 2:
            name, model = stage
            return PipelineStage(model, name=name)
        if isinstance(stage, BaseModel):
            return PipelineStage(stage)
        raise TypeError(
            f"Stage must be a BaseModel, PipelineStage, or (name, model) tuple. "
            f"Got {type(stage).__name__}"
        )

    def forward(self, X):
        if self.mode == "sequential":
            return self._forward_sequential(X)
        elif self.mode == "ensemble":
            return self._forward_ensemble(X)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _forward_sequential(self, X):
        current = np.array(X)
        for stage in self.stages:
            inp = current[:, stage.input_slice] if stage.input_slice else current
            n_features = inp.shape[-1]

            model = stage.model
            if hasattr(model, 'n_qubits') and model.n_qubits != n_features:
                raise ValueError(
                    f"Stage '{stage.name}' expects {model.n_qubits} features "
                    f"(n_qubits={model.n_qubits}) but received {n_features} features "
                    f"from the previous stage. Adjust n_qubits or measure_wires."
                )

            out = np.array(model.forward(inp))
            if out.ndim == 1:
                out = out.reshape(-1, 1)
            current = np.concatenate([inp, out], axis=-1) if stage.passthrough else out
        return current

    def _forward_ensemble(self, X):
        outputs = [stage.model.forward(X) for stage in self.stages]
        outputs = np.array(outputs) 
        
        if callable(self.aggregation):
            return self.aggregation(outputs)
        elif self.aggregation == "mean":
            return np.mean(outputs, axis=0)
        elif self.aggregation == "vote":
            preds = np.round(outputs)
            return np.apply_along_axis(
                lambda x: np.bincount(x.astype(int)).argmax(), 0, preds
            )
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

    def fit(self, X, y, trainers=None, fit_mode="end_to_end"):
        if fit_mode == "end_to_end":
            self._fit_sequential(X, y, trainers)
        elif fit_mode == "frozen_backbone":
            self._fit_frozen_backbone(X, y, trainers)
        elif fit_mode == "independent":
            self._fit_independent(X, y, trainers)
        return self

    def _fit_sequential(self, X, y, trainers):
        current = np.array(X)
        for i, stage in enumerate(self.stages):
            if not stage.trainable:
                out = np.array([stage.model.forward(np.atleast_1d(x)) for x in current])
                current = out.reshape(-1, 1) if out.ndim == 1 else out
                continue
            trainer = self._get_trainer(trainers, i, stage.name)
            stage.model.fit(current, y, trainer=trainer)
            out = np.array([stage.model.forward(np.atleast_1d(x)) for x in current])
            current = out.reshape(-1, 1) if out.ndim == 1 else out


    def _fit_frozen_backbone(self, X, y, trainers):
        current = X
        for stage in self.stages[:-1]:
            current = stage.model.forward(current)
        last = self.stages[-1]
        trainer = self._get_trainer(trainers, -1, last.name)
        last.model.fit(current, y, trainer=trainer)

    def _fit_independent(self, X, y, trainers):
        for i, stage in enumerate(self.stages):
            trainer = self._get_trainer(trainers, i, stage.name)
            stage.model.fit(X, y, trainer=trainer)

    def _get_trainer(self, trainers, idx, name):
        if trainers is None:
            return None
        if isinstance(trainers, dict):
            return trainers.get(name) or trainers.get(idx)
        if isinstance(trainers, list):
            return trainers[idx] if idx < len(trainers) else None
        return trainers 

    def __call__(self, X):
        return self.forward(X)

    def __repr__(self):
        stage_str = "\n  ".join(str(s) for s in self.stages)
        return f"QuantumPipeline(mode={self.mode})\n  {stage_str}"

    def summary(self):
        print(f"QuantumPipeline — mode: {self.mode}")
        print("-" * 40)
        for i, stage in enumerate(self.stages):
            print(f"  [{i}] {stage}")
        print("-" * 40)