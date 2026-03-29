import numpy as np
import pennylane.numpy as pnp
import pytest

from pyqit.core.pipeline import PipelineStage, QuantumPipeline
from pyqit.core.trainer import Trainer
from pyqit.data.datamodule import DataModule
from pyqit.models.base.base import BaseModel
from pyqit.tests._fixture_generators import BaseFixtureGenerator
from pyqit.tests.scenarios import make_scenario


def _extract_numpy(tensor):
    """Helper to safely extract numpy arrays from either Pennylane or Torch tensors."""
    if type(tensor).__module__.startswith("torch"):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


class TestAllModels(BaseFixtureGenerator):
    """Package-level tests for all ``BaseModel`` subclasses."""

    object_type_filter = "model"

    def _get_matching_datamodule(
        self, model_instance, n_samples=16, batch_size=8, split=(0.6, 0.2, 0.2)
    ):
        """Helper to generate a dm that perfectly matches the model's architecture."""
        n_features = getattr(model_instance, "n_qubits", 4)
        n_classes = getattr(model_instance, "n_classes", 2)

        scenario = make_scenario(
            n_samples=n_samples, n_features=n_features, n_classes=n_classes, seed=42
        )

        return DataModule(
            X=scenario["X"], y=scenario["y"], batch_size=batch_size, split=split
        )

    @pytest.mark.parametrize("backend", ["pennylane", "torch"])
    def test_basic_trainer_flow(self, object_class, backend):
        """
        Verifies the foundational architecture:
        DataModule feeds Trainer -> Trainer updates Model -> Trainer predicts.
        """
        if backend == "torch":
            pytest.importorskip("torch")

        params = object_class.get_test_params()[0]
        params["backend"] = backend
        model = object_class(**params)

        dm = self._get_matching_datamodule(model, n_samples=16, batch_size=8)
        trainer = Trainer(backend_type=backend, max_epochs=1, learning_rate=0.1)

        weight_keys = list(model.weights.keys())
        initial_weights = [_extract_numpy(model.weights[k]).copy() for k in weight_keys]

        trainer.fit(model, datamodule=dm)

        for k, initial_w in zip(weight_keys, initial_weights):
            current_w = _extract_numpy(model.weights[k])
            assert not np.allclose(
                initial_w, current_w
            ), f"Weights for '{k}' did not update during {backend} fit."

        preds = trainer.predict(model, datamodule=dm, return_format="numpy")

        assert len(preds) > 0, "Trainer predict returned an empty array."
        assert not np.isnan(preds).any(), "Trainer predict returned NaNs."
        assert np.issubdtype(
            preds.dtype, np.integer
        ), "Predict did not return hard integer labels."

    @pytest.mark.parametrize(
        "pipeline_mode, fit_mode",
        [
            ("sequential", "sequential_greedy"),
            ("sequential", "frozen_backbone"),
            ("ensemble", "independent"),
        ],
    )
    def test_pipeline_permutations(self, object_class, pipeline_mode, fit_mode):
        """
        Verifies that pipelines can seamlessly route DataModules and execute
        all valid training architectures using the Trainer.
        """
        params = object_class.get_test_params()[0]
        model_a = object_class(**params)
        model_b = object_class(**params)

        dm = self._get_matching_datamodule(model_a, n_samples=16, batch_size=8)
        trainer = Trainer(backend_type="pennylane", max_epochs=1, learning_rate=0.1)

        trainable_a = False if fit_mode == "frozen_backbone" else True

        pipeline = QuantumPipeline(
            [
                PipelineStage(model_a, name="stage_1", trainable=trainable_a),
                PipelineStage(model_b, name="stage_2", trainable=True),
            ],
            mode=pipeline_mode,
        )

        pipeline.fit(datamodule=dm, trainers=trainer, fit_mode=fit_mode)

        X_new = dm.X_raw[:5]
        preds = pipeline.predict(X_new, batch_size=8, backend="pennylane")

        assert len(preds) == len(
            X_new
        ), "Pipeline predict returned incorrect batch size."
        assert not np.isnan(preds).any(), "Pipeline returned NaNs after processing."
        assert np.issubdtype(
            preds.dtype, np.integer
        ), "Pipeline predict did not return hard integer labels."

    def test_pipeline_invalid_fit_mode(self, object_class):
        """Verifies the pipeline catches invalid training configurations."""
        params = object_class.get_test_params()[0]
        model_a = object_class(**params)
        dm = self._get_matching_datamodule(model_a)

        pipeline = QuantumPipeline([PipelineStage(model_a)], mode="sequential")

        with pytest.raises(ValueError, match="not valid for sequential pipeline"):
            pipeline.fit(datamodule=dm, trainers=None, fit_mode="independent")
