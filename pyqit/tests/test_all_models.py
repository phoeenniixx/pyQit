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
        self,
        model_instance,
        n_samples=16,
        batch_size=8,
        split=(0.6, 0.2, 0.2),
        backend="pennylane",
    ):
        """Helper to generate a dm that perfectly matches the model's architecture."""
        n_features = getattr(model_instance, "n_qubits", 4)
        n_classes = getattr(model_instance, "n_classes", 2)

        scenario = make_scenario(
            n_samples=n_samples, n_features=n_features, n_classes=n_classes, seed=42
        )

        return DataModule(
            X=scenario["X"],
            y=scenario["y"],
            batch_size=batch_size,
            split=split,
            backend=backend,
        )

    @pytest.mark.parametrize("backend", ["pennylane", "torch"])
    def test_basic_trainer_flow(self, object_instance, trainer_kwargs, backend):
        """
        Verifies the foundational architecture:
        DataModule feeds Trainer -> Trainer updates Model -> Trainer predicts.
        """
        if backend == "torch":
            pytest.importorskip("torch")
            pytest.importorskip("lightning")

        model = object_instance.clone()
        trainer_args = {
            "backend_type": backend,
            "max_epochs": 1,
            "learning_rate": 0.1,
            "enable_checkpointing": False,
            "logger": False,
        }
        trainer_args.update(trainer_kwargs)

        dm = self._get_matching_datamodule(
            model, n_samples=16, batch_size=8, backend=backend
        )
        trainer = Trainer(**trainer_args)

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

    @pytest.mark.parametrize("backend", ["pennylane", "torch"])
    def test_checkpointing(
        self, object_instance, trainer_kwargs, backend, tmp_path, monkeypatch
    ):
        """
        Verifies that both backends successfully save model checkpoints to disk
        when enable_checkpointing=True, without crashing the test suite.
        """
        if backend == "torch":
            pytest.importorskip("torch")
            pytest.importorskip("lightning")

        monkeypatch.chdir(tmp_path)

        model = object_instance.clone()

        trainer_args = {
            "backend_type": backend,
            "max_epochs": 1,
            "learning_rate": 0.1,
            "enable_checkpointing": True,
            "logger": True,
        }
        trainer_args.update(trainer_kwargs)

        dm = self._get_matching_datamodule(
            model, n_samples=16, batch_size=8, backend=backend
        )

        trainer = Trainer(**trainer_args)

        trainer.fit(model, datamodule=dm)

        import glob
        import os

        if backend == "pennylane":
            assert os.path.exists(
                "pyqit_pennylane_checkpoint.npz"
            ), "PennyLane checkpoint was not created!"

        elif backend == "torch":
            ckpt_files = glob.glob("lightning_logs/**/*.ckpt", recursive=True)
            assert len(ckpt_files) > 0, "Lightning PyTorch checkpoint was not created!"
