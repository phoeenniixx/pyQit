"""Tests for Trainer orchestration, backend loops and callbacks.

These are not class-discovery tests: none of the discovery suites filter on
``object_type`` ``"trainer"``, ``"training_loop"`` or ``"callback"``, so the
behaviour here would otherwise be uncovered.
"""

import numpy as np
import pytest

import pyqit
from pyqit.core.callbacks import EarlyStopping, ModelCheckpoint
from pyqit.core.trainer import Trainer, TrainingHistory
from pyqit.core.trainer.loops import loop_registry
from pyqit.data.datamodule import DataModule
from pyqit.models.classification.vqc import VQCClassifier
from pyqit.tests.scenarios import make_scenario

BACKENDS = ["pennylane", "torch"]


def _require(backend):
    if backend == "torch":
        pytest.importorskip("torch")
        pytest.importorskip("lightning")
    pyqit.set_backend(backend)


def _model(n_qubits=3, n_layers=1):
    pyqit.set_seed(42)
    return VQCClassifier(n_qubits=n_qubits, n_layers=n_layers)


def _dm(n_qubits=3, n_samples=16, batch_size=8, split=(0.6, 0.2, 0.2)):
    scenario = make_scenario(
        n_samples=n_samples, n_features=n_qubits, n_classes=2, seed=42
    )
    return DataModule(
        X=scenario["X"], y=scenario["y"], batch_size=batch_size, split=split
    )


def _to_numpy(value):
    if type(value).__module__.startswith("torch"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class TestTrainerObject:
    """Trainer participates in the skbase object model like everything else."""

    def test_is_cloneable_pyqit_object(self):
        trainer = Trainer(max_epochs=7, learning_rate=0.5, verbose=0)
        params = trainer.get_params()

        assert params["max_epochs"] == 7
        assert params["learning_rate"] == 0.5
        assert trainer.clone().get_params() == params

    def test_constructor_args_are_stored_verbatim(self):
        """skbase's contract, and what forced learning_rate/optimizer to change."""
        trainer = Trainer(optimizer="ADAM", learning_rate=0.25)

        assert trainer.optimizer == "ADAM", "optimizer must not be mutated in __init__"
        assert trainer.learning_rate == 0.25

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize("name", ["sgd", "SGD", "Sgd"])
    def test_optimizer_name_is_case_insensitive(self, backend, name):
        """Trainer stores it verbatim, so each loop must case-fold at use."""
        _require(backend)

        if backend == "torch":
            from pyqit.core.adapters.lightning import _LightningModelAdapter
            from pyqit.core.losses import get_loss_fn

            adapter = _LightningModelAdapter(
                _model(), 0.1, name, get_loss_fn("mse", backend="torch")
            )
            assert type(adapter.configure_optimizers()).__name__ == "SGD"
        else:
            from pyqit.core.trainer.loops import PennyLaneLoop

            trainer = Trainer(optimizer=name, learning_rate=0.1)
            optimizer = PennyLaneLoop._make_optimizer(trainer)
            assert type(optimizer).__name__ == "GradientDescentOptimizer"

    def test_both_loops_are_discovered(self):
        assert {"pennylane", "torch"} <= set(loop_registry())


class TestTrainingHistory:
    def test_best_falls_back_to_train_loss_without_validation(self):
        """NaN loses every comparison, which used to leave best_score at inf."""
        history = TrainingHistory()
        for epoch, loss in enumerate([1.0, 0.4, 0.7]):
            history.record(epoch, train_loss=loss)

        assert history.best_metric == "train_loss"
        assert history.best_epoch == 1
        assert history.best_score == pytest.approx(0.4)

    def test_val_loss_is_preferred_when_present(self):
        history = TrainingHistory()
        history.record(0, train_loss=0.1, val_loss=0.9)
        history.record(1, train_loss=0.8, val_loss=0.2)

        assert history.best_metric == "val_loss"
        assert history.best_epoch == 1


class TestBackendKwargs:
    def test_rejected_on_pennylane(self):
        """Unambiguously a Lightning setting on a backend with no Lightning."""
        pyqit.set_backend("pennylane")
        trainer = Trainer(max_epochs=1, verbose=0, backend_kwargs={"devices": 2})

        with pytest.raises(ValueError, match="not supported on the 'pennylane'"):
            trainer.fit(_model(), _dm())

    def test_forwarded_on_torch(self):
        _require("torch")
        trainer = Trainer(
            max_epochs=1, verbose=0, backend_kwargs={"gradient_clip_val": 0.5}
        )
        assert len(trainer.fit(_model(), _dm()).train_loss) == 1

    def test_reserved_key_raises_rather_than_a_duplicate_kwarg_typeerror(self):
        _require("torch")
        trainer = Trainer(max_epochs=1, verbose=0, backend_kwargs={"max_epochs": 9})

        with pytest.raises(ValueError, match="backend_kwargs may not set"):
            trainer.fit(_model(), _dm())

    def test_logger_only_warns_on_pennylane(self):
        """The run is still correct, so this degrades rather than failing."""
        pyqit.set_backend("pennylane")
        trainer = Trainer(max_epochs=1, verbose=0, logger=True)

        with pytest.warns(UserWarning, match="only partly honoured"):
            trainer.fit(_model(), _dm())


class TestCallbacks:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_early_stopping_halts_both_backends(self, backend):
        _require(backend)
        stopper = EarlyStopping(monitor="train_loss", patience=0, verbose=False)
        # lr=0 guarantees no improvement, so the stop is the only way out.
        history = Trainer(
            max_epochs=20, learning_rate=0.0, verbose=0, callbacks=[stopper]
        ).fit(_model(), _dm())

        assert stopper.stopped_epoch is not None
        assert len(history.train_loss) < 20

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_checkpoint_restores_best_weights_into_the_model(self, backend, tmp_path):
        """``update_weights`` is a no-op under torch, so this is the real check."""
        _require(backend)
        checkpoint = ModelCheckpoint(dirpath=str(tmp_path), monitor="val_loss")
        model = _model()
        Trainer(max_epochs=4, learning_rate=0.3, verbose=0, callbacks=[checkpoint]).fit(
            model, _dm()
        )

        assert checkpoint.best_path is not None

        if backend == "torch":
            import torch

            saved = torch.load(checkpoint.best_path, weights_only=False)["state_dict"]
        else:
            saved = dict(np.load(checkpoint.best_path))

        assert set(saved) == set(model.weights)
        for key, value in saved.items():
            np.testing.assert_allclose(
                _to_numpy(model.weights[key]),
                _to_numpy(value),
                err_msg=f"best weights for {key!r} were not restored into the model",
            )

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_user_callback_sees_every_epoch(self, backend):
        _require(backend)
        from pyqit.core.callbacks import BaseCallback

        class Spy(BaseCallback):
            def __init__(self):
                self.epochs = []
                self.started = self.ended = 0
                super().__init__()

            def on_fit_start(self, state):
                self.started += 1

            def on_epoch_end(self, state):
                self.epochs.append((state.epoch, sorted(state.metrics)))

            def on_fit_end(self, state):
                self.ended += 1

        spy = Spy()
        Trainer(max_epochs=3, verbose=0, callbacks=[spy]).fit(_model(), _dm())

        assert spy.started == 1 and spy.ended == 1
        assert [e for e, _ in spy.epochs] == [0, 1, 2]
        for _, keys in spy.epochs:
            assert {"train_loss", "val_loss", "train_acc", "val_acc"} <= set(keys)


class TestPennyLaneEvaluation:
    def test_eval_train_acc_false_drops_only_train_accuracy(self):
        """The opt-out must not disturb any other recorded metric."""
        pyqit.set_backend("pennylane")
        history = Trainer(max_epochs=2, verbose=0, seed=42, eval_train_acc=False).fit(
            _model(), _dm()
        )

        assert all(np.isnan(a) for a in history.train_acc)
        assert not any(np.isnan(v) for v in history.val_loss)
        assert not any(np.isnan(v) for v in history.val_acc)
        assert not any(np.isnan(t) for t in history.train_loss)

    def test_merged_evaluation_matches_separate_passes(self):
        """One pass must give the same numbers two passes did."""
        pyqit.set_backend("pennylane")
        from pyqit.core.losses import get_loss_fn
        from pyqit.core.trainer.loops import PennyLaneLoop

        model = _model()
        dm = _dm()
        dm.setup(stage="fit", n_qubits=3, encoder=type(model.embedding_obj))
        loader = dm.val_loader(shuffle=False)
        loss_fn = get_loss_fn("mse", backend="pennylane")

        loss, acc = PennyLaneLoop._evaluate(model, loader, loss_fn)
        loss_only, _ = PennyLaneLoop._evaluate(model, loader, loss_fn)
        _, acc_only = PennyLaneLoop._evaluate(model, loader)

        assert loss == pytest.approx(loss_only)
        assert acc == pytest.approx(acc_only)
        assert np.isnan(PennyLaneLoop._evaluate(model, None)[0])
