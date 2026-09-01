"""Tests for Trainer orchestration, backend loops and callbacks."""

import numpy as np
import pytest

import pyqit
from pyqit.core.callbacks import BaseCallback, EarlyStopping, ModelCheckpoint
from pyqit.core.trainer import Trainer, TrainingHistory
from pyqit.core.trainer.loops import PennyLaneLoop, loop_registry
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


def test_trainer_clone_preserves_params_verbatim():
    """skbase compares params across a clone, so a mutating __init__ fails here."""
    trainer = Trainer(max_epochs=7, optimizer="ADAM", verbose=0)

    assert trainer.clone().get_params() == trainer.get_params()
    assert trainer.optimizer == "ADAM"


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", ["sgd", "SGD"])
def test_optimizer_name_is_case_insensitive(backend, name):
    """Trainer stores it verbatim, so each backend must case-fold at use."""
    _require(backend)

    if backend == "torch":
        from pyqit.core.adapters.lightning import _LightningModelAdapter
        from pyqit.core.losses import get_loss_fn

        adapter = _LightningModelAdapter(
            _model(), 0.1, name, get_loss_fn("mse", backend="torch")
        )
        assert type(adapter.configure_optimizers()).__name__ == "SGD"
    else:
        trainer = Trainer(optimizer=name, learning_rate=0.1)
        optimizer = PennyLaneLoop._make_optimizer(trainer)
        assert type(optimizer).__name__ == "GradientDescentOptimizer"


def test_both_loops_are_discovered():
    """Registration is by tag; a loop in a private module would silently vanish."""
    assert {"pennylane", "torch"} <= set(loop_registry())


def test_history_tracks_best_on_val_loss_or_train_loss():
    """Without a validation split val_loss is NaN, which loses every comparison
    and used to leave best_score at inf."""
    with_val = TrainingHistory()
    with_val.record(0, train_loss=0.1, val_loss=0.9)
    with_val.record(1, train_loss=0.8, val_loss=0.2)

    assert with_val.best_metric == "val_loss"
    assert with_val.best_epoch == 1

    without_val = TrainingHistory()
    for epoch, loss in enumerate([1.0, 0.4, 0.7]):
        without_val.record(epoch, train_loss=loss)

    assert without_val.best_metric == "train_loss"
    assert without_val.best_epoch == 1
    assert without_val.best_score == pytest.approx(0.4)


def test_backend_kwargs_rejected_on_pennylane():
    """Unambiguously a Lightning setting on a backend with no Lightning."""
    pyqit.set_backend("pennylane")
    trainer = Trainer(max_epochs=1, verbose=0, backend_kwargs={"devices": 2})

    with pytest.raises(ValueError, match="not supported on the 'pennylane'"):
        trainer.fit(_model(), _dm())


def test_backend_kwargs_forwarded_on_torch():
    _require("torch")
    trainer = Trainer(
        max_epochs=1, verbose=0, backend_kwargs={"gradient_clip_val": 0.5}
    )

    assert len(trainer.fit(_model(), _dm()).train_loss) == 1


def test_backend_kwargs_reserved_key_raises():
    """Otherwise Lightning gets max_epochs twice and raises a TypeError."""
    _require("torch")
    trainer = Trainer(max_epochs=1, verbose=0, backend_kwargs={"max_epochs": 9})

    with pytest.raises(ValueError, match="backend_kwargs may not set"):
        trainer.fit(_model(), _dm())


def test_logger_only_warns_on_pennylane():
    """The run is still correct, so this degrades rather than failing."""
    pyqit.set_backend("pennylane")
    trainer = Trainer(max_epochs=1, verbose=0, logger=True)

    with pytest.warns(UserWarning, match="is ignored on the 'pennylane'"):
        trainer.fit(_model(), _dm())


@pytest.mark.parametrize("backend", BACKENDS)
def test_early_stopping_halts_both_backends(backend):
    _require(backend)
    stopper = EarlyStopping(monitor="train_loss", patience=0, verbose=False)
    # lr=0 guarantees no improvement, so the stop is the only way out.
    history = Trainer(
        max_epochs=20, learning_rate=0.0, verbose=0, callbacks=[stopper]
    ).fit(_model(), _dm())

    assert stopper.stopped_epoch is not None
    assert len(history.train_loss) < 20


@pytest.mark.parametrize("backend", BACKENDS)
def test_checkpoint_restores_best_weights_into_the_model(backend, tmp_path):
    """``update_weights`` is a no-op under torch, so this is the real check."""
    _require(backend)
    checkpoint = ModelCheckpoint(dirpath=str(tmp_path), monitor="val_loss")
    model = _model()
    Trainer(max_epochs=4, learning_rate=0.3, verbose=0, callbacks=[checkpoint]).fit(
        model, _dm()
    )

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
def test_user_callback_sees_every_epoch(backend):
    _require(backend)

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


def _drive(callback, values):
    """Feed ``values`` to a callback as metric ``m``; return the stopping epoch."""
    from pyqit.core.callbacks import LoopState

    class _SilentReporter:
        def warn(self, *args, **kwargs):
            pass

    state = LoopState(
        model=None,
        datamodule=None,
        history=None,
        reporter=_SilentReporter(),
        max_epochs=len(values),
    )
    for epoch, value in enumerate(values):
        state.epoch = epoch
        state.metrics = {"m": value}
        callback.on_epoch_end(state)
        if state.stop:
            return epoch
    return None


@pytest.mark.parametrize("patience, expected_stop", [(0, 2), (2, 3)])
def test_patience_counts_the_way_lightning_counts_it(patience, expected_stop):
    """Lightning stops once the wait *reaches* patience; ``>`` cost an extra epoch.

    Best is 0.5 at epoch 1, so epoch 2 is the first non-improvement (wait 1) and
    epoch 3 the second (wait 2).
    """
    values = [1.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    stopper = EarlyStopping(monitor="m", patience=patience, verbose=False)

    assert _drive(stopper, values) == expected_stop


def test_check_finite_stops_on_nan():
    """A diverged circuit yields NaN, which never 'fails to improve'."""
    stopper = EarlyStopping(monitor="m", patience=99, verbose=False)
    assert _drive(stopper, [1.0, float("nan")]) == 1
    assert "not finite" in stopper.stopping_reason

    lenient = EarlyStopping(monitor="m", patience=99, check_finite=False, verbose=False)
    assert _drive(lenient, [1.0, float("nan")]) is None


def test_hard_label_rule_is_identical_across_backends():
    """One rule, or the two backends report different accuracy silently."""
    torch = pytest.importorskip("torch")
    from pyqit.utils.utils import _hard_labels

    for array in (
        np.array([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6]]),
        np.array([[0.2], [0.5], [0.7]]),
        np.array([0.2, 0.5, 0.7]),
    ):
        np.testing.assert_array_equal(
            _to_numpy(_hard_labels(array)),
            _to_numpy(_hard_labels(torch.as_tensor(array))),
        )


def test_eval_train_acc_false_drops_only_train_accuracy():
    """The opt-out must not disturb any other recorded metric."""
    pyqit.set_backend("pennylane")
    history = Trainer(max_epochs=2, verbose=0, eval_train_acc=False).fit(
        _model(), _dm()
    )

    assert all(np.isnan(a) for a in history.train_acc)
    assert not any(np.isnan(v) for v in history.val_loss)
    assert not any(np.isnan(v) for v in history.val_acc)
    assert not any(np.isnan(t) for t in history.train_loss)


def test_predict_prescales_an_unfitted_datamodule():
    """Predict must shape inputs the way fit does, or the circuit sees raw data."""
    pyqit.set_backend("pennylane")
    model = _model()

    prepared = _dm()
    prepared.setup(
        stage="predict", n_qubits=model.n_qubits, encoder=type(model.embedding_obj)
    )
    expected = Trainer(verbose=0).predict(model, prepared)

    fresh = _dm()
    actual = Trainer(verbose=0).predict(model, fresh)

    np.testing.assert_allclose(_to_numpy(actual), _to_numpy(expected), rtol=1e-6)


def test_lightning_adapter_reports_an_unprepared_datamodule():
    """Lightning's setup hook cannot supply n_qubits/encoder, so it must not guess."""
    _require("torch")
    model = _model()

    bare = _dm().to_lightning()
    with pytest.raises(RuntimeError, match="not set up"):
        bare.setup("fit")

    dm = _dm()
    dm.setup(stage="fit", n_qubits=model.n_qubits, encoder=type(model.embedding_obj))
    ready = dm.to_lightning()
    ready.setup("fit")

    assert len(next(iter(ready.train_dataloader()))) == 2


def _hide_torch(monkeypatch, module):
    """Make ``module``'s soft-dependency check report torch as missing."""
    import skbase.utils.dependencies as dep

    real = dep._check_soft_dependencies

    def fake(*packages, **kwargs):
        flat = [p for group in packages for p in _as_list(group)]
        if "torch" in flat:
            return False
        return real(*packages, **kwargs)

    monkeypatch.setattr(module, "_check_soft_dependencies", fake)


def _as_list(value):
    return value if isinstance(value, (list, tuple)) else [value]


def test_set_backend_torch_requires_torch(monkeypatch):
    """Otherwise the model is built the pennylane way and fails later elsewhere."""
    from pyqit.core import config

    pyqit.set_backend("pennylane")
    _hide_torch(monkeypatch, config)

    with pytest.raises(ImportError, match="requires torch"):
        pyqit.set_backend("torch")

    assert pyqit.get_backend() == "pennylane"


def test_predict_torch_format_requires_torch(monkeypatch):
    """Reachable from the pennylane backend, so set_backend cannot cover it."""
    from pyqit.core.trainer import trainer as trainer_module

    pyqit.set_backend("pennylane")
    _hide_torch(monkeypatch, trainer_module)

    with pytest.raises(ImportError, match="return_format='torch' requires torch"):
        Trainer(verbose=0).predict(_model(), _dm(), return_format="torch")
