"""Tests for Trainer orchestration, backend loops and callbacks."""

import os

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


@pytest.mark.parametrize(
    "kwargs, expectation",
    [
        ({"backend_kwargs": {"devices": 2}}, "raise"),
        ({"logger": True}, "warn"),
    ],
)
def test_pennylane_reports_settings_it_cannot_honour(kwargs, expectation):
    """The alternative is a run that quietly ignored what was asked for."""
    pyqit.set_backend("pennylane")
    trainer = Trainer(max_epochs=1, verbose=0, **kwargs)

    if expectation == "raise":
        with pytest.raises(ValueError, match="not supported on the 'pennylane'"):
            trainer.fit(_model(), _dm())
    else:
        # Warns rather than raising: the resulting model is still correct.
        with pytest.warns(UserWarning, match="is ignored on the 'pennylane'"):
            trainer.fit(_model(), _dm())


def test_backend_kwargs_actually_reach_lightning():
    """An unknown key must surface from Lightning; silence means it was dropped."""
    _require("torch")
    trainer = Trainer(
        max_epochs=1, verbose=0, backend_kwargs={"not_a_real_lightning_kwarg": 1}
    )

    with pytest.raises(TypeError, match="not_a_real_lightning_kwarg"):
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


@pytest.mark.parametrize("backend", BACKENDS)
def test_every_metric_is_recorded_on_both_backends(backend):
    """train_acc is scored from the training pass, so no metric is ever NaN."""
    _require(backend)
    history = Trainer(max_epochs=2, verbose=0).fit(_model(), _dm())

    for name in ("train_loss", "train_acc", "val_loss", "val_acc"):
        values = getattr(history, name)
        assert len(values) == 2
        assert not any(np.isnan(v) for v in values), f"{name} contains NaN"
    assert all(0.0 <= a <= 1.0 for a in history.train_acc)


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


@pytest.mark.parametrize("return_format", ["auto", "numpy", "torch", "pennylane"])
def test_predict_return_format(return_format):
    """Each explicit format is requestable regardless of the active backend.

    ``"auto"`` and ``"numpy"`` both collapse pennylane's native
    ``pnp.tensor`` predictions down to a bare ``ndarray`` via ``np.asarray``;
    ``"torch"`` and ``"pennylane"`` are opt-ins that instead preserve (or
    convert into) an autograd-carrying tensor type.
    """
    import pennylane.numpy as pnp

    expected_type = {
        "auto": np.ndarray,
        "numpy": np.ndarray,
        "torch": pytest.importorskip("torch").Tensor,
        "pennylane": pnp.tensor,
    }[return_format]

    pyqit.set_backend("pennylane")
    preds = Trainer(verbose=0).predict(_model(), _dm(), return_format=return_format)

    assert type(preds) is expected_type


def _load(path):
    if path.endswith(".ckpt"):
        import torch

        return torch.load(path, weights_only=False)["state_dict"]
    return dict(np.load(path))


def _weights_equal(a, b):
    return all(np.allclose(_to_numpy(a[k]), _to_numpy(b[k]), atol=1e-8) for k in a)


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_last_holds_the_final_epoch_not_the_best(backend, tmp_path):
    """Written before restore_best rewrites the model, or 'last' would be best."""
    _require(backend)
    checkpoint = ModelCheckpoint(
        dirpath=str(tmp_path), save_last=True, monitor="train_loss"
    )
    model = _model()
    Trainer(max_epochs=4, learning_rate=0.5, verbose=0, callbacks=[checkpoint]).fit(
        model, _dm()
    )

    last = _load(checkpoint.last_path)
    best = _load(checkpoint.best_path)

    # restore_best defaults on, so the model now holds the best epoch.
    assert _weights_equal(best, model.weights)
    assert not _weights_equal(last, best), "last must not be the restored best"


def test_save_last_only_writes_no_best_and_does_not_restore(tmp_path):
    """restore_best follows save_best, so last-only leaves the final weights."""
    pyqit.set_backend("pennylane")
    checkpoint = ModelCheckpoint(
        dirpath=str(tmp_path), save_best=False, save_last=True, monitor="train_loss"
    )
    model = _model()
    Trainer(max_epochs=3, learning_rate=0.5, verbose=0, callbacks=[checkpoint]).fit(
        model, _dm()
    )

    assert checkpoint.best_path is None
    assert sorted(p.name for p in tmp_path.iterdir()) == ["last.npz"]
    assert _weights_equal(_load(checkpoint.last_path), model.weights)


def test_every_n_epochs_writes_periodic_snapshots(tmp_path):
    pyqit.set_backend("pennylane")
    checkpoint = ModelCheckpoint(
        dirpath=str(tmp_path), every_n_epochs=2, monitor="train_loss"
    )
    Trainer(max_epochs=5, verbose=0, callbacks=[checkpoint]).fit(_model(), _dm())

    # Zero-based epochs, matching best_epoch: fires after epochs 1 and 3.
    assert [os.path.basename(p) for p in checkpoint.periodic_paths] == [
        "epoch1.npz",
        "epoch3.npz",
    ]


def test_train_accuracy_costs_no_extra_circuit_pass(monkeypatch):
    """It is scored from the training pass, not a second sweep of the split."""
    pyqit.set_backend("pennylane")
    model = _model()
    # 12 train / 4 val samples at batch_size 8 -> 2 train batches, 1 val batch.
    dm = _dm(n_samples=20, batch_size=8, split=(0.6, 0.2, 0.2))

    calls = []
    original = model.forward
    monkeypatch.setattr(
        model, "forward", lambda X, **kw: (calls.append(1), original(X, **kw))[1]
    )

    # batch_size lives on the Trainer, which overrides the DataModule at setup.
    history = Trainer(max_epochs=3, batch_size=8, verbose=0).fit(model, dm)

    # Per epoch: 2 gradient steps + 1 validation pass. A second sweep for
    # train_acc would add 2 more per epoch.
    assert len(calls) == 3 * 3
    assert all(0.0 <= a <= 1.0 for a in history.train_acc)
