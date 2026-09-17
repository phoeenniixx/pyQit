"""Tests for Trainer orchestration, backend loops and callbacks."""

import os

import numpy as np
import pytest

import pyqit
from pyqit.core.callbacks import BaseCallback, ModelCheckpoint
from pyqit.core.trainer import Trainer, TrainingHistory
from pyqit.core.trainer.loops import PennyLaneLoop
from pyqit.data.datamodule import DataModule
from pyqit.models.classification.vqc import VQCClassifier
from pyqit.tests.scenarios import make_scenario
from pyqit.utils.utils import _hard_labels, _restore_weights

BACKENDS = ["pennylane", "torch"]
SIMULATORS = ["default.qubit", "lightning.qubit", "default.mixed", "reference.qubit"]
QISKIT_SIMULATORS = ["qiskit.aer", "qiskit.basicsim"]


def _require(backend):
    if backend == "torch":
        pytest.importorskip("torch")
        pytest.importorskip("lightning")
    pyqit.set_backend(backend)


def _model(n_qubits=3, n_layers=1, **kwargs):
    pyqit.set_seed(42)
    return VQCClassifier(n_qubits=n_qubits, n_layers=n_layers, **kwargs)


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


def _mse_gradient(model, X, y):
    """Flat MSE gradient of ``model`` at its current weights, either backend."""
    from pyqit.core.losses import get_loss_fn

    if any(type(w).__module__.startswith("torch") for w in model.weights.values()):
        import torch

        for param in model.weights.values():
            param.grad = None
        loss = get_loss_fn("mse", backend="torch")
        loss(model.forward(torch.as_tensor(X)), torch.as_tensor(y)).backward()
        return np.concatenate(
            [_to_numpy(p.grad).ravel() for p in model.weights.values()]
        )

    import pennylane as qml
    import pennylane.numpy as pnp

    loss = get_loss_fn("mse", backend="pennylane")
    keys = list(model.weights)

    def cost(*weights):
        preds = model.forward(
            pnp.array(X, requires_grad=False), **dict(zip(keys, weights))
        )
        return loss(preds, pnp.array(y, requires_grad=False))

    grads = qml.grad(cost)(
        *[pnp.array(model.weights[k], requires_grad=True) for k in keys]
    )
    return np.concatenate([np.asarray(g).ravel() for g in grads])


@pytest.mark.parametrize("backend", BACKENDS)
def test_reported_loss_is_the_mean_over_rows_when_the_last_batch_is_short(backend):
    """13 test rows in batches of 8 and 5; a mean of batch means would be off."""
    _require(backend)
    model = _model()
    dm = _dm(n_samples=26, split=(0.5, 0.0, 0.5))
    trainer = Trainer(max_epochs=1, verbose=0, batch_size=8)
    trainer.fit(model, dm)
    X = dm.X_test
    if backend == "torch":
        import torch

        X = torch.as_tensor(X, dtype=torch.float32)

    probs = _to_numpy(model.forward(X))

    assert trainer.test(model, dm)["test_loss"] == pytest.approx(
        np.mean((probs - dm.y_test.ravel()) ** 2), rel=1e-5
    )


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


@pytest.mark.parametrize("backend", BACKENDS)
def test_test_scores_the_test_split_with_the_configured_loss(backend):
    """``test_acc`` must agree with predict-then-count, or one of them is wrong."""
    _require(backend)
    model = _model()
    dm = _dm(n_samples=40)
    trainer = Trainer(max_epochs=1, verbose=0, loss_fn="cross_entropy")
    trainer.fit(model, dm)

    metrics = trainer.test(model, dm)

    preds = _to_numpy(trainer.predict(model, dm))
    expected_acc = np.mean(_hard_labels(preds) == dm.y_test.astype(int).flatten())
    assert set(metrics) == {"test_loss", "test_acc"}
    assert metrics["test_acc"] == pytest.approx(expected_acc)
    assert np.isfinite(metrics["test_loss"])


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


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", SIMULATORS)
def test_finite_shots_train_at_the_default_batch_size(backend, device):
    """float32 torch inputs failed PennyLane's 1e-7 sampling check past ~8 rows."""
    _require(backend)
    pyqit.set_seed(42)
    model = VQCClassifier(n_qubits=3, n_layers=1, device=device, shots=100)

    history = Trainer(max_epochs=1, batch_size=32, verbose=0).fit(
        model, _dm(n_samples=64)
    )

    assert all(0.0 <= a <= 1.0 for a in history.train_acc)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", SIMULATORS[1:])
def test_every_analytic_simulator_gives_the_same_loss_curve(backend, device):
    """From the same starting weights, only how gradients are computed changes."""
    _require(backend)
    reference, candidate = _model(), _model(device=device)
    _restore_weights(
        candidate, {k: np.array(_to_numpy(v)) for k, v in reference.weights.items()}
    )

    curves = [
        Trainer(max_epochs=2, verbose=0).fit(model, _dm())
        for model in (reference, candidate)
    ]

    for metric in ("train_loss", "val_loss"):
        np.testing.assert_allclose(
            getattr(curves[1], metric), getattr(curves[0], metric), rtol=1e-5
        )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "device, shots, method",
    [
        ("default.qubit", None, "backprop"),
        ("default.qubit", 100, "parameter-shift"),
        ("lightning.qubit", None, "adjoint"),
        ("reference.qubit", None, "parameter-shift"),
    ],
)
def test_resolved_diff_method_follows_the_device_and_shots(
    backend, device, shots, method
):
    """What "best" becomes is what a user on hardware pays for."""
    _require(backend)
    model = _model(device=device, shots=shots)

    assert model.diff_methods(_dm().setup(n_qubits=3).X_train[:1]) == {
        "main_circuit": method
    }


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "device, per_sample", [("default.qubit", 1), ("reference.qubit", 1 + 2 * 9)]
)
def test_bp_check_reports_its_circuit_executions(backend, device, per_sample):
    """One execution per gradient under backprop; 1 + 2 per parameter otherwise."""
    from pyqit.utils.diagnostic import check_barren_plateau

    _require(backend)
    model = _model(device=device)
    dm = _dm().setup(n_qubits=3, encoder=type(model.embedding_obj))

    result = check_barren_plateau(model, dm, num_samples=4, plot=False)

    assert result.n_executions == 4 * per_sample
    assert "Circuit Executions" in repr(result)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", QISKIT_SIMULATORS)
def test_qiskit_simulators_train_and_agree_with_default_qubit_within_shot_noise(
    backend, device
):
    """Check if qiskit plugin work correctly."""
    pytest.importorskip("pennylane_qiskit")
    _require(backend)
    reference, candidate = _model(), _model(device=device, shots=4000)
    _restore_weights(
        candidate, {k: np.array(_to_numpy(v)) for k, v in reference.weights.items()}
    )
    dm = _dm().setup(n_qubits=3, encoder=type(reference.embedding_obj))
    X, y = dm.X_train[:4], dm.y_train[:4].astype(np.float64)
    X_in = X
    if backend == "torch":
        import torch

        X_in = torch.as_tensor(X)

    np.testing.assert_allclose(
        _to_numpy(candidate.forward(X_in)),
        _to_numpy(reference.forward(X_in)),
        atol=0.05,
    )
    np.testing.assert_allclose(
        _mse_gradient(candidate, X, y), _mse_gradient(reference, X, y), atol=0.02
    )
    history = Trainer(max_epochs=1, verbose=0).fit(candidate, dm)
    assert np.isfinite(history.train_loss).all()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", SIMULATORS[1:])
def test_same_seed_gives_the_same_starting_weights_on_every_simulator(backend, device):
    """Devices draw from numpy's RNG at construction, so weights come first."""
    _require(backend)
    reference, candidate = _model(), _model(device=device)

    for key, value in reference.weights.items():
        np.testing.assert_array_equal(
            _to_numpy(candidate.weights[key]), _to_numpy(value)
        )
