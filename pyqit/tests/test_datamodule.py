import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import pyqit
from pyqit.core.trainer import Trainer
from pyqit.data.datamodule import DataModule
from pyqit.models.classification.vqc import VQCClassifier
from pyqit.tests.scenarios import make_scenario
from pyqit.tests.test_trainer import BACKENDS, _require, _to_numpy


def _data(n_samples=12, n_features=2):
    scenario = make_scenario(n_samples=n_samples, n_features=n_features, seed=0)
    return scenario["X"], scenario["y"]


def _model():
    pyqit.set_seed(0)
    return VQCClassifier(n_qubits=2, n_layers=1)


def _record_inputs(model):
    """Keep a copy of every batch ``model.forward`` is handed."""
    seen, forward = [], model.forward

    def recording(X, **kwargs):
        seen.append(np.array(_to_numpy(X)))
        return forward(X, **kwargs)

    model.forward = recording
    return seen


@pytest.mark.parametrize(
    "method, reference", [("minmax", MinMaxScaler), ("zscore", StandardScaler)]
)
def test_val_and_test_are_scaled_with_train_statistics(method, reference):
    """Same as sklearn's scalers fit on train alone, so nothing leaks from val/test."""
    _require("pennylane")
    X, y = _data(n_samples=40, n_features=3)
    raw = DataModule(X, y).setup()
    scaled = DataModule(X, y, normalize=method).setup()

    scaler = reference().fit(raw.X_train)
    for raw_split, scaled_split in [
        (raw.X_train, scaled.X_train),
        (raw.X_val, scaled.X_val),
        (raw.X_test, scaled.X_test),
    ]:
        np.testing.assert_allclose(scaled_split, scaler.transform(raw_split))


@pytest.mark.parametrize("backend", BACKENDS)
def test_transform_runs_after_normalizing_and_prescaling(backend):
    """The circuit trains on transform(prescale(normalize(X))) on either backend."""
    _require(backend)
    X, y = _data()
    model = _model()
    seen = _record_inputs(model)
    dm = DataModule(
        X * 10 + 5,
        y,
        normalize="minmax",
        split=(1.0, 0.0, 0.0),
        transform=lambda x: x / 2,
    )

    Trainer(max_epochs=1, verbose=0).fit(model, dm)

    # minmax puts each train column on [0, 1], angle prescaling on [0, pi], and
    # the transform halves that.
    trained_on = np.concatenate(seen)
    np.testing.assert_allclose(trained_on.min(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(trained_on.max(axis=0), np.pi / 2, rtol=1e-6)


@pytest.mark.parametrize(
    "split",
    [
        (0.7, 0.15, 0.15),
        (0.75, 0.0, 0.25),
        pytest.param(
            (0.75, 0.25, 0.0),
            marks=pytest.mark.xfail(
                strict=True,
                reason="int() truncation drops the remainder row when there is "
                "no test split; see notes.md",
            ),
        ),
    ],
)
def test_seeded_split_is_reproducible_and_keeps_every_sample(split):
    _require("pennylane")
    ids = np.arange(10.0).reshape(-1, 1)
    y = np.arange(10) % 2

    first = DataModule(ids, y, split=split, seed=3).setup()
    again = DataModule(ids, y, split=split, seed=3).setup()

    def rows(dm):
        return [None if s is None else s.tolist() for s in dm.splits]

    assert rows(first) == rows(again)
    kept = [s for s in (first.X_train, first.X_val, first.X_test) if s is not None]
    assert sorted(np.concatenate(kept).ravel()) == list(range(10))


def test_stratified_split_keeps_a_rare_class_in_every_split():
    """A 20% minority stays 20% of train, val and test alike."""
    _require("pennylane")
    X = np.random.default_rng(0).random((50, 2))
    y = np.array([0.0] * 40 + [1.0] * 10)

    dm = DataModule(X, y, split=(0.6, 0.2, 0.2), stratify=True).setup()

    for split_y in (dm.y_train, dm.y_val, dm.y_test):
        assert split_y.mean() == pytest.approx(0.2)


@pytest.mark.parametrize("backend", BACKENDS)
def test_trainer_keeps_a_users_split_but_applies_its_batch_size(backend):
    """Setting up by hand to inspect the data must not re-split it on fit."""
    _require(backend)
    X, y = _data(n_samples=20)
    model = _model()
    dm = DataModule(X, y, batch_size=16)
    dm.setup(n_qubits=model.n_qubits, encoder=type(model.embedding_obj))
    inspected = dm.X_train.copy()
    seen = _record_inputs(model)

    Trainer(max_epochs=1, batch_size=4, verbose=0).fit(model, dm)

    np.testing.assert_array_equal(dm.X_train, inspected)
    assert max(len(batch) for batch in seen) == 4


def test_retraining_after_reconfigure_uses_the_new_normalization():
    """The model's prescaling persists; only what the user changed moves."""
    _require("pennylane")
    X, y = _data(n_samples=20)
    model = _model()
    dm = DataModule(X, y, normalize="minmax")
    Trainer(max_epochs=1, verbose=0).fit(model, dm)

    dm.reconfigure(normalize="zscore")
    Trainer(max_epochs=1, verbose=0).fit(model, dm)

    # Dividing out the angle prescaling must leave standardized columns.
    standardized = dm.X_train / np.pi
    np.testing.assert_allclose(standardized.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(standardized.std(axis=0), 1.0)

    # A forced re-setup that omits the encoder still prescales.
    before = dm.X_train.copy()
    dm.setup(force=True)
    np.testing.assert_array_equal(dm.X_train, before)


_PENNYLANE_SHUFFLE = pytest.mark.xfail(
    strict=True,
    reason="PennyLaneLoop hardcodes shuffle=True and _NumpyLoader reseeds on every "
    "epoch, so the order neither follows the setting nor changes; see notes.md",
)


@pytest.mark.parametrize(
    "backend", [pytest.param("pennylane", marks=_PENNYLANE_SHUFFLE), "torch"]
)
@pytest.mark.parametrize("shuffle", [False, True])
def test_training_batch_order_follows_the_shuffle_setting(backend, shuffle):
    _require(backend)
    X, y = _data(n_samples=12)
    model = _model()
    seen = _record_inputs(model)
    dm = DataModule(X, y, split=(1.0, 0.0, 0.0), batch_size=4, shuffle=shuffle)

    Trainer(max_epochs=2, batch_size=4, verbose=0).fit(model, dm)

    # 12 rows at batch 4 with no val split: 3 training batches per epoch.
    epoch1, epoch2 = np.concatenate(seen[:3]), np.concatenate(seen[3:])
    if shuffle:
        assert not np.allclose(epoch1, epoch2)
    else:
        np.testing.assert_allclose(epoch1, dm.X_train, rtol=1e-6)
        np.testing.assert_allclose(epoch2, dm.X_train, rtol=1e-6)


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(
            "pennylane",
            marks=pytest.mark.xfail(
                strict=True,
                reason="DataModule loaders never receive drop_last; only the "
                "Lightning adapter honours it; see notes.md",
            ),
        ),
        "torch",
    ],
)
def test_drop_last_leaves_no_ragged_training_batch(backend):
    _require(backend)
    X, y = _data(n_samples=10)
    model = _model()
    seen = _record_inputs(model)
    dm = DataModule(X, y, split=(1.0, 0.0, 0.0), batch_size=4, drop_last=True)

    Trainer(max_epochs=1, batch_size=4, verbose=0).fit(model, dm)

    assert [len(batch) for batch in seen] == [4, 4]


def test_csv_label_column_in_the_middle_does_not_leak_into_features(tmp_path):
    _require("pennylane")
    df = pd.DataFrame(
        {"f0": [0.1, 0.2, 0.3, 0.4], "label": [0, 1, 0, 1], "f1": [1.0, 2.0, 3.0, 4.0]}
    )
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    dm = DataModule.from_csv(path, label_col="label")

    np.testing.assert_array_equal(dm.X_raw, df[["f0", "f1"]].to_numpy())
    np.testing.assert_array_equal(dm.y_raw, df["label"].to_numpy())
