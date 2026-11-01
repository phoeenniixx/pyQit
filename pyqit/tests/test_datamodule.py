import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import pyqit
from pyqit.core.embeddings import AmplitudeEmbedding
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
        (0.75, 0.25, 0.0),
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


@pytest.mark.parametrize("backend", BACKENDS)
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


@pytest.mark.parametrize("backend", BACKENDS)
def test_drop_last_leaves_no_ragged_training_batch(backend):
    _require(backend)
    X, y = _data(n_samples=10)
    model = _model()
    seen = _record_inputs(model)
    dm = DataModule(X, y, split=(1.0, 0.0, 0.0), batch_size=4, drop_last=True)

    Trainer(max_epochs=1, batch_size=4, verbose=0).fit(model, dm)

    assert [len(batch) for batch in seen] == [4, 4]


@pytest.mark.parametrize("backend", BACKENDS)
def test_drop_last_never_costs_validation_or_prediction_rows(backend):
    """Three val rows at batch 4 would otherwise leave nothing to validate on."""
    _require(backend)
    X, y = _data(n_samples=10)
    model = _model()
    dm = DataModule(X, y, split=(0.7, 0.3, 0.0), batch_size=4, drop_last=True)
    trainer = Trainer(max_epochs=1, batch_size=4, verbose=0)

    history = trainer.fit(model, dm)

    assert np.isfinite(history.val_loss).all()
    assert len(trainer.predict(model, dm)) == len(dm.y_val)


@pytest.mark.parametrize("backend", BACKENDS)
def test_predict_on_a_normalized_datamodule_that_was_never_fit_raises(backend):
    """It used to feed the circuit raw values; fitting on predict rows is wrong."""
    _require(backend)
    X, y = _data()
    model = _model()

    with pytest.raises(RuntimeError, match="never been fit"):
        Trainer(verbose=0).predict(model, DataModule(X * 10 + 5, y, normalize="minmax"))


@pytest.mark.parametrize("backend", BACKENDS)
def test_predict_applies_a_stateless_normalizer_without_a_fit(backend):
    _require(backend)
    X, y = _data()
    model = _model()
    seen = _record_inputs(model)

    Trainer(verbose=0).predict(model, DataModule(X * 10 + 5, y, normalize="l2"))

    rows = np.concatenate(seen) / np.pi
    np.testing.assert_allclose(np.linalg.norm(rows, axis=1), 1.0, rtol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_more_features_than_the_embedding_takes_raises(backend):
    """Extra columns used to be dropped without a word."""
    _require(backend)
    X, y = _data(n_features=3)

    with pytest.raises(ValueError, match="at most 2"):
        Trainer(max_epochs=1, verbose=0).fit(_model(), DataModule(X, y))


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


def _rows(shape, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random(shape), (np.arange(shape[0]) % 2).astype(float)


@pytest.mark.parametrize(
    "shape", [(30, 3, 2), (30, 2, 3, 2)], ids=["sequence", "image"]
)
@pytest.mark.parametrize(
    "method, reference", [("minmax", MinMaxScaler), ("zscore", StandardScaler)]
)
def test_nd_scaling_is_per_feature(shape, method, reference):
    """Each last-axis feature gets one statistic, pooled over positions, from train."""
    _require("pennylane")
    X, y = _rows(shape)
    raw = DataModule(X, y).setup()
    scaled = DataModule(X, y, normalize=method).setup()

    scaler = reference().fit(raw.X_train.reshape(-1, shape[-1]))
    for raw_split, scaled_split in [
        (raw.X_train, scaled.X_train),
        (raw.X_val, scaled.X_val),
    ]:
        flat = scaler.transform(raw_split.reshape(-1, shape[-1]))
        np.testing.assert_allclose(scaled_split, flat.reshape(raw_split.shape))


@pytest.mark.parametrize("method, order", [("l2", 2), ("l1", 1)])
def test_nd_unit_norm_covers_the_whole_sample(method, order):
    _require("pennylane")
    X, y = _rows((12, 3, 2))

    dm = DataModule(X * 10 + 5, y, normalize=method).setup()

    assert dm.X_train.shape[1:] == (3, 2)
    flat = dm.X_train.reshape(len(dm.X_train), -1)
    np.testing.assert_allclose(np.linalg.norm(flat, ord=order, axis=1), 1.0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_embedding_reads_images_flattened(backend):
    """Twenty 4x4 images used to reach the circuit padded to (n, 16, 4)."""
    _require(backend)
    X, y = _rows((20, 4, 4))
    pyqit.set_seed(0)
    model = VQCClassifier(n_qubits=4, n_layers=1, encoder=AmplitudeEmbedding)
    seen = _record_inputs(model)
    settings = {"split": (1.0, 0.0, 0.0), "batch_size": 20, "shuffle": False}

    Trainer(max_epochs=1, verbose=0).fit(model, DataModule(X, y, **settings))

    pixels = DataModule(X, y, **settings).setup().X_train.reshape(20, -1)
    expected = pixels / np.linalg.norm(pixels, axis=1, keepdims=True)
    np.testing.assert_allclose(seen[0], expected)


def test_width_check_counts_every_axis():
    """A (2, 2) sample holds 4 features; reading shape[1] would let it through."""
    _require("pennylane")
    X, y = _rows((12, 2, 2))

    with pytest.raises(ValueError, match="X has 4 features"):
        Trainer(max_epochs=1, verbose=0).fit(_model(), DataModule(X, y))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("y_axes", [(2,), (2, 3)], ids=["targets", "quantiles"])
def test_loaders_keep_target_axes_aligned_with_rows(backend, y_axes):
    _require(backend)
    ids = np.arange(24.0)
    X = np.stack([ids, ids], axis=1)
    y = ids.reshape(-1, *[1] * len(y_axes)) * np.ones((24, *y_axes))

    dm = DataModule(X, y, batch_size=5).setup()

    served = []
    for loader in (dm.train_loader(), dm.val_loader(), dm.test_loader()):
        for X_batch, y_batch in loader:
            X_batch, y_batch = _to_numpy(X_batch), _to_numpy(y_batch)
            assert y_batch.shape[1:] == y_axes
            np.testing.assert_array_equal(
                y_batch.reshape(len(y_batch), -1).max(axis=1), X_batch[:, 0]
            )
            served.extend(X_batch[:, 0])
    assert sorted(served) == ids.tolist()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("route", ["for_prediction", "load"])
def test_new_images_match_the_test_split(backend, route, tmp_path):
    """Saved statistics and flattening treat new rows as the test split was treated."""
    _require(backend)
    X, y = _rows((30, 4, 4))
    fitted = DataModule(X, y, normalize="minmax").setup(
        n_qubits=4, encoder=AmplitudeEmbedding
    )
    new_rows = DataModule(X, y).setup().X_test

    if route == "for_prediction":
        new = fitted.for_prediction(new_rows)
    else:
        new = DataModule.load(fitted.save(str(tmp_path / "dm.pkl")), new_rows)
    new.setup(stage="predict")

    assert new.X_test.shape == (len(new_rows), 16)
    np.testing.assert_allclose(new.X_test, fitted.X_test)


def test_new_rows_with_other_axes_raise():
    """Four flat columns would otherwise pad to the 8 a (2, 2, 2) sample holds."""
    _require("pennylane")
    X, y = _rows((12, 2, 2, 2))
    fitted = DataModule(X, y).setup(n_qubits=3, encoder=AmplitudeEmbedding)

    with pytest.raises(ValueError, match="4 axes"):
        fitted.for_prediction(np.zeros((5, 4)))


@pytest.mark.parametrize("backend", BACKENDS)
def test_cross_entropy_refuses_a_column_of_labels(backend):
    """On pennylane (n, 1) broadcast against (n,) and scored n * n pairs."""
    _require(backend)
    X, y = _data()

    with pytest.raises(ValueError, match="one label per sample"):
        Trainer(max_epochs=1, verbose=0, loss_fn="cross_entropy").fit(
            _model(), DataModule(X, y.reshape(-1, 1))
        )
