import numpy as np
import pytest

import pyqit
from pyqit import DataModule, Trainer
from pyqit.datasets import (
    load_iris,
    make_ad_hoc,
    make_bars_and_stripes,
    make_circles,
    make_hidden_manifold,
    make_hyperplanes_parity,
    make_linearly_separable,
    make_moons,
    make_sine,
    make_two_curves,
)
from pyqit.models import VQCClassifier, VQCRegressor

BOWLES = [
    make_linearly_separable,
    make_two_curves,
    make_hidden_manifold,
    make_hyperplanes_parity,
]
SIZED = BOWLES + [make_ad_hoc]


@pytest.mark.parametrize("make", SIZED + [make_moons, make_circles])
def test_a_generated_binary_set_trains_through_the_trainer(make):
    """The one-import workflow: generate, wrap in a DataModule, fit."""
    kwargs = {"n_features": 2} if make in SIZED else {}
    X, y = make(n_samples=16, seed=0, **kwargs)
    pyqit.set_seed(0)
    model = VQCClassifier(n_qubits=2, n_layers=1, n_classes=2)
    dm = DataModule(X, y, normalize="minmax", batch_size=8)
    history = Trainer(max_epochs=1, verbose=0).fit(model, dm)
    assert np.isfinite(history.train_loss[-1])


def test_iris_trains_a_three_class_model():
    X, y = load_iris()
    pyqit.set_seed(0)
    model = VQCClassifier(n_qubits=4, n_layers=1, n_classes=3)
    dm = DataModule(X, y, normalize="minmax", batch_size=32)
    trainer = Trainer(max_epochs=1, loss_fn="cross_entropy", verbose=0)
    history = trainer.fit(model, dm)
    assert np.isfinite(history.train_loss[-1])


def test_sine_trains_the_regressor():
    X, y = make_sine(n_samples=16)
    pyqit.set_seed(0)
    dm = DataModule(X, y, normalize="minmax", batch_size=8)
    history = Trainer(max_epochs=1, verbose=0).fit(VQCRegressor(n_qubits=1), dm)
    assert np.isfinite(history.train_loss[-1])


def test_bars_and_stripes_images_train_through_the_trainer():
    X, y = make_bars_and_stripes(n_samples=16, height=2, width=2, seed=0)
    pyqit.set_seed(0)
    model = VQCClassifier(n_qubits=4, n_layers=1, n_classes=2)
    dm = DataModule(X, y, normalize="minmax", batch_size=8)
    history = Trainer(max_epochs=1, verbose=0).fit(model, dm)
    assert np.isfinite(history.train_loss[-1])


def test_noiseless_bars_and_stripes_are_constant_along_the_labelled_axis():
    """A stripe image repeats one column across its width, a bar one row."""
    X, y = make_bars_and_stripes(
        n_samples=60, height=3, width=5, noise=0, flatten=False, seed=0
    )
    assert X.shape == (60, 1, 3, 5)
    images = X[:, 0]
    rows_constant = (images == images[:, :, :1]).all(axis=(1, 2))
    cols_constant = (images == images[:, :1, :]).all(axis=(1, 2))
    assert rows_constant[y == 1].all() and cols_constant[y == 0].all()
    assert 0 < y.sum() < 60
    flat, _ = make_bars_and_stripes(n_samples=60, height=3, width=5, noise=0, seed=0)
    assert np.array_equal(flat, images.reshape(60, -1))


def test_ad_hoc_inputs_are_angles_with_both_classes_present():
    X, y = make_ad_hoc(n_samples=40, n_features=2, gap=0.3, seed=5)
    assert X.shape == (40, 2) and (X >= 0).all() and (X < 2 * np.pi).all()
    assert set(y.tolist()) == {0, 1}


@pytest.mark.parametrize("make", SIZED)
def test_set_seed_and_the_seed_argument_both_reproduce_a_set(make):
    """``pyqit.set_seed`` covers a bare call, and ``seed=`` ignores global state."""
    pyqit.set_seed(3)
    X1, y1 = make(n_samples=20, n_features=3)
    pyqit.set_seed(3)
    X2, y2 = make(n_samples=20, n_features=3)
    assert np.array_equal(X1, X2) and np.array_equal(y1, y2)

    pyqit.set_seed(4)
    X3, _ = make(n_samples=20, n_features=3, seed=7)
    pyqit.set_seed(5)
    X4, _ = make(n_samples=20, n_features=3, seed=7)
    assert np.array_equal(X3, X4)
    assert not np.array_equal(X1, X3)


def test_linearly_separable_rows_clear_the_margin_and_follow_the_hyperplane():
    """The paper's construction: ``sum(x) = 0`` separates, with an empty band."""
    X, y = make_linearly_separable(n_samples=200, n_features=5, margin=0.3, seed=1)
    score = X.sum(axis=1)
    assert X.shape == (200, 5)
    assert (np.abs(score) > 0.3).all()
    assert np.array_equal(y, (score > 0).astype(int))


@pytest.mark.parametrize(
    "make", [make_two_curves, make_hidden_manifold, make_hyperplanes_parity]
)
def test_the_balanced_generators_give_equal_classes_and_integer_labels(make):
    X, y = make(n_samples=50, n_features=3, seed=2)
    assert X.shape == (50, 3)
    assert y.dtype.kind == "i"
    assert np.bincount(y).tolist() == [25, 25]
