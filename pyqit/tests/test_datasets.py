import numpy as np
import pytest

import pyqit
from pyqit import DataModule, Trainer
from pyqit.datasets import (
    load_iris,
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


@pytest.mark.parametrize("make", BOWLES + [make_moons, make_circles])
def test_a_generated_binary_set_trains_through_the_trainer(make):
    """The one-import workflow: generate, wrap in a DataModule, fit."""
    kwargs = {"n_features": 2} if make in BOWLES else {}
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


@pytest.mark.parametrize("make", BOWLES)
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
