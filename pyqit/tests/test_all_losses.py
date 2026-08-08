import numpy as np
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from pyqit.tests._fixture_generators import BaseFixtureGenerator


def _sample_batch(loss_cls, n=6, n_classes=3, seed=0):
    """Inputs shaped for whatever targets the loss declares."""
    rng = np.random.default_rng(seed)
    if loss_cls.get_class_tag("target_dtype") == "int":
        preds = rng.uniform(0.05, 1.0, size=(n, n_classes))
        preds = preds / preds.sum(axis=1, keepdims=True)
        targets = rng.integers(0, n_classes, size=n).astype(np.float64)
    else:
        preds = rng.uniform(0.0, 1.0, size=n)
        targets = rng.integers(0, 2, size=n).astype(np.float64)
    return preds, targets


class TestAllLosses(BaseFixtureGenerator):
    """Package-level tests for all ``BaseLoss`` subclasses."""

    object_type_filter = "loss"

    def test_pennylane_returns_finite_scalar(self, object_instance):
        cls = type(object_instance)
        preds, targets = _sample_batch(cls)
        value = float(cls(backend="pennylane")(preds, targets))
        assert np.isfinite(value), f"{cls.__name__} returned {value}"

    def test_backends_agree_numerically(self, object_instance):
        """The same loss must mean the same thing on both backends.

        Catches a torch implementation that runs cleanly but computes a
        different quantity than its pennylane twin.
        """
        cls = type(object_instance)
        if "torch" not in cls.get_class_tag("backends"):
            pytest.skip(f"{cls.__name__} does not declare torch support")
        if not _check_soft_dependencies("torch", severity="none"):
            pytest.skip("PyTorch is not installed.")

        import torch

        preds, targets = _sample_batch(cls)
        pl_value = float(cls(backend="pennylane")(preds, targets))
        torch_value = float(
            cls(backend="torch")(
                torch.tensor(preds, dtype=torch.float64),
                torch.tensor(targets, dtype=torch.float64),
            )
        )
        assert np.isclose(pl_value, torch_value, rtol=1e-6), (
            f"{cls.__name__}: pennylane={pl_value!r} but torch={torch_value!r}. "
            "The two backends compute different quantities."
        )
