from pyqit.base import _PyQitObject


class RegressorMixin(_PyQitObject):
    """Marks a model as a regressor: predictions are its raw output.

    Mixed into a model alongside a ``BaseModel``, which supplies ``forward``.
    The training loops skip accuracy for regressors and record it as NaN.
    """

    _tags = {"estimator_type": "regressor"}

    def predict_step(self, X):
        """Return ``forward(X)`` flattened to one value per row."""
        return self.forward(X).flatten()
