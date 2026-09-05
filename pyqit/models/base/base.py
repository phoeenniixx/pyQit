from abc import abstractmethod

from pyqit.base import _PyQitObject


class BaseModel(_PyQitObject):
    """Base class for all trainable models in PyQit."""

    _tags = {
        "object_type": "model",  # "ansatz" | "hybrid" | "kernel" | "classical"
        "is_quantum": True,
        "n_qubits": None,
        "differentiable": True,
        "requires_fit": True,
    }

    @abstractmethod
    def forward(self, X):
        """Run the model on a batch and return its raw output."""

    def __call__(self, X):
        """Alias for `forward`."""
        return self.forward(X)

    def is_fitted(self) -> bool:
        """Whether `Trainer.fit` has trained this model."""
        return getattr(self, "_is_fitted", False)

    def _mark_fitted(self):
        self._is_fitted = True
