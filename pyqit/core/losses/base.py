from abc import abstractmethod

from pyqit.base.base_object import _PyQitObject
from pyqit.core.config import get_backend


class BaseLoss(_PyQitObject):
    """Base class for backend-dispatching losses.

    Subclasses declare what they support through tags rather than through
    control flow, so a loss missing on one backend fails with a clear message
    at construction instead of surfacing as an unhandled branch.

    Tags
    ----
    name : str
        Key used by ``get_loss_fn`` and by ``Trainer(loss_fn=...)``.
    backends : tuple of str
        Backends this loss implements. Any backend listed here must have a
        matching ``_pennylane`` / ``_torch`` method.
    target_dtype : {"float", "int"}
        What the loss expects its targets to be. ``"int"`` marks losses taking
        class indices, which is how the Lightning adapter knows to cast.
    """

    _tags = {
        "object_type": "loss",
        "name": None,
        "backends": ("pennylane", "torch"),
        "target_dtype": "float",
    }

    def __init__(self, backend=None):
        self.backend = backend
        super().__init__()

        self._backend = backend or get_backend()
        supported = self.get_tag("backends")
        if self._backend not in supported:
            raise ValueError(
                f"Loss {self.get_tag('name')!r} is not implemented for the "
                f"{self._backend!r} backend. Supported: {list(supported)}."
            )

    def __call__(self, preds, targets):
        if self._backend == "torch":
            return self._torch(preds, targets)
        return self._pennylane(preds, targets)

    @abstractmethod
    def _pennylane(self, preds, targets):
        """Evaluate the loss using pennylane.numpy."""

    def _torch(self, preds, targets):
        raise NotImplementedError(
            f"{type(self).__name__} declares torch support via its 'backends' "
            "tag but does not implement _torch."
        )

    def __repr__(self):
        return f"{type(self).__name__}(backend={self._backend!r})"

    @classmethod
    def get_test_params(cls):
        return [{"backend": "pennylane"}]
