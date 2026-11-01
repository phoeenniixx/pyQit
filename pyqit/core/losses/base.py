from abc import abstractmethod

import pennylane as qml

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
    target_shape : {"same_as_preds", "one_per_sample"}
        What shape of targets the loss takes, checked on every call so that a
        mismatch raises rather than broadcasts into a wrong number.
    """

    _tags = {
        "object_type": "loss",
        "name": None,
        "backends": ("pennylane", "torch"),
        "target_dtype": "float",
        "target_shape": "same_as_preds",
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
        """Evaluate the loss, dispatching to the active backend.

        Parameters
        ----------
        preds : array-like
            Model output. Probabilities, not logits.
        targets : array-like

        Returns
        -------
        float or tensor
        """
        self._check_target_shape(preds, targets)
        if self._backend == "torch":
            return self._torch(preds, targets)
        return self._pennylane(preds, targets)

    def _check_target_shape(self, preds, targets):
        name = self.get_tag("name")
        p, t = tuple(qml.math.shape(preds)), tuple(qml.math.shape(targets))
        if self.get_tag("target_shape") == "one_per_sample":
            if len(t) != 1:
                raise ValueError(
                    f"Loss {name!r} needs one label per sample, a 1-D y, got shape "
                    f"{t}. Flatten a single column of labels with y.ravel()."
                )
        elif p != t:
            raise ValueError(
                f"Loss {name!r} needs predictions and targets of the same shape, "
                f"got {p} and {t}. A multi-column output scored against class "
                "indices needs a multi-target loss, such as loss_fn='cross_entropy'."
            )

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
        """List constructor kwargs used to parametrize this class in the test suite."""
        return [{"backend": "pennylane"}]
