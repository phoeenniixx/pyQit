"""Backend-neutral callback protocol.

A callback written against these three hooks runs unchanged on both backends:
the pennylane loop calls them directly, and the Lightning loop reaches them
through ``_PyQitCallbackShim``, which translates Lightning's hooks onto them.
That is the whole reason the protocol is pyqit's own rather than Lightning's --
a ``lightning.Callback`` cannot be honoured by a ``qml.AdamOptimizer`` loop, so
accepting one would mean silently ignoring it on half the package.
"""

from dataclasses import dataclass, field
from typing import Any

from pyqit.base.base_object import _PyQitObject


@dataclass
class LoopState:
    """Everything a callback may read, and the one flag it may write.

    Attributes
    ----------
    stop : bool
        Set by a callback to request that training end after this epoch. Both
        loops check it; the Lightning loop forwards it to ``trainer.should_stop``.
    """

    model: Any
    datamodule: Any
    history: Any
    reporter: Any
    max_epochs: int
    epoch: int = -1
    metrics: dict[str, float] = field(default_factory=dict)
    stop: bool = False


class BaseCallback(_PyQitObject):
    """Base class for pyqit callbacks.

    Subclasses override any subset of the three hooks. Every hook takes the
    single ``LoopState`` argument, so adding information later does not change
    any signature.
    """

    _tags = {
        "object_type": "callback",
    }

    def on_fit_start(self, state: LoopState) -> None:
        """Called once, after setup and before the first epoch."""

    def on_epoch_end(self, state: LoopState) -> None:
        """Called once per epoch, with ``state.metrics`` filled for that epoch."""

    def on_fit_end(self, state: LoopState) -> None:
        """Called once, after the last epoch, including after an early stop."""

    @classmethod
    def get_test_params(cls):
        return [{}]
