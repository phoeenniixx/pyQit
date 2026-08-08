from contextvars import ContextVar
import logging

import numpy as np
from skbase.utils.dependencies import _check_soft_dependencies

logger = logging.getLogger("pyqit")

_BACKEND: ContextVar[str] = ContextVar("backend", default="pennylane")
_EXPLICITLY_SET: ContextVar[bool] = ContextVar("explicitly_set", default=False)
_WARNED_UNSET: ContextVar[bool] = ContextVar("warned_unset", default=False)


def set_backend(backend: str):
    """Set the quantum computing backend for the current context."""
    backend = backend.lower()
    if backend not in ["pennylane", "torch"]:
        raise ValueError(f"Unsupported backend '{backend}'.")

    _BACKEND.set(backend)
    _EXPLICITLY_SET.set(True)

    logger.info(f"Backend safely set to '{backend}' for current context.")


def set_seed(seed: int) -> int:
    """Set the seed.

    Seeds numpy and torch when it is installed.

    Parameters
    ----------
    seed : int
        Value applied to every supported RNG.

    Returns
    -------
    int
        The seed that was applied.

    Notes
    -----
    This mutates *global* RNG state, the same contract as Lightning's
    ``seed_everything``. ``Trainer.fit`` calls it for you; call it yourself
    before constructing a model if you also want its initial weights to be
    reproducible, since those are drawn at construction time.
    """
    np.random.seed(seed)
    seeded = ["numpy"]

    if _check_soft_dependencies("torch", severity="none"):
        import torch

        torch.manual_seed(seed)
        seeded.append("torch")

    logger.info(f"Seeded {' and '.join(seeded)} with {seed}.")
    return seed


def get_backend() -> str:
    """Get the current quantum computing backend for the context."""
    if not _EXPLICITLY_SET.get() and not _WARNED_UNSET.get():
        logger.warning(
            "No backend explicitly set for this context. Defaulting to 'pennylane'."
        )
        _WARNED_UNSET.set(True)

    return _BACKEND.get()
