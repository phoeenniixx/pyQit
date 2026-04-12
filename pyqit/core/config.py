from contextvars import ContextVar
import logging

logger = logging.getLogger("pyqit")

_BACKEND: ContextVar[str] = ContextVar("backend", default="pennylane")
_EXPLICITLY_SET: ContextVar[bool] = ContextVar("explicitly_set", default=False)


def set_backend(backend: str):
    backend = backend.lower()
    if backend not in ["pennylane", "torch"]:
        raise ValueError(f"Unsupported backend '{backend}'.")

    _BACKEND.set(backend)
    _EXPLICITLY_SET.set(True)

    logger.info(f"Backend safely set to '{backend}' for current context.")


def get_backend() -> str:
    if not _EXPLICITLY_SET.get():
        logger.warning(
            "No backend explicitly set for this context. Defaulting to 'pennylane'."
        )
        _EXPLICITLY_SET.set(True)

    return _BACKEND.get()
