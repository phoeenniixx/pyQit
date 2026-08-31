"""Backend -> training loop lookup.

Built from the ``backend`` tag by ``all_objects``, the same way losses are
registered: a loop is available because it exists in an importable module with
the right tag, not because anything hand-registers it.

Caveat inherited from skbase's walk: a loop defined in a module whose name
starts with ``_`` is silently never discovered. Keep loop modules public.
"""

from pyqit.base.base_object import all_objects

_REGISTRY_CACHE = None


def loop_registry() -> dict:
    """Map every discoverable backend name to its training-loop class."""
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is None:
        _REGISTRY_CACHE = {
            cls.get_class_tag("backend"): cls
            for cls in all_objects(object_types="training_loop", return_names=False)
            if cls.get_class_tag("backend")
        }
    return _REGISTRY_CACHE


def get_training_loop(backend: str, trainer, reporter):
    """Instantiate the loop serving ``backend``.

    Constructing it validates the Trainer's settings against the loop's
    ``rejects`` / ``warns`` tags, so an unsupported setting is caught before any
    data is touched rather than partway through a fit.
    """
    registry = loop_registry()
    if backend not in registry:
        raise ValueError(
            f"No training loop registered for backend {backend!r}. "
            f"Available: {sorted(registry)}."
        )
    return registry[backend](trainer=trainer, reporter=reporter)
