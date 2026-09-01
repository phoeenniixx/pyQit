"""Backend -> training loop lookup, keyed on each loop's ``backend`` tag."""

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
    data is touched.

    Parameters
    ----------
    backend : str
        Backend name, as returned by ``get_backend()``.
    trainer : Trainer
        Passed to the loop and validated against it.
    reporter : Reporter
        Console output for the run.

    Returns
    -------
    BaseTrainingLoop
        A loop instance bound to ``trainer``.

    Raises
    ------
    ValueError
        If no loop declares ``backend``.
    """
    registry = loop_registry()
    if backend not in registry:
        raise ValueError(
            f"No training loop registered for backend {backend!r}. "
            f"Available: {sorted(registry)}."
        )
    return registry[backend](trainer=trainer, reporter=reporter)
