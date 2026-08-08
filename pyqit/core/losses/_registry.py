from pyqit.base.base_object import all_objects

_REGISTRY_CACHE = None


def loss_registry() -> dict:
    """Map every discoverable loss name to its class."""
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is None:
        _REGISTRY_CACHE = {
            cls.get_class_tag("name"): cls
            for cls in all_objects(object_types="loss", return_names=False)
            if cls.get_class_tag("name")
        }
    return _REGISTRY_CACHE


def get_loss_fn(name, backend: str = "pennylane"):
    """Resolve a loss name to a callable bound to ``backend``.

    Callables pass straight through, so a user-supplied loss is never
    reinterpreted; anything else is looked up by its ``name`` tag.
    """
    if callable(name):
        return name

    key = name.lower()
    registry = loss_registry()
    if key not in registry:
        raise ValueError(f"Unknown loss {key!r}. Available: {sorted(registry)}.")

    return registry[key](backend=backend)
