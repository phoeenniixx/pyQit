from skbase.base import BaseObject
from skbase.lookup import all_objects as _skbase_all_objects


class _PyQitObject(BaseObject):
    _tags = {
        "python_deps": None,
        "tests:skip_tests": [],  # list of test names to skip for this class
        "tests:skip_params": [],  # list of param-set instance names to skip
    }


def all_objects(
    object_types=None,
    return_names: bool = True,
    exclude_objects=None,
    filter_tags=None,
    package_name: str = "pyqit",
):
    """Retrieve all concrete PyQit objects, optionally filtered.

    Parameters
    ----------
    object_types : str, list of str, or None
        One or more ``object_type`` tag values to filter by, e.g. ``"model"``,
        ``["embedding", "ansatz"]``.  ``None`` returns every object.
    return_names : bool, default True
        If True return ``(name, class)`` tuples; else return classes only.
    exclude_objects : list of str or None
        Class names to exclude.
    filter_tags : dict or None
        Additional ``{tag: value}`` pairs that every returned class must satisfy.
    package_name : str, default "pyqit"
        Root package scanned by skbase.

    Returns
    -------
    list of class or list of (str, class)
    """
    combined_filter: dict = {}

    if object_types is not None:
        if isinstance(object_types, str):
            object_types = [object_types]
        combined_filter["object_type"] = object_types

    if filter_tags:
        combined_filter.update(filter_tags)

    results = _skbase_all_objects(
        object_types=_PyQitObject,
        return_names=return_names,
        exclude_objects=exclude_objects or [],
        filter_tags=combined_filter or None,
        package_name=package_name,
    )

    return results
