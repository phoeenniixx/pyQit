import numpy as np
from skbase.lookup import all_objects

from pyqit.base.base_object import _PyQitObject


class BaseScenario(_PyQitObject):
    """A scenario the test suite runs models on.

    Subclass it in any public module of ``pyqit.tests.scenarios`` and implement
    ``_load``. ``all_scenarios`` picks it up through its tags:

    - ``estimator_type``: ``"classifier"`` or ``"regressor"``.
    - ``n_classes``: class counts it can produce, ``None`` for any.
    - ``n_features``: widths it can produce, ``None`` for any.
    """

    _tags = {
        "object_type": "scenario",
        "estimator_type": "classifier",
        "n_classes": (2,),
        "n_features": None,
    }

    def __init__(self, n_samples=16, n_features=4, n_classes=2, seed=0):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_classes
        self.seed = seed
        super().__init__()

    def _load(self):
        raise NotImplementedError

    def load(self) -> dict:
        X, y = self._load()
        return {
            "name": type(self).__name__,
            "X": np.asarray(X, dtype=np.float64),
            "y": np.asarray(y, dtype=np.float64),
            "n_classes": self.n_classes,
            "n_features": self.n_features,
        }

    @classmethod
    def fits(cls, n_features, n_classes) -> bool:
        regression = n_classes is None
        if (cls.get_class_tag("estimator_type") == "regressor") != regression:
            return False
        widths = cls.get_class_tag("n_features")
        counts = cls.get_class_tag("n_classes")
        return (widths is None or n_features in widths) and (
            regression or counts is None or n_classes in counts
        )


def all_scenarios(n_features, n_classes) -> list[type]:
    """Every scenario class that can produce this width and class count."""
    classes = all_objects(
        object_types=BaseScenario,
        return_names=False,
        package_name="pyqit.tests.scenarios",
        modules_to_ignore=None,
        filter_tags={"object_type": "scenario"},
    )
    return [cls for cls in classes if cls.fits(n_features, n_classes)]
