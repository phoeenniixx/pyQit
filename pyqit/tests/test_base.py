from skbase.base import BaseMetaObject

from pyqit.base import OBJECT_TAGS, all_objects
from pyqit.core import QuantumPipeline
from pyqit.models import VQCClassifier


def test_every_tag_is_registered():
    """A tag a user finds on any object has an entry saying what it means."""
    carried = {
        (name, tag) for name, cls in all_objects() for tag in cls.get_class_tags()
    }
    pipeline = QuantumPipeline([VQCClassifier(n_qubits=2, n_layers=1)])
    skbase_own = set(BaseMetaObject.get_class_tags())
    carried |= {
        ("QuantumPipeline", tag) for tag in pipeline.get_tags() if tag not in skbase_own
    }

    unregistered = {(name, tag) for name, tag in carried if tag not in OBJECT_TAGS}
    assert unregistered == set()

    for tag_name, tag in OBJECT_TAGS.items():
        assert tag.get_class_tag("short_descr"), tag_name
        assert tag.get_class_tag("parent_type"), tag_name
