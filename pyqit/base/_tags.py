"""Register of every tag a PyQit object can carry.

Each tag is a class below, named after the tag, whose docstring says what the
tag means, which objects carry it, what reads it and what its values are.
``OBJECT_TAGS`` maps tag name to class. A tag on any object that is not in
this file fails ``test_every_tag_is_registered``.

NOTE
----
Inspired from sktime's tag system:
    https://github.com/sktime/sktime/blob/main/sktime/registry/_tags.py
"""

from skbase.base import BaseObject


class _BaseTag(BaseObject):
    """Base class for tag definitions.

    ``parent_type`` is the ``object_type`` of the objects that carry the tag,
    or a tuple of them; ``"object"`` means every ``_PyQitObject``.
    ``user_facing`` says whether the tag is part of the public metadata a user
    filters or reads, as opposed to plumbing the framework reads.

    NOTE
    ----
    Inspired from sktime's tag system:
        https://github.com/sktime/sktime/blob/main/sktime/registry/_tags.py
    """

    _tags = {
        "object_type": "tag",
        "tag_name": None,
        "parent_type": None,
        "tag_type": None,
        "short_descr": None,
        "user_facing": False,
    }


class authors(_BaseTag):
    """Authors of the object, GitHub IDs.

    - String name: ``"authors"``
    - Public metadata tag
    - Values: string or list of strings
    - Example: ``["phoeenniixx", "fkiraly"]``
    - Default: ``"phoeenniixx"``

    Credits the original implementation and later contributions. For a wrapper
    around a third-party circuit, include the wrapped code's authors.
    """

    _tags = {
        "tag_name": "authors",
        "parent_type": "object",
        "tag_type": ("list", "str"),
        "short_descr": "list of authors of the object, each author a GitHub handle",
        "user_facing": True,
    }


class object_type(_BaseTag):
    """The kind of object, which decides its test suite and hub page.

    - String name: ``"object_type"``
    - Public metadata tag
    - Values: ``"model"``, ``"layer"``, ``"ansatz"``, ``"embedding"``,
      ``"loss"``, ``"callback"``, ``"trainer"``, ``"training_loop"``,
      ``"pipeline"``, ``"datamodule"``
    - Example: ``"model"``
    - Default: none, every base class sets it

    ``all_objects(object_types=...)`` filters on it, and each
    ``object_overview()`` lists every class with one, ``DataModule`` and
    ``QuantumPipeline`` included though skbase does not discover them.
    """

    _tags = {
        "tag_name": "object_type",
        "parent_type": "object",
        "tag_type": "str",
        "short_descr": "kind of object: model, layer, ansatz, embedding, ...",
        "user_facing": True,
    }


class python_dependencies(_BaseTag):
    """Soft dependencies the object needs beyond PennyLane and NumPy.

    - String name: ``"python_dependencies"``
    - Public metadata tag
    - Values: package name, list of package names, or None
    - Example: ``"pennylane-qiskit"``
    - Default: ``None``

    The test fixture generator skips a class whose dependencies are not
    installed, so the no-soft-deps CI job passes without it.
    """

    _tags = {
        "tag_name": "python_dependencies",
        "parent_type": "object",
        "tag_type": ("str", "list", None),
        "short_descr": "soft dependencies the object needs, pip names",
        "user_facing": True,
    }


class tests_skip_tests(_BaseTag):
    """Test names the class opts out of.

    - String name: ``"tests:skip_tests"``
    - Private test tag
    - Values: list of test function names
    - Default: ``[]``

    Read by the fixture generator only.
    """

    _tags = {
        "tag_name": "tests:skip_tests",
        "parent_type": "object",
        "tag_type": "list",
        "short_descr": "test names the class skips",
        "user_facing": False,
    }


class tests_skip_params(_BaseTag):
    """Parameter-set instance names (``ClassName-<i>``) the tests skip.

    - String name: ``"tests:skip_params"``
    - Private test tag
    - Values: list of strings
    - Default: ``[]``

    Read by the fixture generator only.
    """

    _tags = {
        "tag_name": "tests:skip_params",
        "parent_type": "object",
        "tag_type": "list",
        "short_descr": "parameter-set instance names the tests skip",
        "user_facing": False,
    }


class model_type(_BaseTag):
    """What kind of network a model or layer is.

    - String name: ``"model_type"``
    - Public metadata tag
    - Values: ``"quantum"``, ``"hybrid"``, ``"classical"``
    - Example: ``"hybrid"``
    - Default: ``"quantum"`` on ``BaseModel``

    ``Trainer`` prints it in the model summary with ``estimator_type``, and
    ``all_objects("model", filter_tags={"model_type": ...})`` selects on it.
    Behaviour keys on ``is_quantum`` instead, since a hybrid has a circuit too.
    """

    _tags = {
        "tag_name": "model_type",
        "parent_type": ("model", "layer"),
        "tag_type": "str",
        "short_descr": "quantum, hybrid or classical",
        "user_facing": True,
    }


class estimator_type(_BaseTag):
    """Whether a model predicts classes or values.

    - String name: ``"estimator_type"``
    - Public metadata tag
    - Values: ``"classifier"``, ``"regressor"``
    - Default: none; set by ``ClassifierMixin`` and ``RegressorMixin``

    The loops threshold a classifier's output into labels and skip accuracy
    for a regressor. A ``QuantumPipeline`` takes its final stage's value, or
    the common value of an ensemble, and rejects a mixed ensemble.
    """

    _tags = {
        "tag_name": "estimator_type",
        "parent_type": ("model", "layer", "pipeline"),
        "tag_type": "str",
        "short_descr": "classifier or regressor",
        "user_facing": True,
    }


class is_quantum(_BaseTag):
    """Whether the object runs a circuit.

    - String name: ``"is_quantum"``
    - Public metadata tag
    - Values: bool
    - Default: ``True`` on ``BaseModel``; ``False`` on the dense layers

    ``QuantumPipeline`` reads it off its stages to set ``has_quantum``.
    """

    _tags = {
        "tag_name": "is_quantum",
        "parent_type": ("model", "layer"),
        "tag_type": "bool",
        "short_descr": "whether the object runs a circuit",
        "user_facing": True,
    }


class bp_scale_factor(_BaseTag):
    """Multiplier on the barren-plateau variance floor.

    - String name: ``"bp_scale_factor"``
    - Public metadata tag
    - Values: float
    - Example: ``0.25`` on ``ClassifierMixin``
    - Default: ``1.0`` when absent

    ``check_barren_plateau`` multiplies the theoretical floor by it, so a
    readout whose gradients are legitimately smaller is not flagged.
    """

    _tags = {
        "tag_name": "bp_scale_factor",
        "parent_type": "model",
        "tag_type": "float",
        "short_descr": "multiplier on the barren-plateau variance floor",
        "user_facing": True,
    }


class differentiable(_BaseTag):
    """Whether gradients flow through the object's parameters or inputs.

    - String name: ``"differentiable"``
    - Public metadata tag
    - Values: bool or None
    - Example: ``False`` on ``AmplitudeEmbedding`` and ``IQPEmbedding``
    - Default: ``True`` on models, ``None`` on the ansatz and embedding bases

    """

    _tags = {
        "tag_name": "differentiable",
        "parent_type": ("model", "embedding", "ansatz"),
        "tag_type": ("bool", None),
        "short_descr": "whether gradients flow through the object",
        "user_facing": True,
    }


class n_qubits(_BaseTag):
    """Placeholder on ``BaseModel``; the width lives on the instance.

    - String name: ``"n_qubits"``
    - Private tag
    - Values: None
    - Default: ``None``

    """

    _tags = {
        "tag_name": "n_qubits",
        "parent_type": "model",
        "tag_type": None,
        "short_descr": "unused placeholder; width is the n_qubits attribute",
        "user_facing": False,
    }


class requires_fit(_BaseTag):
    """Whether the model must be trained before it predicts.

    - String name: ``"requires_fit"``
    - Public metadata tag
    - Values: bool
    - Default: ``True`` on ``BaseModel``
    """

    _tags = {
        "tag_name": "requires_fit",
        "parent_type": "model",
        "tag_type": "bool",
        "short_descr": "whether the model must be trained before predicting",
        "user_facing": True,
    }


class embedding_type(_BaseTag):
    """Family of the embedding circuit.

    - String name: ``"embedding_type"``
    - Public metadata tag
    - Values: ``"angle"``, ``"amplitude"``, ``"iqp"``, ``"zz"``
    - Default: ``None`` on ``BaseEmbedding``

    """

    _tags = {
        "tag_name": "embedding_type",
        "parent_type": "embedding",
        "tag_type": "str",
        "short_descr": "family of the embedding circuit",
        "user_facing": True,
    }


class prescale(_BaseTag):
    """How the DataModule shapes input for the embedding.

    - String name: ``"prescale"``
    - Public metadata tag
    - Values: ``None``, ``"angle_pi"``, ``"angle_half_pi"``, ``"amplitude"``,
      ``"binary"``
    - Example: ``"amplitude"`` pads to ``2**n_qubits`` and L2-normalizes
    - Default: ``None`` on ``BaseEmbedding``

    """

    _tags = {
        "tag_name": "prescale",
        "parent_type": "embedding",
        "tag_type": ("str", None),
        "short_descr": "input shaping the DataModule applies for the embedding",
        "user_facing": True,
    }


class n_qubits_min(_BaseTag):
    """Narrowest circuit the ansatz or embedding builds.

    - String name: ``"n_qubits_min"``
    - Public metadata tag
    - Values: int
    - Example: ``2`` on ``IQPEmbedding`` and ``SimplifiedTwoDesignAnsatz``
    - Default: ``1``

    """

    _tags = {
        "tag_name": "n_qubits_min",
        "parent_type": ("embedding", "ansatz"),
        "tag_type": "int",
        "short_descr": "narrowest circuit the object builds",
        "user_facing": True,
    }


class ansatz_type(_BaseTag):
    """Family of the ansatz circuit.

    - String name: ``"ansatz_type"``
    - Public metadata tag
    - Values: str or None
    - Default: ``None`` on ``BaseAnsatz``

    """

    _tags = {
        "tag_name": "ansatz_type",
        "parent_type": "ansatz",
        "tag_type": ("str", None),
        "short_descr": "family of the ansatz circuit",
        "user_facing": True,
    }


class name(_BaseTag):
    """The string a loss is asked for by.

    - String name: ``"name"``
    - Public metadata tag
    - Values: str
    - Example: ``"cross_entropy"`` for ``Trainer(loss_fn="cross_entropy")``
    - Default: ``None`` on ``BaseLoss``, which keeps it out of the registry
    """

    _tags = {
        "tag_name": "name",
        "parent_type": "loss",
        "tag_type": "str",
        "short_descr": "the string a loss is asked for by",
        "user_facing": True,
    }


class backends(_BaseTag):
    """Backends a loss implements.

    - String name: ``"backends"``
    - Public metadata tag
    - Values: tuple of ``"pennylane"`` and/or ``"torch"``
    - Default: ``("pennylane", "torch")``

    A listed backend must have the matching ``_pennylane`` or ``_torch``
    method; ``BaseLoss`` raises when asked for one that is not listed.
    """

    _tags = {
        "tag_name": "backends",
        "parent_type": "loss",
        "tag_type": "tuple",
        "short_descr": "backends the loss implements",
        "user_facing": True,
    }


class target_dtype(_BaseTag):
    """Whether a loss wants class indices or floats as targets.

    - String name: ``"target_dtype"``
    - Public metadata tag
    - Values: ``"float"``, ``"int"``
    - Example: ``"int"`` on ``CrossEntropyLoss``
    - Default: ``"float"``

    The Lightning data adapter casts targets to match.
    """

    _tags = {
        "tag_name": "target_dtype",
        "parent_type": "loss",
        "tag_type": "str",
        "short_descr": "target dtype the loss expects, float or int",
        "user_facing": True,
    }


class backend(_BaseTag):
    """Backend a training loop serves.

    - String name: ``"backend"``
    - Private tag
    - Values: ``"pennylane"``, ``"torch"``
    - Default: ``None`` on ``BaseTrainingLoop``

    ``loop_registry()`` keys every loop on it; ``Trainer.fit`` looks the
    active backend up there.
    """

    _tags = {
        "tag_name": "backend",
        "parent_type": "training_loop",
        "tag_type": "str",
        "short_descr": "backend the training loop serves",
        "user_facing": False,
    }


class mode(_BaseTag):
    """How a pipeline composes its stages.

    - String name: ``"mode"``
    - Public metadata tag
    - Values: ``"sequential"``, ``"ensemble"``
    - Default: ``"sequential"``

    Set from the constructor argument of the same name.
    """

    _tags = {
        "tag_name": "mode",
        "parent_type": "pipeline",
        "tag_type": "str",
        "short_descr": "sequential or ensemble",
        "user_facing": False,
    }


class n_stages(_BaseTag):
    """Number of stages in a pipeline.

    - String name: ``"n_stages"``
    - Public metadata tag
    - Values: int
    - Default: ``0`` before construction fills it
    """

    _tags = {
        "tag_name": "n_stages",
        "parent_type": "pipeline",
        "tag_type": "int",
        "short_descr": "number of stages in the pipeline",
        "user_facing": False,
    }


class has_quantum(_BaseTag):
    """Whether any stage of a pipeline is quantum.

    - String name: ``"has_quantum"``
    - Public metadata tag
    - Values: bool
    - Default: ``False`` before construction fills it

    Derived from the stages' ``is_quantum`` tags at construction.
    """

    _tags = {
        "tag_name": "has_quantum",
        "parent_type": "pipeline",
        "tag_type": "bool",
        "short_descr": "whether any pipeline stage is quantum",
        "user_facing": False,
    }


OBJECT_TAGS = {cls.get_class_tag("tag_name"): cls for cls in _BaseTag.__subclasses__()}


_PUBLIC_MODULES = (
    "pyqit.models.layers",
    "pyqit.core.callbacks",
    "pyqit.core.embeddings",
    "pyqit.ansatzes",
    "pyqit.models",
    "pyqit.core",
    "pyqit",
)


def _public_path(cls):
    """``module.Name`` under the shortest public module that exports ``cls``."""
    import importlib

    for module in _PUBLIC_MODULES:
        if getattr(importlib.import_module(module), cls.__name__, None) is cls:
            return f"{module}.{cls.__name__}"
    return f"{cls.__module__}.{cls.__name__}"


def object_overview():
    """Every public object with its user-facing tags, for finding one by what it is.

    Returns
    -------
    list of dict
        One per class, sorted by ``object_type`` then name, with ``name``,
        ``path`` (the import path its documentation page uses),
        ``object_type`` and ``tags``: the user-facing tags of
        ``OBJECT_TAGS`` the class sets to a value other than ``None``.
    """
    from pyqit.base.base_object import all_objects
    from pyqit.core.pipeline import QuantumPipeline
    from pyqit.data.datamodule import DataModule

    classes = [cls for _, cls in all_objects()] + [QuantumPipeline, DataModule]
    user_facing = [
        name
        for name, tag in OBJECT_TAGS.items()
        if tag.get_class_tag("user_facing") and name != "object_type"
    ]
    rows = []
    for cls in classes:
        tags = cls.get_class_tags() if hasattr(cls, "get_class_tags") else cls._tags
        if tags.get("object_type") is None:
            continue
        rows.append(
            {
                "name": cls.__name__,
                "path": _public_path(cls),
                "object_type": tags["object_type"],
                "tags": {k: tags[k] for k in user_facing if tags.get(k) is not None},
            }
        )
    return sorted(rows, key=lambda r: (r["object_type"], r["name"]))
