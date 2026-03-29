from abc import abstractmethod

import pennylane as qml

from pyqit.base.base_object import _PyQitObject


class BaseEmbedding(_PyQitObject):
    """
    Base class for PennyLane circuit embedding wrappers.
    """

    _tags = {
        "object_type": "embedding",
        "embedding_type": None,  # "angle"|"amplitude"|"iqp"|"qaoa"|"basis"
        "differentiable": None,
        "prescale": None,
        "n_qubits_min": 1,
    }

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        prescale = None
        for base in cls.__mro__:
            tags = base.__dict__.get("_tags", {})
            if "prescale" in tags:
                prescale = tags["prescale"]
                break
        cls.PRESCALE = prescale

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        super().__init__()

    @abstractmethod
    def forward(self, inputs):
        pass

    def __call__(self, inputs):
        return self.forward(inputs)

    @classmethod
    def prescale_key(cls) -> str | None:
        return cls.get_class_tags().get("prescale")


class AngleEmbedding(BaseEmbedding):
    _tags = {
        "embedding_type": "angle",
        "differentiable": True,
        "prescale": "angle_pi",
        "n_qubits_min": 1,
    }

    def __init__(self, n_qubits: int, rotation: str = "X"):
        self.rotation = rotation
        super().__init__(n_qubits=n_qubits)

    def forward(self, inputs):
        qml.AngleEmbedding(
            features=inputs, wires=range(self.n_qubits), rotation=self.rotation
        )

    @classmethod
    def get_test_params(cls):
        return [{"n_qubits": 2}, {"n_qubits": 4, "rotation": "Y"}]


class AmplitudeEmbedding(BaseEmbedding):
    _tags = {
        "embedding_type": "amplitude",
        "differentiable": False,
        "prescale": "amplitude",
        "n_qubits_min": 1,
    }

    def forward(self, inputs):
        qml.AmplitudeEmbedding(
            features=inputs, wires=range(self.n_qubits), normalize=False
        )

    @classmethod
    def get_test_params(cls):
        return {"n_qubits": 2}


class IQPEmbedding(BaseEmbedding):
    _tags = {
        "embedding_type": "iqp",
        "differentiable": False,
        "prescale": "angle_pi",
        "n_qubits_min": 2,
    }

    def forward(self, inputs):
        qml.IQPEmbedding(features=inputs, wires=range(self.n_qubits))

    @classmethod
    def get_test_params(cls):
        return {"n_qubits": 2}
