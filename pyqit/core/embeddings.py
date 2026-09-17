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
        """Apply the embedding circuit to `inputs`, in place on the QNode."""

    def __call__(self, inputs):
        """Alias for `forward`."""
        return self.forward(inputs)


class AngleEmbedding(BaseEmbedding):
    """One rotation per qubit, PennyLane's `AngleEmbedding`.

    Takes one feature per wire. The `DataModule` zero-pads narrower input to
    `n_qubits` columns and multiplies by pi, which maps features normalized to
    `[0, 1]` onto `[0, pi]`. Input wider than `n_qubits` raises.

    Parameters
    ----------
    n_qubits : int
    rotation : {"X", "Y", "Z"}, default "X"
        Rotation gate the features drive.

    Examples
    --------
    >>> from pyqit.core import AngleEmbedding
    >>> from pyqit.models import VQCClassifier
    >>> model = VQCClassifier(n_qubits=4, encoder=AngleEmbedding)
    """

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
        """Apply one rotation gate per qubit. Expects `inputs` scaled to [0, pi]."""
        qml.AngleEmbedding(
            features=inputs, wires=range(self.n_qubits), rotation=self.rotation
        )

    @classmethod
    def get_test_params(cls):
        """List constructor kwargs used to parametrize this class in the test suite."""
        return [{"n_qubits": 2}, {"n_qubits": 4, "rotation": "Y"}]


class HadamardAngleEmbedding(BaseEmbedding):
    """The encoding of Mari et al. (2020): a Hadamard layer, then one RY per wire.

    The Hadamards start every wire at ``|+>``, so an angle in
    ``[-pi/2, pi/2]`` covers the arc from ``|0>`` to ``|1>``. Inputs are
    prescaled by ``pi / 2``, which maps a ``tanh`` layer's output onto that
    range, as in the paper.

    Parameters
    ----------
    n_qubits : int

    References
    ----------
    Mari, Bromley, Izaac, Schuld, Killoran, "Transfer learning in hybrid
    classical-quantum neural networks", Quantum 4, 340 (2020).
    """

    _tags = {
        "embedding_type": "angle",
        "differentiable": True,
        "prescale": "angle_half_pi",
        "n_qubits_min": 1,
    }

    def forward(self, inputs):
        """Apply H then RY on every wire. Expects `inputs` scaled by pi / 2."""
        for w in range(self.n_qubits):
            qml.Hadamard(wires=w)
        for w in range(self.n_qubits):
            qml.RY(inputs[..., w], wires=w)

    @classmethod
    def get_test_params(cls):
        """List constructor kwargs used to parametrize this class in the test suite."""
        return [{"n_qubits": 2}, {"n_qubits": 3}]


class AmplitudeEmbedding(BaseEmbedding):
    """Features as state amplitudes, PennyLane's `AmplitudeEmbedding`.

    `n_qubits` wires carry up to `2 ** n_qubits` features, so four qubits take
    sixteen. The `DataModule` zero-pads each row to that width and
    L2-normalizes it. Wider input raises.

    Parameters
    ----------
    n_qubits : int
    normalize : bool, default True
        Passed to PennyLane's template, which renormalizes the state vector.
    pad_with : float, default 0.0
        Passed to PennyLane's template, which pads a short feature vector with
        this value.

    Examples
    --------
    >>> from pyqit.core import AmplitudeEmbedding
    >>> from pyqit.models import VQCClassifier
    >>> model = VQCClassifier(n_qubits=4, encoder=AmplitudeEmbedding)
    """

    _tags = {
        "embedding_type": "amplitude",
        "differentiable": False,
        "prescale": "amplitude",
        "n_qubits_min": 1,
    }

    def __init__(self, n_qubits: int, normalize: bool = True, pad_with: float = 0.0):
        self.normalize = normalize
        self.pad_with = pad_with
        super().__init__(n_qubits=n_qubits)

    def forward(self, inputs):
        """Encode `inputs` into amplitudes. Expects `2 ** n_qubits` features."""
        qml.AmplitudeEmbedding(
            features=inputs,
            wires=range(self.n_qubits),
            normalize=self.normalize,
            pad_with=self.pad_with,
        )

    @classmethod
    def get_test_params(cls):
        """List constructor kwargs used to parametrize this class in the test suite."""
        return [{"n_qubits": 2}]


class IQPEmbedding(BaseEmbedding):
    """IQP feature map of Havlicek et al. (2019), PennyLane's `IQPEmbedding`.

    Takes one feature per wire, prescaled the way `AngleEmbedding` is. Needs at
    least two qubits.

    Parameters
    ----------
    n_qubits : int
    """

    _tags = {
        "embedding_type": "iqp",
        "differentiable": False,
        "prescale": "angle_pi",
        "n_qubits_min": 2,
    }

    def __init__(self, n_qubits: int):
        super().__init__(n_qubits=n_qubits)

    def forward(self, inputs):
        """Apply the IQP feature map. Expects `inputs` scaled to [0, pi]."""
        qml.IQPEmbedding(features=inputs, wires=range(self.n_qubits))

    @classmethod
    def get_test_params(cls):
        """List constructor kwargs used to parametrize this class in the test suite."""
        return [{"n_qubits": 2}]


class ZZFeatureMap(BaseEmbedding):
    """The second-order Pauli-Z feature map of Havlicek et al. (2019).

    The feature map Qiskit ML's `VQC` uses. It is a different circuit from
    `IQPEmbedding`. Takes one feature per wire, prescaled the way
    `AngleEmbedding` is, and needs at least two qubits.

    Parameters
    ----------
    n_qubits : int
    n_repeats : int, default 2
        Repetitions of the map, Qiskit's default.

    References
    ----------
    Havlicek et al., "Supervised learning with quantum-enhanced feature
    spaces", Nature 567, 209 (2019).
    """

    _tags = {
        "embedding_type": "zz",
        "differentiable": True,
        "prescale": "angle_pi",
        "n_qubits_min": 2,
    }

    def __init__(self, n_qubits: int, n_repeats: int = 2):
        self.n_repeats = n_repeats
        super().__init__(n_qubits=n_qubits)

    def forward(self, inputs):
        """Apply the feature map. Expects one feature per wire."""
        wires = range(self.n_qubits)
        for _ in range(self.n_repeats):
            for i in wires:
                qml.Hadamard(wires=i)
                qml.RZ(2.0 * inputs[..., i], wires=i)
            for i in wires:
                for j in range(i + 1, self.n_qubits):
                    phase = (
                        2.0
                        * (qml.numpy.pi - inputs[..., i])
                        * (qml.numpy.pi - inputs[..., j])
                    )
                    qml.MultiRZ(phase, wires=[i, j])

    @classmethod
    def get_test_params(cls):
        """List constructor kwargs used to parametrize this class in the test suite."""
        return [{"n_qubits": 2}, {"n_qubits": 3, "n_repeats": 1}]
