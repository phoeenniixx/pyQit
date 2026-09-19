import pennylane as qml

from pyqit.ansatzes.sel import SELAnsatz
from pyqit.core.embeddings import AngleEmbedding
from pyqit.core.measurements import measure_probs
from pyqit.models.base.base import BaseModel
from pyqit.models.classification.classifier_mixin import ClassifierMixin
from pyqit.models.layers.dense import ACTIVATIONS, to_probabilities
from pyqit.models.layers.vqc import BaseVQC, z_from_probs


class DenseLayer(BaseModel):
    """Classical dense stage: ``activation(X @ weight.T + bias)``.

    Emits features, not predictions, so it is a `QuantumPipeline` stage rather
    than something to fit alone. Stack several for a deeper network. Weights
    use ``torch.nn.Linear``'s default init on both backends, under
    ``dense.weight`` and ``dense.bias``.

    Parameters
    ----------
    n_features : int
    n_out : int
    activation : {None, "tanh", "relu", "sigmoid"}, default None

    Examples
    --------
    >>> from pyqit.models.layers import DenseLayer
    >>> layer = DenseLayer(n_features=8, n_out=4, activation="tanh")
    """

    _tags = {"object_type": "layer", "is_quantum": False, "model_type": "classical"}

    def __init__(self, n_features, n_out, activation=None):
        if activation not in ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {list(ACTIVATIONS)}, got {activation!r}."
            )
        super().__init__()
        self.n_features = n_features
        self.n_out = n_out
        self.activation = activation
        self.register_dense("dense", n_features, n_out)

    def forward(self, X, **custom_weights):
        """Return ``(n_samples, n_out)`` features."""
        out = self.execute_qnode("dense", X, **custom_weights)
        return ACTIVATIONS[self.activation](out)

    @classmethod
    def get_test_params(cls):
        """List constructor kwargs used to parametrize this class in the test suite."""
        return [
            {"n_features": 3, "n_out": 2},
            {"n_features": 2, "n_out": 3, "activation": "tanh"},
        ]


class DenseClassifier(BaseModel, ClassifierMixin):
    """Classical head stage: one dense layer, then sigmoid or softmax.

    The last stage of a hybrid `QuantumPipeline`, turning the features of the
    stage before it into class probabilities and hard labels. Like every
    pyqit classifier it emits probabilities, not logits. Weights sit under
    ``dense.weight`` and ``dense.bias``.

    Parameters
    ----------
    n_features : int
    n_classes : int, default 2

    Examples
    --------
    >>> from pyqit.models.layers import DenseClassifier
    >>> head = DenseClassifier(n_features=4, n_classes=3)
    """

    _tags = {"object_type": "layer", "is_quantum": False, "model_type": "classical"}

    def __init__(self, n_features, n_classes=2):
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.register_dense("dense", n_features, 1 if n_classes == 2 else n_classes)

    def forward(self, X, **custom_weights):
        """Return class probabilities.

        Probability of class 1 for binary; a ``(n_samples, n_classes)``
        matrix otherwise.
        """
        logits = self.execute_qnode("dense", X, **custom_weights)
        return to_probabilities(logits, self.n_classes)

    @classmethod
    def get_test_params(cls):
        """List constructor kwargs used to parametrize this class in the test suite."""
        return [{"n_features": 3}, {"n_features": 4, "n_classes": 3}]


class QuantumLayer(BaseVQC):
    """Quantum stage: an embedding, an ansatz, and ``<Z>`` read on every wire.

    Emits ``(n_samples, n_qubits)`` features in ``[-1, 1]``, so it is a
    `QuantumPipeline` stage rather than something to fit alone. The
    expectations are computed from the basis-state probabilities, which costs
    a ``2 ** n_qubits`` vector per sample.

    Parameters
    ----------
    n_qubits : int, default 4
    n_layers : int, default 3
    ansatz : type, default SELAnsatz
    encoder : type, default AngleEmbedding
        Drives the prescaling the pipeline applies to this stage's input.
    device : str, default "default.qubit"
    shots : int, optional

    Examples
    --------
    >>> from pyqit.models.layers import QuantumLayer
    >>> layer = QuantumLayer(n_qubits=4, n_layers=2)
    """

    _tags = {"object_type": "layer"}

    def __init__(
        self,
        n_qubits=4,
        n_layers=3,
        ansatz=SELAnsatz,
        encoder=AngleEmbedding,
        device="default.qubit",
        shots=None,
    ):
        super().__init__(
            n_qubits=n_qubits,
            n_layers=n_layers,
            ansatz=ansatz,
            encoder=encoder,
            device=device,
            shots=shots,
        )
        self._z_from_probs = z_from_probs(n_qubits)

    def _resolve_readout(self, n_qubits, measure_fn, measure_wires):
        self._measure_fn = measure_probs
        self._measure_wires = list(range(n_qubits))

    def forward(self, X, **custom_weights):
        """Return ``(n_samples, n_qubits)`` expectation values."""
        probs = self.execute_qnode("main_circuit", X, **custom_weights)
        return qml.math.dot(probs, qml.math.cast_like(self._z_from_probs, probs))

    @classmethod
    def get_test_params(cls):
        """List constructor kwargs used to parametrize this class in the test suite."""
        return [{"n_qubits": 2, "n_layers": 1}]
