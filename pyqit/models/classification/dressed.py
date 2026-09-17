import numpy as np

from pyqit.ansatzes.cnot_ladder import CNOTLadderAnsatz
from pyqit.core.embeddings import HadamardAngleEmbedding
from pyqit.core.pipeline import QuantumPipeline
from pyqit.models.base.quantum_model import BaseQuantumModel
from pyqit.models.classification.classifier_mixin import ClassifierMixin
from pyqit.models.layers.stages import DenseClassifier, DenseLayer, QuantumLayer
from pyqit.utils.utils import _restore_weights


class DressedQuantumClassifier(BaseQuantumModel, ClassifierMixin):
    """Dressed quantum circuit of Mari et al. (2020): dense, circuit, dense.

    A classical layer maps ``n_features`` to ``n_qubits`` angles through
    ``tanh(.) * pi / 2``; the circuit applies a Hadamard layer, encodes the
    angles with RY, then ``n_layers`` blocks of a CNOT ladder followed by an
    RY layer, and reads ``<Z>`` on every wire; a second classical layer maps
    those to the classes. There is no separate embedding on the model, so the
    DataModule does not prescale.

    Inside, the network is a `QuantumPipeline` of three layers from
    ``pyqit.models.layers``: a `DenseLayer`, a `QuantumLayer` with
    `HadamardAngleEmbedding` and `CNOTLadderAnsatz`, and a `DenseClassifier`.
    Their weights are this model's, under ``pre_net.*``, ``quantum.*`` and
    ``post_net.*``. Compose those layers yourself for a different hybrid.

    The head differs from the paper in one way: pyqit models emit
    probabilities, so binary applies a sigmoid to one logit and multi-class a
    softmax, rather than handing logits to cross-entropy.

    Parameters
    ----------
    n_features : int
    n_qubits : int, default 4
    n_layers : int, default 6
        Variational depth, ``q_depth`` in the paper.
    n_classes : int, default 2
    q_delta : float, default 0.01
        Spread of the normal initial quantum weights, as in the paper. The
        classical layers use ``torch.nn.Linear``'s default init on both
        backends.
    device : str, default "default.qubit"
    shots : int, optional

    References
    ----------
    Mari, Bromley, Izaac, Schuld, Killoran, "Transfer learning in hybrid
    classical-quantum neural networks", Quantum 4, 340 (2020). PennyLane's
    "Quantum transfer learning" demo is the reference implementation.

    Examples
    --------
    >>> import pyqit
    >>> from pyqit.models import DressedQuantumClassifier
    >>> model = DressedQuantumClassifier(n_features=8, n_qubits=4, n_layers=2)
    >>> history = pyqit.Trainer(max_epochs=5).fit(model, dm)  # doctest: +SKIP
    """

    _LAYER_ENTRY = {
        "pre_net": "dense",
        "quantum": "main_circuit",
        "post_net": "dense",
    }

    def __init__(
        self,
        n_features,
        n_qubits=4,
        n_layers=6,
        n_classes=2,
        q_delta=0.01,
        device="default.qubit",
        shots=None,
    ):
        super().__init__(device=device, shots=shots)
        self.n_features = n_features
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.q_delta = q_delta

        pre_net = DenseLayer(n_features, n_qubits, activation="tanh")
        q_init = q_delta * np.random.randn(n_layers, n_qubits)
        post_net = DenseClassifier(n_qubits, n_classes=n_classes)
        quantum = QuantumLayer(
            n_qubits=n_qubits,
            n_layers=n_layers,
            ansatz=CNOTLadderAnsatz,
            encoder=HadamardAngleEmbedding,
            device=device,
            shots=shots,
        )
        _restore_weights(quantum, {"main_circuit.weights": q_init})

        layers = {"pre_net": pre_net, "quantum": quantum, "post_net": post_net}
        self._pipeline = QuantumPipeline(list(layers.items()))
        for name, layer in layers.items():
            entry = layer._qnodes[self._LAYER_ENTRY[name]]
            self._qnodes[name] = entry
            if self.backend == "torch":
                setattr(self, name, entry)

    def __repr__(self):
        return (
            f"DressedQuantumClassifier(n_features={self.n_features}, "
            f"n_qubits={self.n_qubits}, n_layers={self.n_layers}, "
            f"n_classes={self.n_classes}, device='{self.device}')"
        )

    def forward(self, X, **custom_weights):
        """Run dense, circuit, dense and return class probabilities.

        Parameters
        ----------
        X : array-like
            Batch of ``n_features`` columns, normalized but not prescaled.
        **custom_weights
            Override the model's own weights, keyed as in `weights`.

        Returns
        -------
        array-like
            Probability of class 1 for binary; a `(n_samples, n_classes)`
            probability matrix otherwise.
        """
        if X.shape[-1] != self.n_features:
            raise ValueError(
                f"X has {X.shape[-1]} features but the model was built for "
                f"n_features={self.n_features}."
            )
        routed = {}
        for key, value in custom_weights.items():
            name, weight = key.split(".", 1)
            routed[f"{name}.{self._LAYER_ENTRY[name]}.{weight}"] = value
        return self._pipeline.forward(X, **routed)

    @classmethod
    def get_test_params(cls):
        """List constructor kwargs used to parametrize this class in the test suite."""
        return [
            {"n_features": 3, "n_qubits": 2, "n_layers": 2},
            {
                "n_features": 4,
                "n_qubits": 3,
                "n_layers": 1,
                "n_classes": 3,
                "trainer_kwargs": {"loss_fn": "cross_entropy", "check_bp": True},
            },
        ]
