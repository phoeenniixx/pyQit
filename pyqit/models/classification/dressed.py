import numpy as np
import pennylane as qml
import pennylane.numpy as pnp

from pyqit.models.base.quantum_model import BaseQuantumModel
from pyqit.models.classification.classifier_mixin import ClassifierMixin
from pyqit.models.layers.dense import init_dense_weights


class DressedQuantumClassifier(BaseQuantumModel, ClassifierMixin):
    """Dressed quantum circuit of Mari et al. (2020): dense, circuit, dense.

    A classical layer maps ``n_features`` to ``n_qubits`` angles through
    ``tanh(.) * pi / 2``; the circuit applies a Hadamard layer, encodes the
    angles with RY, then ``n_layers`` blocks of a CNOT ladder followed by an
    RY layer, and reads ``<Z>`` on every wire, computed from the basis-state
    probabilities so the QNode returns one tensor on both backends; a second
    classical layer maps those to the classes. There is no separate
    embedding, so the DataModule does not prescale.

    The head differs from the paper in one way: pyqit models emit
    probabilities, so binary applies a sigmoid to one logit and multi-class a
    softmax, rather than handing logits to cross-entropy.

    Parameters
    ----------
    n_features : int
        Input width. ``forward`` raises on any other width.
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

        n_out = 1 if n_classes == 2 else n_classes
        pre = init_dense_weights(n_features, n_qubits)
        q_shapes = {"weights": (n_layers, n_qubits)}
        q_init = {
            "weights": pnp.array(
                q_delta * np.random.randn(n_layers, n_qubits), requires_grad=True
            )
        }
        post = init_dense_weights(n_qubits, n_out)
        bits = (np.arange(2**n_qubits)[:, None] >> np.arange(n_qubits)[::-1]) & 1
        self._z_from_probs = 1.0 - 2.0 * bits

        dev = qml.device(self.device, wires=self.n_qubits)
        qnode = qml.set_shots(
            qml.QNode(self._circuit, dev, interface=self.get_interface()),
            shots=self.shots,
        )
        self.register_dense("pre_net", n_features, n_qubits, weights=pre)
        self.register_qnode("quantum", qnode, q_shapes, weights=q_init)
        self.register_dense("post_net", n_qubits, n_out, weights=post)

    def __repr__(self):
        return (
            f"DressedQuantumClassifier(n_features={self.n_features}, "
            f"n_qubits={self.n_qubits}, n_layers={self.n_layers}, "
            f"n_classes={self.n_classes}, device='{self.device}')"
        )

    def _circuit(self, inputs, weights):
        wires = range(self.n_qubits)
        for w in wires:
            qml.Hadamard(wires=w)
        for w in wires:
            qml.RY(inputs[..., w], wires=w)
        for layer in range(self.n_layers):
            for i in range(0, self.n_qubits - 1, 2):
                qml.CNOT(wires=[i, i + 1])
            for i in range(1, self.n_qubits - 1, 2):
                qml.CNOT(wires=[i, i + 1])
            for w in wires:
                qml.RY(weights[layer, w], wires=w)
        return qml.probs(wires=wires)

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
        q_in = qml.math.tanh(self.execute_qnode("pre_net", X, **custom_weights))
        probs = self.execute_qnode("quantum", q_in * (np.pi / 2.0), **custom_weights)
        q_out = qml.math.dot(probs, qml.math.cast_like(self._z_from_probs, probs))
        logits = self.execute_qnode("post_net", q_out, **custom_weights)
        if self.n_classes == 2:
            return 1.0 / (1.0 + qml.math.exp(-logits[..., 0]))
        exp = qml.math.exp(logits - qml.math.max(logits, axis=-1, keepdims=True))
        return exp / qml.math.sum(exp, axis=-1, keepdims=True)

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
