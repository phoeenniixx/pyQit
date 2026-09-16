import math

import pennylane as qml

from pyqit.models.base.quantum_model import BaseQuantumModel
from pyqit.models.classification.classifier_mixin import ClassifierMixin


class DataReuploadingClassifier(BaseQuantumModel, ClassifierMixin):
    """Data re-uploading classifier of Perez-Salinas et al. (2020).

    Every layer re-encodes the input: on each qubit it applies
    ``Rot(theta + w * x)`` (their Eq. 7), with ``x`` zero-padded to a multiple
    of three and consumed three features per rotation. With more than one
    qubit, a CZ chain entangles neighbouring wires between layers (their
    Sec. 4). There is no separate embedding, so the DataModule does not
    prescale; the model reads the normalized features directly.

    The readout and the loss are pyqit's, not the paper's fidelity cost:
    binary reads ``(1 + <Z_0>) / 2``, multi-class bins basis-state
    probabilities by index modulo ``n_classes``.

    Parameters
    ----------
    n_features : int
        Input width. ``forward`` raises on any other width.
    n_qubits : int, default 1
    n_layers : int, default 3
        Re-uploading layers.
    n_classes : int, default 2
    measure_fn : callable, optional
        Defaults to `measure_expval_z` for binary, `measure_probs` otherwise.
    measure_wires : list of int, optional
        Defaults to `[0]` for binary, all wires otherwise.
    device : str, default "default.qubit"
    shots : int, optional

    References
    ----------
    Perez-Salinas, Cervera-Lierta, Gil-Fuster, Latorre, "Data re-uploading
    for a universal quantum classifier", Quantum 4, 226 (2020). PennyLane's
    "Data re-uploading classifier" demo is the reference implementation.

    Examples
    --------
    >>> import pyqit
    >>> from pyqit.models import DataReuploadingClassifier
    >>> model = DataReuploadingClassifier(n_features=2, n_qubits=1, n_layers=4)
    >>> history = pyqit.Trainer(max_epochs=5).fit(model, dm)  # doctest: +SKIP
    """

    def __init__(
        self,
        n_features,
        n_qubits=1,
        n_layers=3,
        n_classes=2,
        measure_fn=None,
        measure_wires=None,
        device="default.qubit",
        shots=None,
    ):
        super().__init__(device=device, shots=shots)
        self.n_features = n_features
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.measure_fn = measure_fn
        self.measure_wires = measure_wires

        self._n_chunks = math.ceil(n_features / 3)
        self._pad = 3 * self._n_chunks - n_features
        self._resolve_readout(n_qubits, measure_fn, measure_wires)

        shape = (n_layers, n_qubits, self._n_chunks, 3)
        weight_shapes = {"theta": shape, "w": shape}
        init_weights = self.init_weights(weight_shapes)

        dev = qml.device(self.device, wires=self.n_qubits)
        qnode = qml.set_shots(
            qml.QNode(self._circuit, dev, interface=self.get_interface()),
            shots=self.shots,
        )
        self.register_qnode("main_circuit", qnode, weight_shapes, weights=init_weights)

    def __repr__(self):
        return (
            f"DataReuploadingClassifier(n_features={self.n_features}, "
            f"n_qubits={self.n_qubits}, n_layers={self.n_layers}, "
            f"n_classes={self.n_classes}, device='{self.device}')"
        )

    def _circuit(self, inputs, theta, w):
        x = inputs
        if self._pad:
            zeros = qml.math.zeros_like(x[..., :1])
            x = qml.math.concatenate([x] + [zeros] * self._pad, axis=-1)
        for layer in range(self.n_layers):
            for q in range(self.n_qubits):
                for c in range(self._n_chunks):
                    v = w[layer, q, c] * x[..., 3 * c : 3 * c + 3] + theta[layer, q, c]
                    qml.Rot(v[..., 0], v[..., 1], v[..., 2], wires=q)
            if layer < self.n_layers - 1:
                for q in range(self.n_qubits - 1):
                    qml.CZ(wires=[q, q + 1])
        return self._measure_fn(self._measure_wires)

    def forward(self, X, **custom_weights):
        """Run the circuit and return class probabilities.

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
        raw_output = self.execute_qnode("main_circuit", X, **custom_weights)
        return self._to_probabilities(raw_output)

    @classmethod
    def get_test_params(cls):
        """List constructor kwargs used to parametrize this class in the test suite."""
        return [
            {"n_features": 2, "n_qubits": 1, "n_layers": 2},
            {
                "n_features": 4,
                "n_qubits": 2,
                "n_layers": 2,
                "n_classes": 3,
                "trainer_kwargs": {"loss_fn": "cross_entropy"},
            },
        ]
