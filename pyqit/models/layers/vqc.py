import inspect

import numpy as np
import pennylane as qml

from pyqit.ansatzes.sel import SELAnsatz
from pyqit.core.embeddings import AngleEmbedding
from pyqit.models.base.quantum_model import BaseQuantumModel


def z_from_probs(n_qubits: int):
    """Matrix ``M`` with ``probs @ M`` the ``<Z>`` of every wire, wire 0 first.

    Reading ``qml.probs`` and projecting keeps the QNode's output one tensor
    on both backends; a tuple of ``qml.expval`` comes back from ``TorchLayer``
    in a version-dependent shape.
    """
    bits = (np.arange(2**n_qubits)[:, None] >> np.arange(n_qubits)[::-1]) & 1
    return 1.0 - 2.0 * bits


class BaseVQC(BaseQuantumModel):
    """An embedding, an ansatz, a measurement: the circuit the VQC models share.

    Builds the ansatz and the embedding from their classes, draws the weights,
    and registers one QNode under ``main_circuit``. It has no readout of its
    own, so you subclass it and never instantiate it directly. A subclass
    implements ``_resolve_readout(n_qubits, measure_fn, measure_wires)``,
    which sets ``_measure_fn`` and ``_measure_wires``, and ``forward``, which
    runs ``execute_qnode("main_circuit", X, **custom_weights)`` and maps the
    raw output. `VQCClassifier`, `VQCRegressor` and `QuantumLayer` differ only
    in those two methods.

    Parameters
    ----------
    n_qubits : int, default 4
    n_layers : int, default 3
        Depth passed to `ansatz`.
    ansatz : type, default SELAnsatz
        Ansatz class, not an instance.
    encoder : type, default AngleEmbedding
        Embedding class, not an instance. Stored as ``embedding_obj``, which
        drives prescaling.
    measure_fn : callable, optional
        Handed to ``_resolve_readout``, which picks the default.
    measure_wires : list of int, optional
        Handed to ``_resolve_readout``, which picks the default.
    device : str, default "default.qubit"
    shots : int, optional
    """

    def __init__(
        self,
        n_qubits=4,
        n_layers=3,
        ansatz=SELAnsatz,
        encoder=AngleEmbedding,
        measure_fn=None,
        measure_wires=None,
        device="default.qubit",
        shots=None,
    ):
        if not inspect.isclass(ansatz):
            raise TypeError(
                f"'ansatz' must be a class (e.g., SELAnsatz), "
                f"got {type(ansatz).__name__}"
            )

        if not inspect.isclass(encoder):
            raise TypeError(
                f"'encoder' must be a class (e.g., AngleEmbedding), "
                f"got {type(encoder).__name__}"
            )

        super().__init__(device=device, shots=shots)

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.ansatz = ansatz
        self.encoder = encoder
        self.measure_fn = measure_fn
        self.measure_wires = measure_wires

        self._ansatz_name = self.ansatz.__name__
        self._encoder_name = self.encoder.__name__

        self.ansatz_obj = self.ansatz(n_qubits=n_qubits, n_layers=n_layers)
        self.embedding_obj = self.encoder(n_qubits=n_qubits)

        self._resolve_readout(n_qubits, measure_fn, measure_wires)

        weight_shapes = self.ansatz_obj.get_weight_shapes()
        self.weight_keys = list(weight_shapes.keys())
        init_weights = self.init_weights(weight_shapes)

        dev = qml.device(self.device, wires=self.n_qubits)
        primary_qnode = qml.set_shots(
            qml.QNode(self._circuit, dev, interface=self.get_interface()),
            shots=self.shots,
        )

        self.register_qnode(
            "main_circuit", primary_qnode, weight_shapes, weights=init_weights
        )

    def _circuit(self, inputs, **weights):
        self.embedding_obj.forward(inputs)
        self.ansatz_obj.build_circuit(weights)
        return self._measure_fn(self._measure_wires)
