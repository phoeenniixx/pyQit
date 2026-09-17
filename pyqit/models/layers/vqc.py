import inspect

import pennylane as qml

from pyqit.ansatzes.sel import SELAnsatz
from pyqit.core.embeddings import AngleEmbedding
from pyqit.models.base.quantum_model import BaseQuantumModel


class _VQC(BaseQuantumModel):
    """An embedding, an ansatz, a measurement: the circuit shared by the VQC models.

    Subclasses set ``_measure_fn`` and ``_measure_wires`` in
    ``_resolve_readout`` and map the raw output in ``forward``.
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
