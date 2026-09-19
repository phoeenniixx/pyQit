import pennylane.numpy as pnp

from pyqit.ansatzes.sel import SELAnsatz
from pyqit.core.embeddings import AngleEmbedding
from pyqit.core.measurements import measure_parity_z
from pyqit.models.layers.vqc import BaseVQC
from pyqit.models.regression.regressor_mixin import RegressorMixin


class VQCRegressor(BaseVQC, RegressorMixin):
    """Variational quantum regressor: the ``VQCClassifier`` circuit read as a value.

    The circuit is Qiskit ML's ``VQR``: feature map, ansatz, and the parity
    observable ``Z ⊗ ... ⊗ Z`` over every wire by default, an expectation in
    ``[-1, 1]``. On top of it sits a trainable affine head
    ``scale * <Z> + offset``, starting at identity. Mitarai et al. train the
    scale; the offset goes one step further so uncentred targets need no
    preprocessing. ``output_scale=False`` drops the head and reproduces
    ``VQR``, whose targets must then lie in ``[-1, 1]``.

    Parameters
    ----------
    n_qubits : int, default 4
    n_layers : int, default 3
        Depth passed to `ansatz`.
    ansatz : type, default SELAnsatz
        Ansatz class, not an instance.
    encoder : type, default AngleEmbedding
        Embedding class, not an instance. Drives `DataModule` prescaling.
    measure_fn : callable, optional
        Defaults to `measure_parity_z`. Must return a single scalar per sample.
    measure_wires : list of int, optional
        Defaults to every wire.
    output_scale : bool, default True
        Train ``scale`` and ``offset`` on the expectation value. Their keys
        are ``output.weight`` and ``output.bias``.
    device : str, default "default.qubit"
    shots : int, optional
    diff_method : str, default "best"
        Passed to the QNode. ``"best"`` picks backprop on a simulator;
        ``"parameter-shift"`` rehearses a hardware run's gradient cost.

    References
    ----------
    Mitarai, Negoro, Kitagawa, Fujii, "Quantum circuit learning", Phys. Rev.
    A 98, 032309 (2018). Defaults follow Qiskit ML's ``VQR``.

    Examples
    --------
    >>> import pyqit
    >>> from pyqit.models import VQCRegressor
    >>> model = VQCRegressor(n_qubits=2, n_layers=2)
    >>> history = pyqit.Trainer(max_epochs=5).fit(model, dm)  # doctest: +SKIP
    """

    def __init__(
        self,
        n_qubits=4,
        n_layers=3,
        ansatz=SELAnsatz,
        encoder=AngleEmbedding,
        measure_fn=None,
        measure_wires=None,
        output_scale=True,
        device="default.qubit",
        shots=None,
        diff_method="best",
    ):
        self.output_scale = output_scale
        super().__init__(
            n_qubits=n_qubits,
            n_layers=n_layers,
            ansatz=ansatz,
            encoder=encoder,
            measure_fn=measure_fn,
            measure_wires=measure_wires,
            device=device,
            shots=shots,
            diff_method=diff_method,
        )
        if output_scale:
            identity = {
                "weight": pnp.array([[1.0]], requires_grad=True),
                "bias": pnp.array([0.0], requires_grad=True),
            }
            self.register_dense("output", 1, 1, weights=identity)

    def _resolve_readout(self, n_qubits, measure_fn, measure_wires):
        self._measure_fn = measure_parity_z if measure_fn is None else measure_fn
        self._measure_wires = (
            list(range(n_qubits)) if measure_wires is None else measure_wires
        )

    def __repr__(self):
        return (
            f"VQCRegressor(n_qubits={self.n_qubits}, n_layers={self.n_layers}, "
            f"ansatz={self._ansatz_name}, encoder={self._encoder_name}, "
            f"device='{self.device}')"
        )

    def forward(self, X, **custom_weights):
        """Run the circuit and return one value per sample.

        Parameters
        ----------
        X : array-like
            Batch, already prescaled by the DataModule.
        **custom_weights
            Override the model's own weights, keyed as in `weights`.

        Returns
        -------
        array-like
            Shape ``(n_samples,)``.
        """
        z = self.execute_qnode("main_circuit", X, **custom_weights)
        if not self.output_scale:
            return z
        return self.execute_qnode("output", z[..., None], **custom_weights)[..., 0]

    @classmethod
    def get_test_params(cls):
        """List constructor kwargs used to parametrize this class in the test suite."""
        return [
            {"n_qubits": 2, "n_layers": 2},
            {
                "n_qubits": 3,
                "n_layers": 1,
                "measure_wires": [0],
                "output_scale": False,
            },
        ]
