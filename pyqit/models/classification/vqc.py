from pyqit.ansatzes.sel import SELAnsatz
from pyqit.core.embeddings import AngleEmbedding
from pyqit.models.classification.classifier_mixin import ClassifierMixin
from pyqit.models.layers.vqc import _VQC


class VQCClassifier(_VQC, ClassifierMixin):
    """Variational quantum classifier: an embedding, an ansatz, a measurement.

    Parameters
    ----------
    n_qubits : int, default 4
    n_layers : int, default 3
        Depth passed to `ansatz`.
    ansatz : type, default SELAnsatz
        Ansatz class, not an instance.
    encoder : type, default AngleEmbedding
        Embedding class, not an instance. Drives `DataModule` prescaling.
    n_classes : int, default 2
        Binary reads one expectation value; multi-class bins the
        `2 ** n_qubits` basis-state probabilities by index modulo `n_classes`,
        the Qiskit ML `VQC` readout.
    measure_fn : callable, optional
        Defaults to `measure_expval_z` for binary, `measure_probs` otherwise.
    measure_wires : list of int, optional
        Defaults to `[0]` for binary, all wires otherwise.
    device : str, default "default.qubit"
        Any PennyLane device name.
    shots : int, optional
        `None` runs analytic (infinite-shot) simulation.

    References
    ----------
    Havlicek et al., "Supervised learning with quantum-enhanced feature
    spaces", Nature 567, 209 (2019). Readout follows Qiskit ML's ``VQC``.

    Examples
    --------
    >>> import pyqit
    >>> from pyqit.models import VQCClassifier
    >>> model = VQCClassifier(n_qubits=4, n_layers=2)
    >>> history = pyqit.Trainer(max_epochs=5).fit(model, dm)  # doctest: +SKIP
    """

    def __init__(
        self,
        n_qubits=4,
        n_layers=3,
        ansatz=SELAnsatz,
        encoder=AngleEmbedding,
        n_classes=2,
        measure_fn=None,
        measure_wires=None,
        device="default.qubit",
        shots=None,
    ):
        self.n_classes = n_classes
        super().__init__(
            n_qubits=n_qubits,
            n_layers=n_layers,
            ansatz=ansatz,
            encoder=encoder,
            measure_fn=measure_fn,
            measure_wires=measure_wires,
            device=device,
            shots=shots,
        )

    def __repr__(self):
        return (
            f"VQCClassifier(n_qubits={self.n_qubits}, n_layers={self.n_layers}, "
            f"n_classes={self.n_classes}, ansatz={self._ansatz_name}, "
            f"encoder={self._encoder_name}, device='{self.device}')"
        )

    def forward(self, X, **custom_weights):
        """Run the circuit and return class probabilities.

        Parameters
        ----------
        X : array-like
            Batch, already prescaled by the DataModule.
        **custom_weights
            Override the model's own weights, keyed as in `weights`.

        Returns
        -------
        array-like
            Probability of class 1 for binary; a `(n_samples, n_classes)`
            probability matrix otherwise.
        """
        raw_output = self.execute_qnode("main_circuit", X, **custom_weights)
        return self._to_probabilities(raw_output)

    @classmethod
    def get_test_params(cls):
        """List constructor kwargs used to parametrize this class in the test suite."""
        from pyqit.core.embeddings import (
            AmplitudeEmbedding,
            IQPEmbedding,
            ZZFeatureMap,
        )

        return [
            {},
            {
                "n_qubits": 3,
                "n_layers": 2,
                "n_classes": 2,
                "ansatz": SELAnsatz,
                "encoder": IQPEmbedding,
                "trainer_kwargs": {"check_bp": True, "loss_fn": "hinge"},
            },
            {
                "n_qubits": 4,
                "n_layers": 3,
                "n_classes": 4,
                "ansatz": SELAnsatz,
                "encoder": AmplitudeEmbedding,
                "trainer_kwargs": {"loss_fn": "cross_entropy"},
            },
            {"n_qubits": 3, "n_layers": 1, "encoder": ZZFeatureMap},
        ]
