import pennylane as qml
from skbase.utils.dependencies import _check_soft_dependencies

from pyqit.ansatzes.base import BaseAnsatz


class RealAmplitudesAnsatz(BaseAnsatz):
    """Qiskit's `RealAmplitudes`: RY layers separated by CX entanglers.

    The hardware-efficient ansatz of Kandala et al. 2017 (Nature) as Qiskit's
    circuit library builds it, and the default ansatz of Qiskit ML's `VQC`.
    The circuit is Qiskit's own, converted through the `pennylane-qiskit`
    plugin, so it needs the `qiskit` extra and Python 3.11 or newer.

    The weights are Qiskit's flat parameter vector, `weights`, of shape
    `(n_qubits * (n_layers + 1),)`. With `skip_final_rotation_layer` it is
    `(n_qubits * n_layers,)`.

    Parameters
    ----------
    n_qubits : int
    n_layers : int, default 3
        Qiskit's `reps`: the number of entangling blocks. Rotation layers
        number `n_layers + 1` unless `skip_final_rotation_layer`.
    entanglement : str, default "reverse_linear"
        Any entanglement Qiskit accepts, e.g. `"linear"`, `"full"`,
        `"circular"`.
    skip_final_rotation_layer : bool, default False

    References
    ----------
    Kandala et al., "Hardware-efficient variational quantum eigensolver for
    small molecules and quantum magnets", Nature 549, 242 (2017).

    Examples
    --------
    >>> from pyqit.ansatzes import RealAmplitudesAnsatz
    >>> from pyqit.core import ZZFeatureMap
    >>> from pyqit.models import VQCClassifier
    >>> model = VQCClassifier(
    ...     n_qubits=2, ansatz=RealAmplitudesAnsatz, encoder=ZZFeatureMap
    ... )
    """

    _tags = {"python_dependencies": "pennylane-qiskit"}
    _qiskit_circuit = "real_amplitudes"

    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 3,
        entanglement: str = "reverse_linear",
        skip_final_rotation_layer: bool = False,
    ):
        self.entanglement = entanglement
        self.skip_final_rotation_layer = skip_final_rotation_layer
        super().__init__(n_qubits, n_layers)

        if not _check_soft_dependencies("pennylane-qiskit", severity="none"):
            raise ImportError(
                f"{type(self).__name__} wraps Qiskit's circuit through the "
                "pennylane-qiskit plugin, which is not installed. Install it with "
                "`pip install pyqit[qiskit]`"
            )
        from qiskit.circuit import library

        self.circuit = getattr(library, self._qiskit_circuit)(
            n_qubits,
            entanglement=entanglement,
            reps=n_layers,
            skip_final_rotation_layer=skip_final_rotation_layer,
        )
        self._template = qml.from_qiskit(self.circuit)

    def build_circuit(self, weights):
        """Apply the circuit. Expects `weights["weights"]` as a flat vector in
        the order of `circuit.parameters`."""
        self._template(weights["weights"])

    def get_weight_shapes(self) -> dict:
        """Return `{"weights": (n_params,)}`, Qiskit's flat parameter vector."""
        return {"weights": (len(self.circuit.parameters),)}

    @classmethod
    def get_test_params(cls):
        """List constructor kwargs used to parametrize this class in the test suite."""
        return [
            {"n_qubits": 3, "n_layers": 2},
            {"n_qubits": 2, "n_layers": 1, "entanglement": "circular"},
        ]


class EfficientSU2Ansatz(RealAmplitudesAnsatz):
    """Qiskit's `EfficientSU2`: RY and RZ layers separated by CX entanglers.

    `RealAmplitudesAnsatz` with an RZ layer after every RY layer, Qiskit's
    default `su2_gates`. Same parameters and the same reference. The second
    rotation doubles the weight vector to `(2 * n_qubits * (n_layers + 1),)`.
    """

    _qiskit_circuit = "efficient_su2"
