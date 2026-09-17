import numpy as np
import pennylane as qml

from pyqit.base import _PyQitObject
from pyqit.core.measurements import (
    measure_expval_x,
    measure_expval_z,
    measure_probs,
)
from pyqit.utils.utils import _is_torch


class ClassifierMixin(_PyQitObject):
    """Turns raw circuit output into class probabilities and hard labels.

    Mixed into a model alongside a ``BaseModel``, which supplies ``forward``.
    Binary models read one expectation value and threshold at 0.5;
    multi-class models bin the basis-state probabilities by index modulo
    ``n_classes``, the Qiskit ML ``VQC`` readout, and take the argmax.
    """

    _tags = {
        "estimator_type": "classifier",
        "bp_scale_factor": 0.25,
    }

    def _resolve_readout(self, n_qubits, measure_fn, measure_wires):
        """Validate ``n_classes`` against the circuit and fill measurement defaults.

        Sets ``_measure_fn`` and ``_measure_wires``. Binary reads
        ``measure_expval_z`` on wire 0; multi-class reads ``measure_probs`` on
        every wire.
        """
        n_classes = self.n_classes
        if n_classes > 2 and n_classes > 2**n_qubits:
            raise ValueError(
                f"Cannot classify {n_classes} classes with {n_qubits} qubits. "
                f"Maximum: {2**n_qubits}. Increase n_qubits or reduce n_classes."
            )

        if measure_fn is None:
            measure_fn = measure_expval_z if n_classes == 2 else measure_probs
        if measure_wires is None:
            measure_wires = [0] if n_classes == 2 else list(range(n_qubits))

        if (
            n_classes == 2
            and len(measure_wires) != 1
            and measure_fn in (measure_expval_z, measure_expval_x)
        ):
            raise ValueError(
                f"Binary classification reads one expectation value per sample, "
                f"but measure_wires={measure_wires} names "
                f"{len(measure_wires)} wires, which makes "
                f"{measure_fn.__name__} return a tuple. Pass exactly one "
                f"wire, set n_classes > 2, or supply a measure_fn that reduces "
                f"the wires to a single value."
            )
        self._measure_fn = measure_fn
        self._measure_wires = measure_wires

    def _to_probabilities(self, raw_output):
        """Map circuit output to class probabilities.

        Binary: ``(z + 1) / 2`` of one expectation value. Multi-class: the
        ``2 ** n_qubits`` basis probabilities summed by index modulo
        ``n_classes`` into a ``(n_samples, n_classes)`` matrix.
        """
        if self.n_classes == 2:
            return (raw_output + 1.0) / 2.0
        bins = np.eye(self.n_classes)[np.arange(raw_output.shape[-1]) % self.n_classes]
        return qml.math.dot(raw_output, qml.math.cast_like(bins, raw_output))

    def predict_step(self, X):
        """Predict hard class labels for ``X``.

        Parameters
        ----------
        X : array-like
            Input batch.

        Returns
        -------
        array-like
            One label per row: 0/1 for binary, argmax index for multi-class.
        """
        raw_output = self.forward(X)

        is_torch = _is_torch(raw_output)

        if self.n_classes == 2:
            if is_torch:
                preds = (raw_output >= 0.5).int()
            else:
                preds = (raw_output >= 0.5).astype(int)
        else:
            if raw_output.ndim > 1:
                if is_torch:
                    preds = raw_output.argmax(dim=1)
                else:
                    preds = raw_output.argmax(axis=1)
            else:
                if is_torch:
                    preds = raw_output.argmax(dim=0)
                else:
                    preds = raw_output.argmax(axis=0)
        if is_torch:
            import torch

            return torch.atleast_1d(preds)
        else:
            return np.atleast_1d(preds)
