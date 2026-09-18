import pennylane as qml


def measure_probs(wires):
    """Return the ``2 ** len(wires)`` basis-state probabilities of `wires`."""
    return qml.probs(wires=wires)


def measure_expval_z(wires):
    """Return the PauliZ expectation of each wire in `wires`.

    One wire gives a single value in ``[-1, 1]``, several give a tuple with
    one value per wire.
    """
    if len(wires) == 1:
        return qml.expval(qml.PauliZ(wires[0]))
    else:
        return tuple(qml.expval(qml.PauliZ(w)) for w in wires)


def measure_expval_x(wires):
    """Return the PauliX expectation of each wire in `wires`.

    Same shapes as `measure_expval_z`.
    """
    if len(wires) == 1:
        return qml.expval(qml.PauliX(wires[0]))
    else:
        return tuple(qml.expval(qml.PauliX(w)) for w in wires)


def measure_parity_z(wires):
    """Return the expectation of the PauliZ parity ``Z ⊗ ... ⊗ Z`` over `wires`."""
    return qml.expval(qml.prod(*(qml.PauliZ(w) for w in wires)))
