import pennylane as qml

def measure_probs(wires):
    """Returns probabilities for the specified wires."""
    return qml.probs(wires=wires)

def measure_expval_z(wires):
    """Returns PauliZ expectation for EACH specified wire."""
    if len(wires) == 1:
        return qml.expval(qml.PauliZ(wires[0]))
    else:
        return tuple(qml.expval(qml.PauliZ(w)) for w in wires)

def measure_expval_x(wires):
    """Returns PauliX expectation (useful for some physics Hamiltonians)."""
    if len(wires) == 1:
        return qml.expval(qml.PauliX(wires[0]))
    else:
        return tuple(qml.expval(qml.PauliX(w)) for w in wires)