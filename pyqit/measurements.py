import pennylane as qml

def measure_probs(wires):
    """Returns probabilities for the specified wires."""
    return qml.probs(wires=wires)

def measure_expval_z(wires):
    """Returns PauliZ expectation for EACH specified wire."""
    return [qml.expval(qml.PauliZ(w)) for w in wires]

def measure_expval_x(wires):
    """Returns PauliX expectation (useful for some physics Hamiltonians)."""
    return [qml.expval(qml.PauliX(w)) for w in wires]