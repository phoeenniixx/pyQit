import pennylane as qml
import pennylane.numpy as pnp


def init_dense_weights(n_in: int, n_out: int) -> dict:
    """Draw ``torch.nn.Linear``'s default init from numpy's global RNG.

    Uniform ``[-1/sqrt(n_in), 1/sqrt(n_in))`` for weight and bias, so both
    backends start a dense layer from the same point for the same seed.
    """
    bound = 1.0 / n_in**0.5
    return {
        "weight": pnp.random.uniform(
            -bound, bound, size=(n_out, n_in), requires_grad=True
        ),
        "bias": pnp.random.uniform(-bound, bound, size=n_out, requires_grad=True),
    }


ACTIVATIONS = {
    None: lambda x: x,
    "tanh": qml.math.tanh,
    "relu": lambda x: x * (x > 0),
    "sigmoid": lambda x: 1.0 / (1.0 + qml.math.exp(-x)),
}


def to_probabilities(logits, n_classes: int):
    """Sigmoid of one logit for binary, a softmax over the last axis otherwise."""
    if n_classes == 2:
        return ACTIVATIONS["sigmoid"](logits[..., 0])
    exp = qml.math.exp(logits - qml.math.max(logits, axis=-1, keepdims=True))
    return exp / qml.math.sum(exp, axis=-1, keepdims=True)


def dense(X, weight, bias):
    """``X @ weight.T + bias`` on the pennylane backend.

    ``tensordot`` rather than ``@`` or ``dot``: it is the one that traces under
    autograd with a pnp input and ``ArrayBox`` weights.
    """
    return qml.math.tensordot(X, weight, axes=[[-1], [-1]]) + bias
