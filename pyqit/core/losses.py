import pennylane.numpy as pnp


def mse_loss(preds, targets):
    return pnp.mean((preds - targets) ** 2)


def hinge_loss(preds, targets):
    y_signed = 2.0 * targets - 1.0
    return pnp.mean(pnp.maximum(0, 1 - y_signed * preds))


def cross_entropy_loss(preds, targets):
    probs = pnp.clip(preds, 1e-9, 1.0 - 1e-9)

    if probs.ndim == 1 or (probs.ndim == 2 and probs.shape[1] == 1):
        probs = probs.flatten()
        log_p = targets * pnp.log(probs) + (1.0 - targets) * pnp.log(1.0 - probs)
        return -pnp.mean(log_p)

    n = len(targets)
    log_p = pnp.log(probs[pnp.arange(n), targets.astype(int)])
    return -pnp.mean(log_p)
