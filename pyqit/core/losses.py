import pennylane.numpy as pnp


def mse_loss(preds, targets):
    """Mean Squared Error loss function.
    Parameters
    ----------
    preds : array-like
        The predicted values from the model.
    targets : array-like
        The ground truth target values. Expected to have the same shape
        as `preds`.

    Returns
    -------
    float or tensor
        The computed mean squared error loss across the batch.
    """
    return pnp.mean((preds - targets) ** 2)


def hinge_loss(preds, targets):
    """Hinge loss function for binary classification.
    Parameters
    ----------
    preds : array-like
        The predicted raw scores (logits) from the model.
    targets : array-like
        The ground truth binary labels, expected to be encoded as 0 or 1.

    Returns
    -------
    float or tensor
        The computed mean hinge loss across the batch."""
    y_signed = 2.0 * targets - 1.0
    return pnp.mean(pnp.maximum(0, 1 - y_signed * preds))


def cross_entropy_loss(preds, targets):
    """Cross-entropy loss function for binary and multi-class classification.

    Parameters
    ----------
    preds : array-like
        The predicted probabilities from the model. For binary classification,
        this should be a 1D array or a 2D array of shape `(n_samples, 1)`.
        For multi-class classification, this should be a 2D array of shape
        `(n_samples, n_classes)`.
    targets : array-like
        The ground truth labels. For binary classification, values should be
        0 or 1. For multi-class classification, values should be integer
        class indices in the range `[0, n_classes - 1]`.
    Returns
    -------
    float or tensor
        The computed mean cross-entropy loss across the batch."""
    probs = pnp.clip(preds, 1e-9, 1.0 - 1e-9)

    if probs.ndim == 1 or (probs.ndim == 2 and probs.shape[1] == 1):
        probs = probs.flatten()
        log_p = targets * pnp.log(probs) + (1.0 - targets) * pnp.log(1.0 - probs)
        return -pnp.mean(log_p)

    n = len(targets)
    if pnp.max(targets) >= probs.shape[1] or pnp.min(targets) < 0:
        raise ValueError(
            f"Target mismatch: Model output {probs.shape[1]} classes, "
            f"but targets contain class index {pnp.max(targets)}. "
            "Ensure the model's `n_classes` matches the number of "
            "unique classes in your dataset."
        )

    log_p = pnp.log(probs[pnp.arange(n), targets.astype(int)])
    return -pnp.mean(log_p)
