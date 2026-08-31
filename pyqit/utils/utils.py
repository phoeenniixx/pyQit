import logging

import numpy as np

logger = logging.getLogger("pyqit.diagnostics")
logger.setLevel(logging.INFO)


def _is_torch(x) -> bool:
    return type(x).__module__.startswith("torch")


def _to_numpy(x) -> np.ndarray:
    if _is_torch(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _ensure_col(out):
    if _is_torch(out):
        return out.unsqueeze(-1) if out.dim() == 1 else out
    out = np.asarray(out)
    return out.reshape(-1, 1) if out.ndim == 1 else out


def _cat(a, b, axis: int = 1):
    if _is_torch(a) or _is_torch(b):
        import torch

        if not _is_torch(b):
            b = torch.as_tensor(b, dtype=a.dtype, device=a.device)
        if not _is_torch(a):
            a = torch.as_tensor(a, dtype=b.dtype, device=b.device)
        return torch.cat([a, b], dim=axis)
    return np.concatenate([a, b], axis=axis)


def _stack(tensors):
    if any(_is_torch(t) for t in tensors):
        import torch

        tensors = [torch.as_tensor(t) if not _is_torch(t) else t for t in tensors]
        return torch.stack(tensors)
    return np.array(tensors)


def _mean(x, axis=0):
    if _is_torch(x):
        return x.mean(dim=axis)
    return np.mean(x, axis=axis)


def _round(x):
    if _is_torch(x):
        return x.round()
    return np.round(x)


def _count_params(model) -> int | None:
    """Total scalar trainable parameters of ``model``, or None if it exposes none."""
    weights = getattr(model, "weights", None)
    if not weights:
        return None
    return int(sum(np.prod(tuple(w.shape), dtype=int) for w in weights.values()))


def _snapshot_weights(model) -> dict:
    """Detached copy of ``model.weights``, safe to hold across further training.

    Torch parameters are live objects that the optimizer mutates in place, so a
    checkpoint that stored the references would silently track the *current*
    weights rather than the best ones.
    """
    return {k: _to_numpy(v).copy() for k, v in model.weights.items()}


def _restore_weights(model, snapshot: dict) -> None:
    """Write ``snapshot`` back into ``model``, on either backend.

    ``update_weights`` is a no-op under torch because autograd owns the
    ``nn.Parameter`` objects, so the torch path copies into them in place
    instead.
    """
    current = model.weights
    if any(_is_torch(v) for v in current.values()):
        import torch

        with torch.no_grad():
            for key, value in snapshot.items():
                param = current[key]
                param.copy_(torch.as_tensor(value, dtype=param.dtype))
        return

    import pennylane.numpy as pnp

    model.update_weights(
        {k: pnp.array(v, requires_grad=True) for k, v in snapshot.items()}
    )
