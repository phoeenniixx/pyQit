import logging

import matplotlib.pyplot as plt
import numpy as np

from pyqit.core._loss_mapping import get_loss_fn
from pyqit.core.config import get_backend

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
