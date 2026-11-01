from collections.abc import Callable

import numpy as np
import pennylane as qml

from pyqit.core.config import _torch_dtype


def _as_2d(X):
    X = np.asarray(X, dtype=np.float64)
    return X.reshape(-1, 1) if X.ndim == 1 else X


def _is_torch_transform(fn) -> bool:
    return any(
        (getattr(cls, "__module__", None) or "").startswith("torch")
        for cls in type(fn).__mro__
    )


def _map_present(fn, arrays):
    return [None if a is None else fn(a) for a in arrays]


def _compose(fns: list) -> Callable | None:
    if not fns:
        return None
    if len(fns) == 1:
        return fns[0]

    def _chained(x):
        for fn in fns:
            x = fn(x)
        return x

    return _chained


class _NumpyLoader:
    def __init__(
        self, X, y, batch_size, shuffle, seed=None, transform=None, drop_last=False
    ):
        self.X, self.y = X, y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = transform
        self.drop_last = drop_last
        self._rng = np.random.default_rng(seed)

    def __iter__(self):
        n = len(self.X)
        idx = self._rng.permutation(n) if self.shuffle else np.arange(n)
        if self.drop_last:
            idx = idx[: n - n % self.batch_size]
        for start in range(0, len(idx), self.batch_size):
            b = idx[start : start + self.batch_size]
            Xb = self.X[b]
            if self.transform is not None:
                Xb = np.stack([self.transform(x) for x in Xb])
            yield Xb, self.y[b]

    def __len__(self):
        if self.drop_last:
            return len(self.X) // self.batch_size
        return int(np.ceil(len(self.X) / self.batch_size))

    def __repr__(self):
        t = f", transform={self.transform!r}" if self.transform else ""
        return (
            f"_NumpyLoader(n={len(self.X)}, batch_size={self.batch_size}, "
            f"shuffle={self.shuffle}{t})"
        )


def _make_torch_loader(
    X,
    y,
    batch_size,
    shuffle,
    num_workers=0,
    tensor_transform=None,
    drop_last=False,
    seed=None,
):
    import torch
    from torch.utils.data import DataLoader, Dataset, Sampler

    class _DS(Dataset):
        def __init__(self, X, y):
            self.X = torch.as_tensor(X, dtype=_torch_dtype())
            self.y = torch.as_tensor(y, dtype=_torch_dtype())

        def __len__(self):
            return len(self.X)

        def __getitem__(self, i):
            x = self.X[i]
            if tensor_transform is not None:
                x = tensor_transform(x)
            return x, self.y[i]

    class _NumpyOrder(Sampler):
        def __init__(self, n):
            self.n = n
            self._rng = np.random.default_rng(seed)

        def __iter__(self):
            return iter(self._rng.permutation(self.n).tolist())

        def __len__(self):
            return self.n

    return DataLoader(
        _DS(X, y),
        batch_size=batch_size,
        sampler=_NumpyOrder(len(X)) if shuffle else None,
        num_workers=num_workers,
        drop_last=drop_last,
    )


def _unit_rows(X, ord=2):
    n = len(X)
    norms = np.linalg.norm(X.reshape(n, -1), ord=ord, axis=1)
    norms = norms.reshape(n, *[1] * (X.ndim - 1))
    return X / np.where(norms == 0, 1.0, norms)


class _Normalizer:
    METHODS = ("minmax", "zscore", "l2", "l1")

    def __init__(self, method: str):
        if method not in self.METHODS:
            raise ValueError(
                f"Unknown normalizer {method!r}. Choose one of {self.METHODS}.\n"
                f"For torch transforms pass them via transform=."
            )
        self.method = method
        self._params: dict = {}
        self._fitted = False

    def fit(self, X) -> "_Normalizer":
        axes = tuple(range(X.ndim - 1))
        if self.method == "minmax":
            lo = X.min(axis=axes)
            hi = X.max(axis=axes)
            self._params = {"lo": lo, "rng": np.where(hi - lo == 0, 1.0, hi - lo)}
        elif self.method == "zscore":
            mu = X.mean(axis=axes)
            sigma = X.std(axis=axes)
            self._params = {"mu": mu, "sigma": np.where(sigma == 0, 1.0, sigma)}
        self._fitted = True
        return self

    def transform(self, X):
        if self.method in ("minmax", "zscore") and not self._fitted:
            raise RuntimeError(
                f"normalize={self.method!r} scales with statistics of the training "
                "split, but this DataModule has never been fit. Fit it first, "
                "clone_empty() one that was, or pass normalize=None."
            )
        if self.method == "minmax":
            return (X - self._params["lo"]) / self._params["rng"]
        if self.method == "zscore":
            return (X - self._params["mu"]) / self._params["sigma"]
        return _unit_rows(X, ord=2 if self.method == "l2" else 1)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def __repr__(self):
        return f"_Normalizer(method={self.method!r}, fitted={self._fitted})"


def _fit_width(X, width):
    if X.shape[1] > width:
        raise ValueError(
            f"X has {X.shape[1]} features but the embedding takes at most {width}; "
            "the rest would be silently dropped. Select or reduce features first."
        )
    if X.shape[1] < width:
        zeros = qml.math.zeros_like(X[:, :1])[:, [0] * (width - X.shape[1])]
        return qml.math.concatenate([X, zeros], axis=1)
    return X


def _prescale_angle_pi(X, n_qubits):
    return _fit_width(X, n_qubits) * np.pi


def _prescale_angle_half_pi(X, n_qubits):
    return _fit_width(X, n_qubits) * (np.pi / 2)


def _prescale_amplitude(X, n_qubits):
    X = _fit_width(X, 2**n_qubits)
    sq = qml.math.sum(X**2, axis=1, keepdims=True)
    return X / qml.math.sqrt(qml.math.where(sq == 0, 1.0, sq))


def _prescale_binary(X, n_qubits):
    return _fit_width((X >= 0.5).astype(np.float64), n_qubits)


_PRESCALE_FNS = {
    "angle_pi": _prescale_angle_pi,
    "angle_half_pi": _prescale_angle_half_pi,
    "amplitude": _prescale_amplitude,
    "binary": _prescale_binary,
}


def _flatten_samples(X):
    if qml.math.ndim(X) <= 2:
        return X
    return qml.math.reshape(X, (qml.math.shape(X)[0], -1))


def _apply_prescale(X, prescale: str | None, n_qubits: int):
    X = _flatten_samples(X)
    if prescale in (None, "none"):
        return X
    if prescale not in _PRESCALE_FNS:
        raise ValueError(
            f"Unknown PRESCALE {prescale!r}. "
            f"Valid: {[*_PRESCALE_FNS, 'none']}. "
            f"Check your embedding class's PRESCALE attribute."
        )
    return _PRESCALE_FNS[prescale](X, n_qubits)
