from collections.abc import Callable
import copy
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from pyqit.core.config import get_backend


def _is_torch_transform(fn) -> bool:
    return any(
        (getattr(cls, "__module__", None) or "").startswith("torch")
        for cls in type(fn).__mro__
    )


def _map_present(fn, arrays):
    return [None if a is None else fn(a) for a in arrays]


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
    X, y, batch_size, shuffle, num_workers=0, tensor_transform=None, drop_last=False
):
    import torch
    from torch.utils.data import DataLoader, Dataset

    class _DS(Dataset):
        def __init__(self, X, y):
            self.X = torch.as_tensor(X, dtype=torch.float32)
            self.y = torch.as_tensor(y, dtype=torch.float32)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, i):
            x = self.X[i]
            if tensor_transform is not None:
                x = tensor_transform(x)
            return x, self.y[i]

    return DataLoader(
        _DS(X, y),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )


def _unit_rows(X, ord=2):
    norms = np.linalg.norm(X, ord=ord, axis=1, keepdims=True)
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
        if self.method == "minmax":
            lo = X.min(axis=0)
            hi = X.max(axis=0)
            self._params = {"lo": lo, "rng": np.where(hi - lo == 0, 1.0, hi - lo)}
        elif self.method == "zscore":
            mu = X.mean(axis=0)
            sigma = X.std(axis=0)
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
        return np.hstack([X, np.zeros((len(X), width - X.shape[1]))])
    return X


def _prescale_angle_pi(X, n_qubits):
    return _fit_width(X, n_qubits) * np.pi


def _prescale_amplitude(X, n_qubits):
    return _unit_rows(_fit_width(X, 2**n_qubits))


def _prescale_binary(X, n_qubits):
    return _fit_width((X >= 0.5).astype(np.float64), n_qubits)


_PRESCALE_FNS = {
    "angle_pi": _prescale_angle_pi,
    "amplitude": _prescale_amplitude,
    "binary": _prescale_binary,
}


def _apply_prescale(X, prescale: str | None, n_qubits: int):
    if prescale in (None, "none"):
        return X
    if prescale not in _PRESCALE_FNS:
        raise ValueError(
            f"Unknown PRESCALE {prescale!r}. "
            f"Valid: {[*_PRESCALE_FNS, 'none']}. "
            f"Check your embedding class's PRESCALE attribute."
        )
    return _PRESCALE_FNS[prescale](X, n_qubits)


class DataModule:
    """Lazy split, normalize, and prescale for a classical dataset.

    Nothing runs until `setup()`, which `Trainer.fit`/`.predict` call for you.
    Order is split, then stateful normalization fit on train only, then
    stateless quantum prescaling driven by the model's embedding.

    Parameters
    ----------
    X, y : array-like
    name : str, default "dataset"
    normalize : {"minmax", "zscore", "l2", "l1"}, optional
    split : tuple of float, default (0.70, 0.15, 0.15)
        Train, val, test fractions. Must sum to 1.0.
    stratify : bool, default False
    seed : int, optional, default 42
    batch_size : int, default 32
    num_workers : int, default 0
        Torch backend only.
    transform : callable or list of callable, optional
    shuffle : bool, default True
    drop_last : bool, default False

    Examples
    --------
    >>> import pyqit
    >>> dm = pyqit.DataModule(X, y, normalize="minmax", batch_size=16)
    >>> history = pyqit.Trainer(max_epochs=10).fit(model, dm)  # doctest: +SKIP
    """

    _RECONFIGURABLE = (
        "normalize",
        "split",
        "stratify",
        "seed",
        "batch_size",
        "num_workers",
        "transform",
    )

    def __init__(
        self,
        X,
        y,
        name: str = "dataset",
        normalize: str | None = None,
        split: tuple = (0.70, 0.15, 0.15),
        stratify: bool = False,
        seed: int | None = 42,
        batch_size: int = 32,
        num_workers: int = 0,
        transform: Callable | list | None = None,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        self._validate(normalize, split)

        self.X_raw = np.asarray(X, dtype=np.float64)
        self.y_raw = np.asarray(y, dtype=np.float64)
        if self.X_raw.ndim == 1:
            self.X_raw = self.X_raw.reshape(-1, 1)

        self.name = name
        self.normalize = normalize
        self.split = tuple(split)
        self.stratify = stratify
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.encoder = None
        self.n_qubits = None

        self._backend = get_backend()
        self._normalizer = None
        self._numpy_transform = None
        self._torch_transform = None
        self._X_train = self._y_train = None
        self._X_val = self._y_val = None
        self._X_test = self._y_test = None
        self._is_setup = False

    @staticmethod
    def _validate(normalize, split):
        if normalize is not None and normalize not in _Normalizer.METHODS:
            raise ValueError(
                f"normalize={normalize!r} not supported.\n"
                f"Numpy normalizers: {_Normalizer.METHODS}\n"
                f"For torch transforms pass them via transform=."
            )
        if len(split) != 3:
            raise ValueError(
                f"split must be a 3-tuple (train, val, test), got {split!r}."
            )
        train, val, test = split
        if abs(train + val + test - 1.0) > 1e-6:
            raise ValueError(
                f"split fractions must sum to 1.0, "
                f"got {train + val + test:.6f} ({train}, {val}, {test})."
            )
        if train <= 0 and val <= 0 and test <= 0:
            raise ValueError("At least one fraction must be > 0.")

    @classmethod
    def from_numpy(cls, X, y, **kw):
        """Build like the constructor; kept for a consistent `from_*` family."""
        return cls(X, y, **kw)

    @classmethod
    def from_sklearn(cls, loader: Callable, **kw) -> "DataModule":
        """Build from an sklearn dataset loader.

        Parameters
        ----------
        loader : callable
            E.g. `sklearn.datasets.load_iris`.
        **kw
            Forwarded to the constructor.

        Examples
        --------
        >>> from sklearn.datasets import load_iris
        >>> dm = pyqit.DataModule.from_sklearn(load_iris)  # doctest: +SKIP
        """
        bunch = loader()
        name = kw.pop("name", loader.__name__.replace("load_", ""))
        return cls(bunch.data, bunch.target.astype(np.float64), name=name, **kw)

    @classmethod
    def from_dataframe(cls, df: Any, label_col: str | int = -1, **kw) -> "DataModule":
        """Build from a DataFrame, splitting off `label_col` as `y`.

        Parameters
        ----------
        df : pandas.DataFrame
        label_col : str or int, default -1
            Column name, or a position (negative indexes from the end).
        **kw
            Forwarded to the constructor.
        """
        col = (
            df.columns[label_col % len(df.columns)]
            if isinstance(label_col, int)
            else label_col
        )
        if col not in df.columns:
            raise KeyError(f"{col!r} not in {list(df.columns)}")
        name = kw.pop("name", "dataframe")
        return cls(
            df.drop(columns=[col]).to_numpy(float),
            df[col].to_numpy(float),
            name=name,
            **kw,
        )

    @classmethod
    def from_csv(
        cls, path: str | Path, label_col: str | int = -1, delimiter: str = ",", **kw
    ) -> "DataModule":
        """Build from a CSV file. See `from_dataframe` for `label_col`."""
        df = pd.read_csv(path, sep=delimiter)
        name = kw.pop("name", Path(path).stem)
        return cls.from_dataframe(df, label_col=label_col, name=name, **kw)

    def to_lightning(self):
        """Converts itself to a Lightning adapter if the backend requires it."""
        if self._backend != "torch":
            raise ValueError(
                f"Cannot generate a Lightning adapter for {self._backend} backend."
            )

        from pyqit.core.adapters.lightning import _LightningDataAdapter

        return _LightningDataAdapter(self)

    def setup(
        self,
        stage: str | None = None,
        batch_size: int | None = None,
        n_qubits: int | None = None,
        encoder: type | None = None,
        force: bool = False,
    ) -> "DataModule":
        """Split, normalize, and prescale. Idempotent unless `force=True`.

        `Trainer.fit`/`.predict` call this for you, passing `n_qubits` and
        `encoder` from the model so quantum prescaling is never skipped.

        Parameters
        ----------
        stage : {"fit", "val", "test", "predict"}, optional
            `"predict"` skips the split and uses the whole dataset as test.
            Every other stage splits.
        batch_size : int, optional
            Applied even when already set up.
        n_qubits : int, optional
            From the model. Persists across a later `force=True` call that
            omits it.
        encoder : type, optional
            Embedding class. Same persistence as `n_qubits`.
        force : bool, default False
            Redo the split even if already set up.

        Returns
        -------
        DataModule
            `self`.
        """
        if batch_size is not None:
            self.batch_size = batch_size

        if self._is_setup and not force:
            return self

        if encoder is not None:
            self.encoder = encoder
        if n_qubits is not None:
            self.n_qubits = n_qubits

        fns = self.transform if isinstance(self.transform, list) else [self.transform]
        fns = [fn for fn in fns if fn is not None]
        torch_fns = [fn for fn in fns if _is_torch_transform(fn)]
        if torch_fns and self._backend == "pennylane":
            names = [type(fn).__name__ for fn in torch_fns]
            raise RuntimeError(
                f"Torch transform(s) {names} cannot be used with backend='pennylane'."
            )
        self._numpy_transform = _compose(
            [fn for fn in fns if not _is_torch_transform(fn)]
        )
        self._torch_transform = _compose(torch_fns)

        if stage == "predict":
            Xs, ys = [None, None, self.X_raw], [None, None, self.y_raw]
        else:
            Xs, ys = self._do_split()

        if self.normalize is not None:
            if Xs[0] is not None:
                self._normalizer = _Normalizer(self.normalize).fit(Xs[0])
            elif self._normalizer is None:
                self._normalizer = _Normalizer(self.normalize)
            Xs = _map_present(self._normalizer.transform, Xs)

        prescale = self.encoder.PRESCALE if self.encoder is not None else None
        if prescale is not None:
            if self.n_qubits is None:
                raise RuntimeError("Prescaling requires n_qubits.")
            Xs = _map_present(lambda X: _apply_prescale(X, prescale, self.n_qubits), Xs)

        if self._numpy_transform is not None and self._backend == "torch":
            t = self._numpy_transform
            Xs = _map_present(lambda X: np.stack([t(x) for x in X]), Xs)

        self._X_train, self._X_val, self._X_test = Xs
        self._y_train, self._y_val, self._y_test = ys
        self._is_setup = True
        return self

    def train_loader(self, shuffle: bool | None = None, drop_last: bool | None = None):
        """Build a `DataLoader` (torch) or `_NumpyLoader` (pennylane) over train.

        ``shuffle`` and ``drop_last`` default to this DataModule's own settings.
        """
        self._assert_setup("train_loader")
        return self._make_loader(
            self._X_train,
            self._y_train,
            self.shuffle if shuffle is None else shuffle,
            self.drop_last if drop_last is None else drop_last,
        )

    def val_loader(self, shuffle: bool = False):
        """Build a loader like `train_loader`, over val. `None` with no val split."""
        self._assert_setup("val_loader")
        return self._make_loader(self._X_val, self._y_val, shuffle)

    def test_loader(self, shuffle: bool = False):
        """Build a loader like `train_loader`, over test. `None` with no test split."""
        self._assert_setup("test_loader")
        return self._make_loader(self._X_test, self._y_test, shuffle)

    def _make_loader(self, X, y, shuffle, drop_last=False):
        if X is None:
            return None
        if self._backend == "torch":
            return _make_torch_loader(
                X,
                y,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                tensor_transform=self._torch_transform,
                drop_last=drop_last,
            )
        return _NumpyLoader(
            X,
            y,
            self.batch_size,
            shuffle,
            seed=self.seed,
            transform=self._numpy_transform,
            drop_last=drop_last,
        )

    @property
    def X_train(self):
        """Train split features. Raises before `setup()`."""
        self._assert_setup("X_train")
        return self._X_train

    @property
    def y_train(self):
        """Train split targets. Raises before `setup()`."""
        self._assert_setup("y_train")
        return self._y_train

    @property
    def X_val(self):
        """Val split features, or `None` with no val split. Raises before `setup()`."""
        self._assert_setup("X_val")
        return self._X_val

    @property
    def y_val(self):
        """Val split targets, or `None` with no val split. Raises before `setup()`."""
        self._assert_setup("y_val")
        return self._y_val

    @property
    def X_test(self):
        """Test split features, or `None` with no test split. Raises before setup."""
        self._assert_setup("X_test")
        return self._X_test

    @property
    def y_test(self):
        """Test split targets, or `None` with no test split. Raises before `setup()`."""
        self._assert_setup("y_test")
        return self._y_test

    @property
    def splits(self):
        """`(X_train, y_train, X_val, y_val, X_test, y_test)`."""
        self._assert_setup("splits")
        return (
            self._X_train,
            self._y_train,
            self._X_val,
            self._y_val,
            self._X_test,
            self._y_test,
        )

    @property
    def n_samples(self):
        """Total rows in `X`, before splitting."""
        return len(self.X_raw)

    @property
    def n_features(self):
        """Raw feature count, before any prescaling."""
        return self.X_raw.shape[1]

    @property
    def n_classes(self):
        """Number of unique values in `y`."""
        return len(np.unique(self.y_raw))

    @property
    def class_labels(self):
        """Unique values in `y`."""
        return np.unique(self.y_raw)

    @property
    def feature_dim(self):
        """Feature count after `setup()`; falls back to `n_features` before it."""
        return self._X_train.shape[1] if self._X_train is not None else self.n_features

    @property
    def normalizer(self) -> Optional["_Normalizer"]:
        """The fitted `_Normalizer`, or `None` before setup or with no `normalize`."""
        return self._normalizer

    def reconfigure(self, **kwargs) -> "DataModule":
        """Update settings and clear fitted state, requiring a re-`setup()`.

        Parameters
        ----------
        **kwargs
            Any of `normalize`, `split`, `stratify`, `seed`, `batch_size`,
            `num_workers`, `transform`.

        Returns
        -------
        DataModule
            `self`.
        """
        for k in kwargs:
            if k not in self._RECONFIGURABLE:
                raise ValueError(
                    f"reconfigure() does not accept {k!r}. "
                    f"Valid: {sorted(self._RECONFIGURABLE)}"
                )
        self._validate(
            kwargs.get("normalize", self.normalize), kwargs.get("split", self.split)
        )
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._is_setup = False
        self._normalizer = None
        self._numpy_transform = None
        self._torch_transform = None
        return self

    def __repr__(self):
        return (
            f"DataModule(name={self.name!r}, n={self.n_samples}, "
            f"features={self.n_features}, classes={self.n_classes}, "
            f"normalize={self.normalize!r}, "
            f"status={'ready' if self._is_setup else 'pending'})"
        )

    def __len__(self):
        return self.n_samples

    def _assert_setup(self, caller):
        if not self._is_setup:
            raise RuntimeError(
                f"'{caller}' called before setup().\n"
                f"Pass this DataModule to trainer.fit(model, dm) — "
                f"the Trainer calls setup() automatically.\n"
                f"Or call dm.setup() manually."
            )

    def _do_split(self):
        n = len(self.X_raw)
        train, val, test = self.split

        if self.stratify:
            i_tr, i_va, i_te = self._stratified_indices(n, val, test)
        else:
            idx = np.random.default_rng(self.seed).permutation(n)
            n_tr, n_va = int(n * train), int(n * val)
            if test <= 0:
                n_tr, n_va = (n_tr, n - n_tr) if val > 0 else (n, 0)
            i_tr, i_va, i_te = np.split(idx, [n_tr, n_tr + n_va])

        X, y = self.X_raw, self.y_raw
        Xs = [X[i_tr]] + [X[i] if len(i) else None for i in (i_va, i_te)]
        ys = [y[i_tr]] + [y[i] if len(i) else None for i in (i_va, i_te)]
        return Xs, ys

    def _stratified_indices(self, n, val, test):
        from sklearn.model_selection import train_test_split

        idx, held, empty = np.arange(n), val + test, np.array([], dtype=int)
        if held <= 0:
            return idx, empty, empty
        i_tr, i_held = train_test_split(
            idx, test_size=held, random_state=self.seed, stratify=self.y_raw
        )
        if val <= 0:
            return i_tr, empty, i_held
        if test <= 0:
            return i_tr, i_held, empty
        i_va, i_te = train_test_split(
            i_held,
            test_size=test / held,
            random_state=self.seed,
            stratify=self.y_raw[i_held],
        )
        return i_tr, i_va, i_te

    def clone_empty(self) -> "DataModule":
        """Return an unsetup shallow copy holding a copy of the fitted normalizer."""
        new_dm = copy.copy(self)
        new_dm._normalizer = copy.deepcopy(self._normalizer)
        new_dm.X_raw = new_dm.y_raw = None
        new_dm._X_train = new_dm._y_train = None
        new_dm._X_val = new_dm._y_val = None
        new_dm._X_test = new_dm._y_test = None
        new_dm._is_setup = False
        return new_dm

    def _map_features(self, fn) -> "DataModule":
        new_dm = self.clone_empty()
        new_dm._X_train, new_dm._X_val, new_dm._X_test = _map_present(
            fn, (self._X_train, self._X_val, self._X_test)
        )
        new_dm._y_train, new_dm._y_val, new_dm._y_test = (
            self._y_train,
            self._y_val,
            self._y_test,
        )
        new_dm._is_setup = True
        return new_dm


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
