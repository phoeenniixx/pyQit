from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd


def _is_torch_transform(fn) -> bool:
    if fn is None:
        return False

    fn_type = type(fn)

    mro = getattr(fn_type, "__mro__", (fn_type,))

    return any(
        (getattr(cls, "__module__", "") or "").startswith(("torch", "torchvision"))
        for cls in mro
    )


class _NumpyLoader:
    def __init__(self, X, y, batch_size, shuffle, seed=None, transform=None):
        self.X, self.y = X, y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.transform = transform

    def __iter__(self):
        idx = np.arange(len(self.X))
        if self.shuffle:
            idx = np.random.default_rng(self.seed).permutation(idx)
        for s in range(0, len(self.X), self.batch_size):
            b = idx[s : s + self.batch_size]
            Xb = self.X[b]
            if self.transform is not None:
                Xb = np.stack([self.transform(Xb[i]) for i in range(len(Xb))])
            yield Xb, self.y[b]

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __repr__(self):
        t = f", transform={self.transform!r}" if self.transform else ""
        return f"_NumpyLoader(n={len(self.X)}, \
            batch_size={self.batch_size}, shuffle={self.shuffle}{t})"


def _make_torch_loader(
    X,
    y,
    batch_size,
    shuffle,
    num_workers=0,
    numpy_transform=None,
    tensor_transform=None,
):
    import torch
    from torch.utils.data import DataLoader, Dataset

    _tt = tensor_transform

    class _DS(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.long)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, i):
            x = self.X[i]
            if _tt is not None:
                x = _tt(x)
            return x, self.y[i]

    return DataLoader(
        _DS(X, y),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


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
                f"'{self.method}' normalizer must be fitted before transform()."
            )
        if self.method == "minmax":
            return (X - self._params["lo"]) / self._params["rng"]
        if self.method == "zscore":
            return (X - self._params["mu"]) / self._params["sigma"]
        if self.method == "l2":
            n = np.linalg.norm(X, axis=1, keepdims=True)
            return X / np.where(n == 0, 1.0, n)
        if self.method == "l1":
            n = np.abs(X).sum(axis=1, keepdims=True)
            return X / np.where(n == 0, 1.0, n)
        raise ValueError(self.method)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    @property
    def params(self) -> dict:
        return self._params.copy()

    def __repr__(self):
        return f"_Normalizer(method={self.method!r}, fitted={self._fitted})"


def _prescale_angle_pi(X, n_qubits):
    n_f = X.shape[1]
    X = (
        np.hstack([X, np.zeros((len(X), n_qubits - n_f))])
        if n_f < n_qubits
        else X[:, :n_qubits]
    )
    lo = X.min(axis=0)
    hi = X.max(axis=0)
    return (X - lo) / np.where(hi - lo == 0, 1.0, hi - lo) * np.pi


def _prescale_amplitude(X, n_qubits):
    target = 2**n_qubits
    n_f = X.shape[1]
    X = (
        np.hstack([X, np.zeros((len(X), target - n_f))])
        if n_f < target
        else X[:, :target]
    )
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(norms == 0, 1.0, norms)


def _prescale_binary(X, n_qubits):
    X = (X >= 0.5).astype(np.float64)
    n_f = X.shape[1]
    return (
        np.hstack([X, np.zeros((len(X), n_qubits - n_f))])
        if n_f < n_qubits
        else X[:, :n_qubits]
    )


_PRESCALE_FNS = {
    "angle_pi": _prescale_angle_pi,
    "amplitude": _prescale_amplitude,
    "binary": _prescale_binary,
    "none": None,
}


def _apply_prescale(X, prescale: str, n_qubits: int):
    fn = _PRESCALE_FNS.get(prescale)
    if fn is None:
        if prescale != "none":
            raise ValueError(
                f"Unknown PRESCALE {prescale!r}. "
                f"Valid: {list(_PRESCALE_FNS)}. "
                f"Check your embedding class's PRESCALE attribute."
            )
        return X
    return fn(X, n_qubits)


class DataModule:
    _VALID_NORMALIZE = ("minmax", "zscore", "l2", "l1")

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
    ):
        if normalize is not None and normalize not in self._VALID_NORMALIZE:
            raise ValueError(
                f"normalize={normalize!r} not supported.\n"
                f"Numpy normalizers: {self._VALID_NORMALIZE}\n"
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
                f"got {train+val+test:.6f} ({train}, {val}, {test})."
            )
        if train <= 0 and val <= 0 and test <= 0:
            raise ValueError("At least one fraction must be > 0.")

        self.X_raw = np.asarray(X, dtype=np.float64)
        self.y_raw = np.asarray(y, dtype=np.float64)
        if self.X_raw.ndim == 1:
            self.X_raw = self.X_raw.reshape(-1, 1)

        self.name = name
        self.normalize = normalize
        self.split = (train, val, test)
        self.stratify = stratify
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform

        self._backend = "pennylane"
        self._normalizer = None
        self._numpy_transform = None
        self._torch_transform = None
        self._X_train = self._y_train = None
        self._X_val = self._y_val = None
        self._X_test = self._y_test = None
        self._is_setup = False

    @classmethod
    def from_numpy(cls, X, y, **kw):
        return cls(X, y, **kw)

    @classmethod
    def from_sklearn(cls, loader: Callable, **kw) -> "DataModule":
        bunch = loader()
        name = kw.pop("name", loader.__name__.replace("load_", ""))
        return cls(bunch.data, bunch.target.astype(np.float64), name=name, **kw)

    @classmethod
    def from_dataframe(cls, df: Any, label_col: str | int = -1, **kw) -> "DataModule":
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
        df = pd.read_csv(path, sep=delimiter)
        name = kw.pop("name", Path(path).stem)
        return cls.from_dataframe(df, label_col=label_col, name=name, **kw)

    def setup(
        self,
        stage: str | None = None,
        backend: str | None = None,
        batch_size: int | None = None,
        n_qubits: int | None = None,
        encoder: type | None = None,
        force: bool = False,
    ) -> "DataModule":
        if self._is_setup and not force:
            return self

        self._backend = backend
        self.batch_size = batch_size

        self.encoder = encoder
        self.n_qubits = n_qubits

        active_encoder = encoder or self.encoder

        prescale = active_encoder.PRESCALE if active_encoder is not None else None
        nq = n_qubits or self.n_qubits

        numpy_fns, torch_fns = [], []
        fns = (
            self.transform
            if isinstance(self.transform, list)
            else ([self.transform] if self.transform is not None else [])
        )
        for fn in fns:
            if _is_torch_transform(fn):
                torch_fns.append(fn)
            else:
                numpy_fns.append(fn)

        if torch_fns and self._backend == "pennylane":
            names = [type(f).__name__ for f in torch_fns]
            raise RuntimeError(
                f"Torch transform(s) {names} cannot be used with backend='pennylane'."
            )

        self._numpy_transform = _compose(numpy_fns) if numpy_fns else None
        self._torch_transform = _compose(torch_fns) if torch_fns else None

        if stage == "predict":
            X_tr = y_tr = X_va = y_va = None
            X_te, y_te = self.X_raw, self.y_raw
        else:
            X_tr, y_tr, X_va, y_va, X_te, y_te = self._do_split()

        if self.normalize is not None and X_tr is not None:
            norm = _Normalizer(self.normalize)
            X_tr = norm.fit_transform(X_tr)
            X_va = norm.transform(X_va) if X_va is not None else None
            X_te = norm.transform(X_te) if X_te is not None else None
            self._normalizer = norm
        elif self.normalize is not None and stage == "predict":
            if self._normalizer is not None:
                X_te = self._normalizer.transform(X_te)

        prescale = self.encoder.PRESCALE if self.encoder is not None else None
        if prescale is not None:
            if nq is None:
                raise RuntimeError("Prescaling requires n_qubits.")
            X_tr = _apply_prescale(X_tr, prescale, nq) if X_tr is not None else None
            X_va = _apply_prescale(X_va, prescale, nq) if X_va is not None else None
            X_te = _apply_prescale(X_te, prescale, nq) if X_te is not None else None

        if self._numpy_transform is not None and self._backend == "torch":
            if X_tr is not None:
                X_tr = np.stack(
                    [self._numpy_transform(X_tr[i]) for i in range(len(X_tr))]
                )
            if X_va is not None:
                X_va = np.stack(
                    [self._numpy_transform(X_va[i]) for i in range(len(X_va))]
                )
            if X_te is not None:
                X_te = np.stack(
                    [self._numpy_transform(X_te[i]) for i in range(len(X_te))]
                )
            self._numpy_transform_applied_in_setup = True
        else:
            self._numpy_transform_applied_in_setup = False

        self._X_train, self._y_train = X_tr, y_tr
        self._X_val, self._y_val = X_va, y_va
        self._X_test, self._y_test = X_te, y_te
        self._is_setup = True
        return self

    def train_loader(self, shuffle: bool = True):
        self._assert_setup("train_loader")
        return self._make_loader(self._X_train, self._y_train, shuffle)

    def val_loader(self, shuffle: bool = False):
        self._assert_setup("val_loader")
        return (
            self._make_loader(self._X_val, self._y_val, shuffle)
            if self._X_val is not None
            else None
        )

    def test_loader(self, shuffle: bool = False):
        self._assert_setup("test_loader")
        return (
            self._make_loader(self._X_test, self._y_test, shuffle)
            if self._X_test is not None
            else None
        )

    def _make_loader(self, X, y, shuffle):
        if self._backend == "torch":
            return _make_torch_loader(
                X,
                y,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                tensor_transform=self._torch_transform,
            )
        numpy_t = (
            None if self._numpy_transform_applied_in_setup else self._numpy_transform
        )
        return _NumpyLoader(X, y, self.batch_size, shuffle, self.seed, numpy_t)

    @property
    def X_train(self):
        self._assert_setup("X_train")
        return self._X_train

    @property
    def y_train(self):
        self._assert_setup("y_train")
        return self._y_train

    @property
    def X_val(self):
        self._assert_setup("X_val")
        return self._X_val

    @property
    def y_val(self):
        self._assert_setup("y_val")
        return self._y_val

    @property
    def X_test(self):
        self._assert_setup("X_test")
        return self._X_test

    @property
    def y_test(self):
        self._assert_setup("y_test")
        return self._y_test

    @property
    def splits(self):
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
        return len(self.X_raw)

    @property
    def n_features(self):
        return self.X_raw.shape[1]

    @property
    def n_classes(self):
        return len(np.unique(self.y_raw))

    @property
    def class_labels(self):
        return np.unique(self.y_raw)

    @property
    def feature_dim(self):
        return self._X_train.shape[1] if self._X_train is not None else self.n_features

    @property
    def normalizer(self) -> Optional["_Normalizer"]:
        return self._normalizer

    def reconfigure(self, **kwargs) -> "DataModule":
        _ok = {
            "normalize",
            "split",
            "stratify",
            "seed",
            "batch_size",
            "num_workers",
            "transform",
        }
        for k, v in kwargs.items():
            if k not in _ok:
                raise ValueError(
                    f"reconfigure() does not accept {k!r}. Valid: {sorted(_ok)}"
                )
            setattr(self, k, v)
        self._is_setup = False
        self._normalizer = None
        self._numpy_transform = None
        self._torch_transform = None
        return self

    def summary(self):
        # TODO: implement a summary method that prints
        # out the configuration and stats of the DataModule
        pass

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
        X, y = self.X_raw, self.y_raw
        n = len(X)
        train_frac, val_frac, test_frac = self.split

        if self.stratify:
            from sklearn.model_selection import train_test_split as tts

            if val_frac > 0 and test_frac > 0:
                X_tr, X_tmp, y_tr, y_tmp = tts(
                    X,
                    y,
                    test_size=val_frac + test_frac,
                    random_state=self.seed,
                    stratify=y,
                )
                X_va, X_te, y_va, y_te = tts(
                    X_tmp,
                    y_tmp,
                    test_size=test_frac / (val_frac + test_frac),
                    random_state=self.seed,
                    stratify=y_tmp,
                )
            elif val_frac > 0:
                X_tr, X_va, y_tr, y_va = tts(
                    X, y, test_size=val_frac, random_state=self.seed, stratify=y
                )
                X_te = y_te = None
            elif test_frac > 0:
                X_tr, X_te, y_tr, y_te = tts(
                    X, y, test_size=test_frac, random_state=self.seed, stratify=y
                )
                X_va = y_va = None
            else:
                X_tr, y_tr = X, y
                X_va = y_va = X_te = y_te = None
            return X_tr, y_tr, X_va, y_va, X_te, y_te

        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(n)
        n_tr = int(n * train_frac)
        n_va = int(n * val_frac)
        i_tr = idx[:n_tr]
        i_va = idx[n_tr : n_tr + n_va] if val_frac > 0 else np.array([], int)
        i_te = idx[n_tr + n_va :] if test_frac > 0 else np.array([], int)
        return (
            X[i_tr],
            y[i_tr],
            X[i_va] if len(i_va) else None,
            y[i_va] if len(i_va) else None,
            X[i_te] if len(i_te) else None,
            y[i_te] if len(i_te) else None,
        )

    def clone_empty(self) -> "DataModule":
        new_dm = object.__new__(DataModule)

        new_dm.__dict__.update(self.__dict__)

        new_dm.X_raw = None
        new_dm.y_raw = None

        new_dm._is_setup = False
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
