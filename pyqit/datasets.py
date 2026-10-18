"""Datasets."""

import numpy as np
from sklearn import datasets as _sk
from sklearn.utils import check_random_state

__all__ = [
    "load_iris",
    "load_wine",
    "make_ad_hoc",
    "make_bars_and_stripes",
    "make_circles",
    "make_hidden_manifold",
    "make_hyperplanes_parity",
    "make_linearly_separable",
    "make_moons",
    "make_sine",
    "make_two_curves",
]


def _standardize(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)


def make_linearly_separable(n_samples=300, n_features=4, margin=None, seed=None):
    """Generate points in the hypercube ``[-1, 1]^d`` that a hyperplane separates.

    Rows are drawn uniformly and labelled by the sign of ``sum(x)``. Any row
    closer than ``margin`` to the hyperplane ``sum(x) = 0`` is redrawn.

    Parameters
    ----------
    n_samples : int, default 300
    n_features : int, default 4
    margin : float, optional
        Half-width of the empty band around the hyperplane. Defaults to
        ``0.02 * n_features``.
    seed : int or numpy RandomState, optional
        ``None`` uses numpy's global state, which ``pyqit.set_seed`` seeds.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
        ``1`` where ``sum(x) > 0``, else ``0``.

    References
    ----------
    [1] Bowles, Ahmed, Schuld, "Better than classical? The subtle art of
    benchmarking quantum machine learning models", arXiv:2403.07059 (2024).

    [2] ``qml_benchmarks.data.generate_linearly_separable`` in
    https://github.com/XanaduAI/qml-benchmarks (Apache License 2.0), the
    reference implementation this follows.

    Examples
    --------
    >>> from pyqit.datasets import make_linearly_separable
    >>> X, y = make_linearly_separable(n_samples=100, n_features=4, seed=0)
    >>> X.shape, sorted(set(y.tolist()))
    ((100, 4), [0, 1])
    """
    rng = check_random_state(seed)
    margin = 0.02 * n_features if margin is None else margin
    rows = []
    while sum(len(r) for r in rows) < n_samples:
        X = rng.uniform(-1, 1, size=(2 * n_samples, n_features))
        rows.append(X[np.abs(X.sum(axis=1)) > margin])
    X = np.concatenate(rows)[:n_samples]
    return X, (X.sum(axis=1) > 0).astype(int)


def make_two_curves(
    n_samples=300, n_features=4, degree=5, offset=0.1, noise=0.01, seed=None
):
    """Generate two classes that lie on the same random curve, one shifted.

    A one-dimensional curve is embedded in ``n_features`` dimensions by giving
    every feature its own random Fourier series. Both classes sample that curve and
    the second is shifted by ``offset`` along every axis.

    Parameters
    ----------
    n_samples : int, default 300
        Rounded down to an even number so the classes balance.
    n_features : int, default 4
    degree : int, default 5
        Highest frequency in the Fourier series.
    offset : float, default 0.1
        Distance between the two curves before standardisation.
    noise : float, default 0.01
        Standard deviation of the Gaussian noise added to every feature.
    seed : int or numpy RandomState, optional
        ``None`` uses numpy's global state, which ``pyqit.set_seed`` seeds.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Standardised to zero mean and unit variance.
    y : ndarray of shape (n_samples,)
        ``0`` for the base curve, ``1`` for the shifted one.

    References
    ----------
    [1] Bowles, Ahmed, Schuld, "Better than classical? The subtle art of
    benchmarking quantum machine learning models", arXiv:2403.07059 (2024).

    [2] ``qml_benchmarks.data.generate_two_curves`` in
    https://github.com/XanaduAI/qml-benchmarks (Apache License 2.0), the
    reference implementation this follows.

    Examples
    --------
    >>> from pyqit.datasets import make_two_curves
    >>> X, y = make_two_curves(n_samples=100, n_features=4, seed=0)
    >>> X.shape, int(y.sum())
    ((100, 4), 50)
    """
    rng = check_random_state(seed)
    half = n_samples // 2
    coeffs = rng.uniform(size=(n_features, degree + 1, 2))
    coeffs /= np.linalg.norm(coeffs)
    freqs = np.pi * np.arange(1, degree + 1)

    def curve(t):
        angles = np.outer(t, freqs)
        X = coeffs[:, 0, :].sum(axis=1)
        X = X + np.cos(angles) @ coeffs[:, 1:, 0].T
        X = X + np.sin(angles) @ coeffs[:, 1:, 1].T
        return X + rng.normal(scale=noise, size=X.shape)

    X = np.r_[curve(rng.uniform(size=half)), curve(rng.uniform(size=half)) + offset]
    y = np.r_[np.zeros(half, dtype=int), np.ones(half, dtype=int)]
    return _standardize(X), y


def make_hidden_manifold(n_samples=300, n_features=4, manifold_dimension=6, seed=None):
    """Generate points on a hidden low-dimensional manifold with network labels.

    Gaussian points in ``manifold_dimension`` dimensions are mapped into
    ``n_features`` dimensions through a random linear map and a ``tanh``. A
    random one-hidden-layer network on the hidden coordinates gives the labels.

    Parameters
    ----------
    n_samples : int, default 300
    n_features : int, default 4
    manifold_dimension : int, default 6
    seed : int or numpy RandomState, optional
        ``None`` uses numpy's global state, which ``pyqit.set_seed`` seeds.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Values in ``(-1, 1)``.
    y : ndarray of shape (n_samples,)
        Balanced by splitting at the median of the network output.

    References
    ----------
    [1] Bowles, Ahmed, Schuld, "Better than classical? The subtle art of
    benchmarking quantum machine learning models", arXiv:2403.07059 (2024).

    [2] ``qml_benchmarks.data.generate_hidden_manifold_model`` in
    https://github.com/XanaduAI/qml-benchmarks (Apache License 2.0), the
    reference implementation this follows.

    [3] Goldt, Mézard, Krzakala, Zdeborová, "Modeling the influence of data
    structure on learning in neural networks: The hidden manifold model",
    Phys. Rev. X 10, 041044 (2020).

    Examples
    --------
    >>> from pyqit.datasets import make_hidden_manifold
    >>> X, y = make_hidden_manifold(n_samples=100, n_features=4, seed=0)
    >>> X.shape, int(y.sum())
    ((100, 4), 50)
    """
    rng = check_random_state(seed)
    F = rng.normal(size=(manifold_dimension, n_features))
    C = rng.normal(size=(n_samples, manifold_dimension))
    biases = rng.uniform(-1, 1, size=n_features)
    X = np.tanh(C @ F / np.sqrt(manifold_dimension) - biases)
    W = rng.normal(size=(manifold_dimension, manifold_dimension))
    v = rng.normal(size=manifold_dimension)
    score = np.tanh(C @ W.T) @ v / np.sqrt(manifold_dimension)
    return X, (score > np.median(score)).astype(int)


def make_hyperplanes_parity(
    n_samples=300, n_features=4, n_hyperplanes=2, dim_hyperplanes=3, seed=None
):
    """Generate points labelled by the parity of the hyperplanes they lie above.

    Gaussian points in ``dim_hyperplanes`` dimensions are labelled by whether an
    even number of ``n_hyperplanes`` random hyperplanes have them on the positive
    side, then embedded in ``n_features`` dimensions by a random linear map.

    Parameters
    ----------
    n_samples : int, default 300
        Rounded down to an even number so the classes balance.
    n_features : int, default 4
    n_hyperplanes : int, default 2
    dim_hyperplanes : int, default 3
        Dimension the hyperplanes live in before the linear embedding.
    seed : int or numpy RandomState, optional
        ``None`` uses numpy's global state, which ``pyqit.set_seed`` seeds.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Standardised to zero mean and unit variance.
    y : ndarray of shape (n_samples,)
        ``1`` for an even count, ``0`` for odd.

    References
    ----------
    [1] Bowles, Ahmed, Schuld, "Better than classical? The subtle art of
    benchmarking quantum machine learning models", arXiv:2403.07059 (2024).

    [2] ``qml_benchmarks.data.generate_hyperplanes_parity`` in
    https://github.com/XanaduAI/qml-benchmarks (Apache License 2.0), the
    reference implementation this follows.

    Examples
    --------
    >>> from pyqit.datasets import make_hyperplanes_parity
    >>> X, y = make_hyperplanes_parity(n_samples=100, n_features=4, seed=0)
    >>> X.shape, int(y.sum())
    ((100, 4), 50)
    """
    rng = check_random_state(seed)
    half = n_samples // 2
    W = rng.uniform(size=(n_hyperplanes, dim_hyperplanes))
    b = rng.uniform(size=n_hyperplanes)
    pos, neg = np.empty((0, dim_hyperplanes)), np.empty((0, dim_hyperplanes))
    while min(len(pos), len(neg)) < half:
        Z = rng.normal(size=(4 * n_samples, dim_hyperplanes))
        even = (Z @ W.T + b > 0).sum(axis=1) % 2 == 0
        pos, neg = np.r_[pos, Z[even]], np.r_[neg, Z[~even]]
    Z = np.r_[pos[:half], neg[:half]]
    y = np.r_[np.ones(half, dtype=int), np.zeros(half, dtype=int)]
    X = Z @ rng.uniform(size=(dim_hyperplanes, n_features))
    return _standardize(X), y


def make_bars_and_stripes(
    n_samples=300, height=4, width=4, noise=0.5, flatten=True, seed=None
):
    """Generate noisy images of horizontal stripes or vertical bars.

    Each image starts at ``-1``. A random subset of its rows (a stripe image)
    or of its columns (a bar image) is set to ``1``, then Gaussian noise is
    added to every pixel.

    Parameters
    ----------
    n_samples : int, default 300
    height, width : int, default 4
    noise : float, default 0.5
        Standard deviation of the Gaussian pixel noise.
    flatten : bool, default True
        Return one row of ``height * width`` pixels per image, the shape
        ``DataModule`` takes. ``False`` keeps the images as
        ``(n_samples, 1, height, width)`` with a channel axis.
    seed : int or numpy RandomState, optional
        ``None`` uses numpy's global state, which ``pyqit.set_seed`` seeds.

    Returns
    -------
    X : ndarray of shape (n_samples, height * width) or (n_samples, 1, height, width)
    y : ndarray of shape (n_samples,)
        ``1`` for stripes, ``0`` for bars.

    References
    ----------
    [1] Bowles, Ahmed, Schuld, "Better than classical? The subtle art of
    benchmarking quantum machine learning models", arXiv:2403.07059 (2024).

    [2] ``qml_benchmarks.data.generate_bars_and_stripes`` in
    https://github.com/XanaduAI/qml-benchmarks (Apache License 2.0), the
    reference implementation this follows.

    Examples
    --------
    >>> from pyqit.datasets import make_bars_and_stripes
    >>> X, y = make_bars_and_stripes(n_samples=100, height=4, width=4, seed=0)
    >>> X.shape, sorted(set(y.tolist()))
    ((100, 16), [0, 1])
    """
    rng = check_random_state(seed)
    stripes = rng.uniform(size=n_samples) > 0.5
    rows = rng.uniform(size=(n_samples, height, 1)) > 0.5
    cols = rng.uniform(size=(n_samples, 1, width)) > 0.5
    lit = np.where(stripes[:, None, None], rows, cols)
    X = np.where(np.broadcast_to(lit, (n_samples, height, width)), 1.0, -1.0)
    X = X + rng.normal(scale=noise, size=X.shape)
    X = X.reshape(n_samples, -1) if flatten else X[:, None]
    return X, stripes.astype(int)


def make_ad_hoc(n_samples=100, n_features=2, gap=0.3, seed=None):
    """Generate the ad hoc set that the ZZ feature map separates exactly.

    Inputs are drawn uniformly in ``[0, 2 pi)^n``. Each is mapped to the state
    ``U(x) H U(x) H |0>`` of the second-order Pauli-Z feature map with full
    entanglement, and labelled by the sign of ``<V^dag Z^n V>`` in that state for
    one random unitary ``V`` drawn per call. Inputs whose expectation lies within
    ``gap`` of zero are rejected, so the classes are separated by a margin.

    Parameters
    ----------
    n_samples : int, default 100
    n_features : int, default 2
        One feature per qubit. The state has ``2 ** n_features`` amplitudes, so
        keep this small.
    gap : float, default 0.3
        Rejection band around zero expectation.
    seed : int or numpy RandomState, optional
        ``None`` uses numpy's global state, which ``pyqit.set_seed`` seeds.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
        ``1`` where the expectation is positive, else ``0``.

    References
    ----------
    [1] Havlicek, Corcoles, Temme, Harrow, Kandala, Chow, Gambetta, "Supervised
    learning with quantum-enhanced feature spaces", Nature 567, 209 (2019).

    [2] ``qiskit_machine_learning.datasets.ad_hoc_data`` in
    https://github.com/qiskit-community/qiskit-machine-learning (Apache
    License 2.0), the reference implementation this follows.

    Examples
    --------
    >>> from pyqit.datasets import make_ad_hoc
    >>> X, y = make_ad_hoc(n_samples=50, n_features=2, seed=0)
    >>> X.shape, sorted(set(y.tolist()))
    ((50, 2), [0, 1])
    """
    rng = check_random_state(seed)
    dim = 2**n_features
    bits = (np.arange(dim)[:, None] >> np.arange(n_features)[::-1]) & 1
    z = 1.0 - 2.0 * bits
    parity = z.prod(axis=1)
    hadamard = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2)
    H = np.array([[1.0]])
    for _ in range(n_features):
        H = np.kron(H, hadamard)
    Q, R = np.linalg.qr(rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim)))
    V = Q * (np.diag(R) / np.abs(np.diag(R)))
    O = V.conj().T @ (parity[:, None] * V)

    def expectation(X):
        P = np.pi - X
        phase = X @ z.T + 0.5 * ((P @ z.T) ** 2 - (P**2).sum(axis=1)[:, None])
        U = np.exp(-1j * phase)
        psi = U * ((U / np.sqrt(dim)) @ H)
        return np.real(np.einsum("ij,ij->i", psi.conj(), psi @ O.T))

    rows, labels = [], []
    while sum(len(r) for r in rows) < n_samples:
        X = rng.uniform(0, 2 * np.pi, size=(2 * n_samples, n_features))
        value = expectation(X)
        keep = np.abs(value) > gap
        rows.append(X[keep])
        labels.append(value[keep] > 0)
    X = np.concatenate(rows)[:n_samples]
    return X, np.concatenate(labels)[:n_samples].astype(int)


def make_sine(n_samples=80, noise=0.0, seed=None):
    """Generate a sine curve for regression, ``y = sin(pi x)`` on ``[-1, 1]``.

    The inputs are an even grid, so the curve is sampled the same way every
    time; only the optional noise is random.

    Parameters
    ----------
    n_samples : int, default 80
    noise : float, default 0.0
        Standard deviation of Gaussian noise added to ``y``.
    seed : int or numpy RandomState, optional
        Only the noise is random. ``None`` uses numpy's global state.

    Returns
    -------
    X : ndarray of shape (n_samples, 1)
    y : ndarray of shape (n_samples,)

    Examples
    --------
    >>> from pyqit.datasets import make_sine
    >>> X, y = make_sine(n_samples=80)
    >>> X.shape, y.shape
    ((80, 1), (80,))
    """
    X = np.linspace(-1, 1, n_samples).reshape(-1, 1)
    y = np.sin(np.pi * X).ravel()
    if noise:
        y = y + check_random_state(seed).normal(scale=noise, size=y.shape)
    return X, y


def make_moons(n_samples=200, noise=0.1, seed=None):
    """Generate two interleaving half circles, one per class.

    Parameters
    ----------
    n_samples : int, default 200
    noise : float, default 0.1
    seed : int or numpy RandomState, optional

    Returns
    -------
    X : ndarray of shape (n_samples, 2)
    y : ndarray of shape (n_samples,)

    References
    ----------
    [1] ``sklearn.datasets.make_moons``.
    """
    return _sk.make_moons(n_samples=n_samples, noise=noise, random_state=seed)


def make_circles(n_samples=200, noise=0.1, factor=0.5, seed=None):
    """Generate a small circle inside a large one, one per class.

    Parameters
    ----------
    n_samples : int, default 200
    noise : float, default 0.1
    factor : float, default 0.5
        Radius of the inner circle relative to the outer.
    seed : int or numpy RandomState, optional

    Returns
    -------
    X : ndarray of shape (n_samples, 2)
    y : ndarray of shape (n_samples,)

    References
    ----------
    [1] ``sklearn.datasets.make_circles``.
    """
    return _sk.make_circles(
        n_samples=n_samples, noise=noise, factor=factor, random_state=seed
    )


def load_iris():
    """Load the iris flowers, 150 rows of 4 measurements across 3 species.

    Returns
    -------
    X : ndarray of shape (150, 4)
    y : ndarray of shape (150,)
        Class indices ``0``, ``1``, ``2``.

    References
    ----------
    [1] Fisher, "The use of multiple measurements in taxonomic problems",
    Annals of Eugenics 7, 179 (1936).

    [2] ``sklearn.datasets.load_iris``, which this reads from.
    """
    return _sk.load_iris(return_X_y=True)


def load_wine():
    """Load the wine chemistry set, 178 rows of 13 measurements across 3 growers.

    Returns
    -------
    X : ndarray of shape (178, 13)
    y : ndarray of shape (178,)
        Class indices ``0``, ``1``, ``2``.

    References
    ----------
    [1] Forina et al., "PARVUS: An extendible package for data exploration,
    classification and correlation", UCI Machine Learning Repository.

    [2] ``sklearn.datasets.load_wine``, which this reads from.
    """
    return _sk.load_wine(return_X_y=True)
