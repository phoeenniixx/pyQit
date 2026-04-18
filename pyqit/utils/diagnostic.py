from dataclasses import dataclass
import logging

import numpy as np
from skbase.utils.dependencies import _check_soft_dependencies

from pyqit.core._loss_mapping import get_loss_fn
from pyqit.core.config import get_backend

logger = logging.getLogger("pyqit.diagnostics")


@dataclass
class BPResult:
    n_qubits: int
    n_samples: int
    layer_variances: dict[str, float]
    layer_ratios: dict[str, float]
    overall_variance: float
    expected_variance: float
    is_barren: bool
    quantum_variance: float
    classical_variance: float | None = None

    def __repr__(self) -> str:
        has_rich = _check_soft_dependencies("rich", severity="none")

        if has_rich:
            from rich.console import Console

            console = Console(force_terminal=False)
            with console.capture() as capture:
                console.print(self._build_rich_table())
            return "\n" + capture.get().rstrip()
        else:
            return self._build_ascii_table()

    def __rich__(self):
        return self._build_rich_table()

    def _build_rich_table(self):
        from rich.table import Table

        status_color = "red" if self.is_barren else "green"
        status_text = "BARREN PLATEAU" if self.is_barren else "HEALTHY"

        table = Table(
            title=f"BP Diagnostic Result : [{status_color} bold]{status_text}[/]",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Metric / Layer", style="bold")
        table.add_column("Value", justify="right")
        table.add_column("Status", justify="right")

        table.add_row("Qubits", str(self.n_qubits), "")
        table.add_row("Samples", str(self.n_samples), "")
        table.add_row(
            "Expected Variance", f"{self.expected_variance:.2e}", "[dim]Baseline[/]"
        )

        q_color = f"[{status_color}]"
        table.add_row(
            "Quantum Variance",
            f"{q_color}{self.quantum_variance:.2e}[/]",
            f"[{status_color} bold]{status_text}[/]",
        )

        if self.classical_variance is not None:
            table.add_row(
                "Classical Variance", f"{self.classical_variance:.2e}", "[dim]N/A[/]"
            )

        table.add_section()

        for k, r in self.layer_ratios.items():
            is_plateau = r < 1.0
            r_color = "[red]" if is_plateau else "[green]"
            tag = "[red bold]← plateau[/]" if is_plateau else "[green]Healthy[/]"
            table.add_row(f"Layer: {k}", f"{r_color}{r:.3f}x[/]", tag)

        return table

    def _build_ascii_table(self) -> str:
        status = "BARREN PLATEAU" if self.is_barren else "HEALTHY"

        lines = [
            "",
            "=" * 55,
            f" BP Diagnostic Result : {status}".center(55),
            "=" * 55,
            f"{'Qubits':<25} : {self.n_qubits}",
            f"{'Samples':<25} : {self.n_samples}",
            f"{'Expected Variance':<25} : {self.expected_variance:.2e} (Baseline)",
            f"{'Quantum Variance':<25} : {self.quantum_variance:.2e}",
        ]

        if self.classical_variance is not None:
            lines.append(f"{'Classical Variance':<25} : {self.classical_variance:.2e}")

        lines.append("-" * 55)
        lines.append(f"{'Layer':<30} | {'Ratio':<8} | {'Status'}")
        lines.append("-" * 55)

        for k, r in self.layer_ratios.items():
            tag = "← plateau" if r < 1.0 else "Healthy"
            layer_name = (k[:27] + "...") if len(k) > 30 else k
            lines.append(f"{layer_name:<30} | {r:<7.3f}x | {tag}")

        lines.append("=" * 55)
        return "\n".join(lines)


def check_barren_plateau(
    model,
    datamodule_or_X,
    y=None,
    num_samples: int = 200,
    loss_name: str = "mse",
    plot: bool = True,
) -> BPResult:
    backend = getattr(model, "backend", get_backend())

    X, y_target = _resolve_input(datamodule_or_X, y, model)

    quantum_keys, classical_keys = _split_weight_keys(model)
    all_keys = quantum_keys + classical_keys

    if not all_keys:
        raise ValueError("Model has no tracked weights to calculate gradients for.")

    n_qubits = getattr(model, "n_qubits", 1) or 1
    measured_wires = getattr(model, "_measure_wires", range(n_qubits))
    is_local_cost = len(measured_wires) < n_qubits
    if is_local_cost:
        raw_baseline = 1.0 / (2**n_qubits)
    else:
        raw_baseline = 1.0 / (3.0 * (4 ** (n_qubits - 1)))

    scale_factor = model.get_tag("bp_scale_factor", tag_value_default=1.0)

    expected_variance = raw_baseline * scale_factor

    logger.info(
        f"Running Barren Plateau Diagnostic | Model: {type(model).__name__} | "
        f"Backend: {backend} | Samples: {num_samples}"
    )
    if backend == "torch":
        layer_gradients = _sample_gradients_torch(
            model, X, y_target, all_keys, num_samples, loss_name
        )
    else:
        layer_gradients = _sample_gradients_pennylane(
            model, X, y_target, all_keys, num_samples, loss_name
        )
    layer_variances = {k: float(np.var(grads)) for k, grads in layer_gradients.items()}
    layer_ratios = {k: v / expected_variance for k, v in layer_variances.items()}

    quantum_vars = [layer_variances[k] for k in quantum_keys if k in layer_variances]
    classical_vars = [
        layer_variances[k] for k in classical_keys if k in layer_variances
    ]

    quantum_variance = float(np.mean(quantum_vars)) if quantum_vars else float("nan")
    classical_variance = float(np.mean(classical_vars)) if classical_vars else None

    is_barren = quantum_variance < expected_variance

    if is_barren:
        logger.warning(
            f"Severe Barren Plateau detected!"
            f"Quantum gradient variance ({quantum_variance:.2e}) "
            f"is below the theoretical random-circuit baseline"
            f" ({expected_variance:.2e})."
        )
    else:
        ratio = quantum_variance / expected_variance
        logger.info(
            f"Model looks healthy. Variance is {ratio:.1f}x above the random baseline."
        )

    result = BPResult(
        n_qubits=n_qubits,
        n_samples=num_samples,
        layer_variances=layer_variances,
        layer_ratios=layer_ratios,
        overall_variance=quantum_variance,
        expected_variance=expected_variance,
        is_barren=is_barren,
        quantum_variance=quantum_variance,
        classical_variance=classical_variance,
    )

    if plot:
        _plot(layer_gradients, quantum_keys, result)

    return result


def _sample_gradients_torch(model, X, y, weight_keys, num_samples, loss_name):
    """Safely mutates PyTorch nn.Parameters in place to evaluate gradients."""
    import torch

    loss_fn = get_loss_fn(loss_name, backend="torch")

    X_t = torch.as_tensor(np.asarray(X), dtype=torch.float64)[0:1]
    y_t = torch.as_tensor(np.asarray(y), dtype=torch.float64)

    original_state = {
        k: v.detach().clone() for k, v in model.weights.items() if k in weight_keys
    }
    layer_gradients = {k: [] for k in weight_keys}

    try:
        for _ in range(num_samples):
            with torch.no_grad():
                for k, param in model.weights.items():
                    if k in weight_keys:
                        param.copy_(torch.empty_like(param).uniform_(0, 2 * np.pi))

            if hasattr(model, "zero_grad"):
                model.zero_grad()

            preds = model.forward(X_t)
            if preds.ndim == 0:
                preds = preds.unsqueeze(0)

            loss = loss_fn(preds, y_t)
            loss.backward()

            for k in weight_keys:
                grad = model.weights[k].grad
                if grad is not None:
                    layer_gradients[k].extend(
                        grad.detach().cpu().numpy().flatten().tolist()
                    )

    finally:
        with torch.no_grad():
            for k, param in model.weights.items():
                if k in original_state:
                    param.copy_(original_state[k])

    return layer_gradients


def _sample_gradients_pennylane(model, X, y, weight_keys, num_samples, loss_name):
    """Passes kwargs directly to forward() to preserve the PennyLane Autograd graph."""
    import pennylane as qml
    import pennylane.numpy as pnp

    loss_fn = get_loss_fn(loss_name, backend="pennylane")
    X_p = pnp.array(X, requires_grad=False)
    y_p = pnp.array(y, requires_grad=False)

    layer_gradients = {k: [] for k in weight_keys}

    # Use **flat_kwargs routing to bypass state mutation
    def cost(*weight_tensors):
        flat_kwargs = dict(zip(weight_keys, weight_tensors))
        preds = model.forward(X_p, **flat_kwargs)
        if preds.ndim == 0:
            preds = pnp.expand_dims(preds, axis=0)
        return loss_fn(preds, y_p)

    grad_fn = qml.grad(cost)

    for _ in range(num_samples):
        rand_weights = [
            pnp.random.uniform(
                0, 2 * np.pi, size=model.weights[k].shape, requires_grad=True
            )
            for k in weight_keys
        ]

        grads = grad_fn(*rand_weights)
        for k, g in zip(weight_keys, grads):
            layer_gradients[k].extend(np.asarray(g).flatten().tolist())

    return layer_gradients


def _resolve_input(datamodule_or_X, y, model):
    from pyqit.data.datamodule import DataModule

    if isinstance(datamodule_or_X, DataModule):
        if not datamodule_or_X._is_setup:
            raise ValueError(
                "DataModule is not set up. You must call `dm.setup(stage='fit')` "
                "with the correct encoder before passing it to the diagnostic tool."
            )
        X_data = (
            datamodule_or_X.X_val
            if datamodule_or_X.X_val is not None
            else datamodule_or_X.X_train
        )
        y_data = (
            datamodule_or_X.y_val
            if datamodule_or_X.y_val is not None
            else datamodule_or_X.y_train
        )
        return np.asarray(X_data[:32]), np.asarray(y_data[:32])

    if y is None:
        raise ValueError("y must be provided when datamodule_or_X is a raw array.")
    return np.asarray(datamodule_or_X)[:32], np.asarray(y)[:32]


def _split_weight_keys(model):
    all_keys = list(model.weights.keys())
    if model.get_tag("object_type", "ansatz") != "hybrid":
        return all_keys, []

    q_keys = set(getattr(model, "weight_keys", all_keys))
    return [k for k in all_keys if k in q_keys], [
        k for k in all_keys if k not in q_keys
    ]


def _plot(layer_gradients, quantum_keys, result: BPResult):
    if not _check_soft_dependencies("matplotlib", severity="none"):
        logger.warning(
            "Matplotlib is not installed. Skipping barren plateau histogram plot."
        )
        return
    else:
        import matplotlib.pyplot as plt

    n_layers = len(layer_gradients)
    fig, axes = plt.subplots(n_layers, 1, figsize=(9, 3 * n_layers), squeeze=False)
    fig.suptitle(
        f"Gradient Landscape — {result.n_qubits} Qubits\n"
        f"Expected variance floor: {result.expected_variance:.2e}",
        fontsize=13,
        y=1.02,
    )

    for ax, (key, grads) in zip(axes[:, 0], layer_gradients.items()):
        ratio = result.layer_ratios[key]
        is_bp = (key in quantum_keys) and (ratio < 1.0)

        color = "#E74C3C" if is_bp else "#00CBA9"
        label = "quantum" if key in quantum_keys else "classical"

        ax.hist(
            grads,
            bins=min(50, max(10, len(grads) // 10)),
            color=color,
            alpha=0.75,
            edgecolor="black",
        )
        ax.axvline(0, color="black", linestyle="--", alpha=0.6)

        status = (
            "PLATEAU" if is_bp else ("Healthy" if key in quantum_keys else "Classical")
        )
        ax.set_title(
            f"{key} [{label}] — var={result.layer_variances[key]:.2e} \
                ({ratio:.2f}x) [{status}]",
            fontsize=10,
        )
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.show()
