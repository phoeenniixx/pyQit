# PyQit

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status: Active Development](https://img.shields.io/badge/status-active_development-orange.svg)]()
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![Build Status](https://img.shields.io/github/actions/workflow/status/phoeenniixx/pyQit/test.yml)](https://github.com/phoeenniixx/pyqit/actions)

> **A high-level quantum machine learning framework built on PennyLane.**
> It provides a Trainer/DataModule/Model layer over PennyLane QNodes, so training a
> variational circuit does not mean hand-rolling an optimizer loop.

**Version `0.1.0b1`. The API is unstable and still changing.**

### Key Features

* **Lightweight & Modular:** PyQit runs natively on **PennyLane** and **NumPy**.
    > **PyTorch and PyTorch Lightning are strictly optional soft dependencies.** If you don't need deep learning hybrid models or GPU orchestration, you don't have to install them.
* **Backend Agnostic:** Switch between native `pennylane` (pure Autograd) and `torch` (Lightning engine) with one call. Same model, same callbacks, same history.
* **Automated Diagnostics:** A "Pre-Flight Check" runs Monte Carlo gradient sampling to detect Barren Plateaus mathematically *before* you spend the compute.
* **Data Orchestration:** A lazy `DataModule` keeps stateful classical normalization (`minmax`, `zscore`, `l2`, `l1`, fit on train only) separate from stateless quantum prescaling, which the model's embedding drives (`Angle`, `Amplitude`, `IQP`).
* **Portable Callbacks:** `EarlyStopping` and `ModelCheckpoint` are written once and honoured by both backends.

## Installation

Not on PyPI yet. Install from source:

```bash
git clone https://github.com/phoeenniixx/pyqit.git
cd pyqit

pip install -e "."                # base: pennylane + numpy, no torch
pip install -e ".[pytorch]"       # + torch and lightning
pip install -e ".[all_extras]"    # + matplotlib and rich as well
pip install -e ".[dev]"           # contributors: pytest, ruff, pre-commit, sphinx
```

Quote the extras. `zsh` treats bare brackets as a glob.

## Quickstart

```python
from sklearn.datasets import make_moons

import pyqit
from pyqit.ansatzes import SELAnsatz
from pyqit.core import AngleEmbedding
from pyqit.models import VQCClassifier

pyqit.set_seed(42)

X, y = make_moons(n_samples=200, noise=0.1, random_state=0)

# Nothing is split, normalized or prescaled until the Trainer calls setup().
dm = pyqit.DataModule(X, y, normalize="minmax", batch_size=16)

model = VQCClassifier(
    n_qubits=4,
    n_layers=3,
    ansatz=SELAnsatz,
    encoder=AngleEmbedding,
)

trainer = pyqit.Trainer(max_epochs=30, learning_rate=0.05)
history = trainer.fit(model, dm)

print(history.best_epoch, history.best_score)   # 10 0.0936
preds = trainer.predict(model, dm)              # runs on the test split
```

`fit` prints a model table and a progress bar, then returns a `TrainingHistory` holding
`train_loss`, `val_loss`, `train_acc`, `val_acc` and `epoch_times`, one entry per epoch.

Two things that bite people, both because they are read at construction time:

- Seed **before** you build the model. Weights are drawn in `__init__`, so
  `Trainer(seed=...)` alone covers training and diagnostics but not initialisation.
- Set the backend **before** you build the model, for the same reason.

## Switching backends

```python
import pyqit

pyqit.set_backend("torch")     # raises ImportError if torch is not installed
model = VQCClassifier(n_qubits=4, n_layers=2)

trainer = pyqit.Trainer(max_epochs=30)
history = trainer.fit(model, dm)
```

That is the whole change. The QNode is wrapped in a `qml.qnn.TorchLayer` and training goes
through Lightning. Anything Lightning-specific rides along in `backend_kwargs`:

```python
pyqit.Trainer(
    max_epochs=50,
    backend_kwargs={"accelerator": "gpu", "devices": 1, "gradient_clip_val": 0.5},
)
```

Lightning's own constructor is not mirrored onto `Trainer`. It carries roughly forty
parameters, most of which a PennyLane optimizer loop cannot honour. The accelerator
defaults to `"cpu"`, so a GPU is used only when you ask for one.

## Callbacks and checkpointing

```python
from pyqit.core import EarlyStopping, ModelCheckpoint

trainer = pyqit.Trainer(
    max_epochs=100,
    loss_fn="cross_entropy",
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=3),
        ModelCheckpoint(dirpath="ckpts", save_best=True, save_last=True),
    ],
)
history = trainer.fit(model, dm)
```

```
[EarlyStopping] Stopped at epoch 18 -- val_loss did not improve for 3 epoch(s)
[Checkpoint] Restored best weights from epoch 15 (val_loss: 0.3721)
```

`ModelCheckpoint` writes `.npz` on pennylane and `.ckpt` on torch, keyed the same either
way, and restores the best weights when the run ends. Lightning callbacks are rejected on
purpose: they are typed against Lightning's hooks, so the pennylane loop could only ignore
them, and an ignored `EarlyStopping` hands back a fully trained model without saying so.

Writing your own means three optional methods, `on_fit_start`, `on_epoch_end` and
`on_fit_end`, each taking one `LoopState`:

```python
from pyqit.core import BaseCallback

class StopWhenConverged(BaseCallback):
    def on_epoch_end(self, state):
        if state.metrics["train_loss"] < 0.01:
            state.stop = True
```

`state` carries the model, datamodule, history, reporter, epoch index and this epoch's
metrics. `state.stop` is the one field a callback may write.

## Losses

`"mse"`, `"hinge"` and `"cross_entropy"` ship built in. Pass a name or a callable:

```python
import pennylane.numpy as pnp

def weighted_mse(preds, targets):
    return pnp.mean((preds - targets) ** 2 * (1 + targets))

pyqit.Trainer(max_epochs=30, loss_fn=weighted_mse)
```

Models emit probabilities rather than logits, so a torch loss must not use
`F.cross_entropy`. Losses subclass `BaseLoss` and register themselves by existing in an
importable, non-underscore module.

## Barren-plateau diagnostic

```python
trainer = pyqit.Trainer(max_epochs=50, check_bp=True, bp_samples=200)
history = trainer.fit(model, dm)
```

```
           BP Diagnostic Result : BARREN PLATEAU
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ Metric / Layer              ┃    Value ┃         Status ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ Qubits                      │        4 │                │
│ Samples                     │      200 │                │
│ Expected Variance           │ 1.56e-02 │       Baseline │
│ Quantum Variance            │ 3.74e-03 │ BARREN PLATEAU │
├─────────────────────────────┼──────────┼────────────────┤
│ Layer: main_circuit.weights │   0.240x │      ← plateau │
└─────────────────────────────┴──────────┴────────────────┘
```

Gradients are sampled at uniformly random weights and their variance compared against
`1/2**n_qubits` for a local cost or `1/(3·4^(n-1))` for a global one. Call
`check_barren_plateau(model, dm)` from `pyqit.utils.diagnostic` if you want the `BPResult`
without training. The table falls back to ASCII when `rich` is not installed.

## Pipelines

`QuantumPipeline` composes stages sequentially or as an ensemble.

```python
from pyqit.core import QuantumPipeline

pipe = QuantumPipeline(
    [
        ("encode", VQCClassifier(n_qubits=4, n_layers=1)),
        ("head", VQCClassifier(n_qubits=4, n_layers=2)),
    ],
    mode="sequential",
)
pipe.fit(dm, trainers=pyqit.Trainer(max_epochs=20))
preds = pipe.predict(X)          # takes raw arrays, not a DataModule
```

Sequential fitting materializes intermediate data by running each fitted stage over the
whole split, so upstream stages do not re-run per batch. `fit_mode="frozen_backbone"`
requires every non-final stage to be `trainable=False`.

## Extending

Add an ansatz, embedding, model or loss by writing the class and giving it an
`object_type` tag. There is no registration step, and the test suite picks it up
automatically as long as it implements `get_test_params()`. A new backend is one
`BaseTrainingLoop` subclass with a `backend` tag.

One caveat worth knowing before you spend an afternoon on it. skbase's class walk skips
modules whose name starts with `_`, so a loss or loop defined in a private module is
silently never registered.

> #### Have a look at some tutorials [here](https://github.com/phoeenniixx/pyQit/tree/main/docs/tutorials/) for more info!

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the
[issues page](https://github.com/phoeenniixx/pyQit/issues). If you are building novel
ansatzes, custom embeddings, or new diagnostic tools, please submit a PR.

```bash
pip install -e ".[dev,all_extras]"
python -m pytest -n auto
pre-commit run --all-files
```

CI runs the suite twice, once with no soft dependencies and once with all of them, across
Python 3.10 to 3.14 on Linux, macOS and Windows. New code touching torch, lightning,
matplotlib or rich has to degrade gracefully in the no-softdeps job.
