# PyQit

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status: Active Development](https://img.shields.io/badge/status-active_development-orange.svg)]()
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Build Status](https://img.shields.io/github/actions/workflow/status/phoeenniixx/pyQit/test.yml)](https://github.com/phoeenniixx/pyqit/actions)

> **A high-level quantum machine learning framework built on PennyLane.**
> It aims to make quantum machine learning more accessible by reducing the steep learning curve, streamlining the boilerplate code required for training, and providing mathematically rigorous diagnostics.


## The Philosophy

Scaling Quantum Machine Learning (QML) research from toy models to enterprise hybrid pipelines is painful. Researchers spend hours rewriting training loops, debugging shape mismatches, and blindly waiting for models to train, only to realize their deep circuit hit a Barren Plateau at epoch 1.

**PyQit** abstracts away the infrastructure so you can focus on the science.

### Key Features
* **Lightweight & Modular:** PyQit runs natively on **PennyLane** and **NumPy**.
    > **PyTorch and PyTorch Lightning are strictly optional soft dependencies.** If you don't need deep learning hybrid models or GPU orchestration, you don't have to install them.
* **Backend Agnostic**: Seamlessly switch between native `pennylane` (pure Autograd) and `torch` (Lightning engine) with a single parameter.
* **Automated Diagnostics**: Features a mathematical "Pre-Flight Check" that runs Monte Carlo gradient sampling to detect Barren Plateaus mathematically *before* you waste compute time.
* **Enterprise Data Orchestration**: A stateful `DataModule` handles classical normalization (`minmax`, `zscore`) safely and separately from stateless quantum embedding projections (`Amplitude`, `Angle`).


## Installation

PyQit is currently in active development and is installed directly from source via `pyproject.toml`.

```bash
# 1. Clone the repository
git clone [https://github.com/yourusername/pyqit.git](https://github.com/yourusername/pyqit.git)
cd pyqit

# 2. Base Installation (PennyLane native, NO PyTorch required)
pip install -e .

# 3. Optional Installs
# If you want PyTorch and PyTorch Lightning support:
pip install -e .[pytorch]

# If you want the full suite (PyTorch, Matplotlib for diagnostics, Rich for terminal tables):
pip install -e .[all_extras]

# For contributors (pytest, sphinx, ruff, etc.):
pip install -e .[dev]
```

## Quickstart
Here is how you build a Variational Quantum Classifier (VQC) and run an automated Barren Plateau diagnostic in under 20 lines of code:
```python
from sklearn.datasets import make_moons
import numpy as np

import pyqit
from pyqit.ansatzes.sel import SELAnsatz
from pyqit.core.embeddings import AngleEmbedding
from pyqit.models.classification.vqc import VQCClassifier
from pyqit.data.datamodule import DataModule
from pyqit.core.trainer import Trainer

# 1. Prepare Data
X, y = make_moons(n_samples=200, noise=0.1)
# DataModule handle stateful normalization securely
dm = DataModule(X=X, y=y, normalize="minmax", batch_size=16)

# 2. Define Model
model = VQCClassifier(
    n_qubits=4,
    n_layers=3,
    n_classes=2,
    ansatz=SELAnsatz,
    encoder=AngleEmbedding
)

# 3. Initialize Trainer with Barren Plateau Pre-Flight Check
trainer = Trainer(
    max_epochs=10,
    check_bp=True,   # Automatically diagnoses gradient variance before training
    bp_samples=100,
    verbose=1
)

# 4. Fit
# Will print a diagnostic table evaluating your circuit against the McClean et al. baseline
history = trainer.fit(model, datamodule=dm)
```

## Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/phoeenniixx/pyQit/issues). If you are building novel ansatzes, custom embeddings, or new diagnostic tools, please submit a PR.
