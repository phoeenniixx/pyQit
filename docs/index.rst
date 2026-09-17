=====
PyQit
=====

PyQit is a quantum machine learning framework built on PennyLane. It adds a
Trainer, a DataModule and a set of models on top of PennyLane QNodes, so you
train a variational circuit by calling ``trainer.fit(model, dm)``.

.. warning::

   Version |release|. The API is unstable and still changing.

The base install needs only PennyLane and NumPy. PyTorch and PyTorch Lightning
are optional. Installing them adds a second backend that trains through
Lightning.

Quickstart
==========

.. code-block:: python

   from sklearn.datasets import make_moons

   import pyqit
   from pyqit.ansatzes import SELAnsatz
   from pyqit.core import AngleEmbedding
   from pyqit.models import VQCClassifier

   pyqit.set_seed(42)

   X, y = make_moons(n_samples=200, noise=0.1, random_state=0)
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

Install with ``pip install -e ".[dev]"`` for the base package, or
``pip install -e ".[dev,all_extras]"`` to add torch, lightning, matplotlib and
rich.

How it works
============

A run involves three objects.

A :doc:`DataModule <api/datamodule>` holds the data and does nothing with it
until ``setup()`` runs, which the Trainer calls for you. It splits the data,
normalizes it with statistics fitted on the training split only, then prescales
the features for the circuit.

A :doc:`model <api/models>` owns the QNode. It composes an
:doc:`embedding <api/embeddings>`, which maps features onto wires, an
:doc:`ansatz <api/ansatzes>`, which holds the trainable weights, and a
:doc:`measurement <api/measurements>`, which turns the final state into numbers.

A :doc:`Trainer <api/trainer>` runs the two together. It seeds, sets up the
data, assembles :doc:`callbacks <api/callbacks>`, and hands off to the training
loop of the active backend.

The model decides how its input is shaped. Each embedding carries a tag naming
the prescaling its circuit needs, such as zero-padding to one feature per wire
and multiplying by pi, and ``setup()`` applies it. You do not reshape features
by hand. Input wider than the embedding takes raises an error.

Switching backends
==================

.. code-block:: python

   pyqit.set_backend("torch")     # raises ImportError if torch is missing

The torch backend wraps the QNode in a ``qml.qnn.TorchLayer`` and trains through
Lightning. Models, callbacks and the returned history work the same way on both
backends.

The backend is a global setting, and each object reads it once in its
``__init__``. Set the backend and the seed before you build a model.
:doc:`api/config` covers the ordering.

Features
========

- :doc:`Callbacks <api/callbacks>` for early stopping and checkpointing that
  run on both backends. PyQit does not accept Lightning callbacks, because the
  PennyLane loop cannot run them.
- A :doc:`barren-plateau diagnostic <api/diagnostics>`. It samples gradients at
  random weights and compares their variance against a theoretical floor.
  ``Trainer(check_bp=True)`` runs it before training starts.
- :doc:`Pipelines <api/pipeline>` that compose models in sequence or as an
  ensemble, including a frozen backbone with a trainable head.
- :doc:`Losses <api/losses>` selected by name. A callable works anywhere a name
  does.
- Any PennyLane :doc:`device <api/models>`, plugins included. The
  PennyLane-Qiskit plugin is tested through its local simulators.
  ``Trainer(verbose=2)`` prints the differentiation method PennyLane picks for
  the device, which sets how many circuits each gradient costs.

To add a model, ansatz, embedding or loss, write a class and tag it. The test
suite finds it by walking the package. See :doc:`contributing`.

Where to go next
================

:doc:`tutorials/index` has three worked notebooks. Start with the
:doc:`VQC tutorial <tutorials/vqc>`. The :doc:`api_reference` has one page per
kind of object and one page per class.

.. toctree::
   :hidden:
   :caption: Tutorials
   :maxdepth: 1

   tutorials/index

.. toctree::
   :hidden:
   :caption: Reference
   :maxdepth: 2

   api_reference

.. toctree::
   :hidden:
   :caption: Project

   contributing
   changelog
