=====
PyQit
=====

A high-level quantum machine learning framework built on PennyLane. It puts a
Trainer, DataModule and Model layer over PennyLane QNodes, so training a
variational circuit does not mean hand-rolling an optimizer loop.

.. warning::

   Version |release|. The API is unstable and still changing.

PyQit runs on PennyLane and NumPy alone. PyTorch and PyTorch Lightning are
optional. If you do not need hybrid models or Lightning's orchestration, you do
not have to install them.

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

Three objects do the work, and they stay separate on purpose.

A :doc:`DataModule <api/datamodule>` holds the data and does nothing with it
until ``setup()`` runs. It splits, then normalizes with a normalizer fitted on
the training split only, then prescales the features for the circuit.

A :doc:`model <api/models>` owns the QNode. It composes an
:doc:`ansatz <api/ansatzes>`, which holds the trainable weights, with an
:doc:`embedding <api/embeddings>`, which maps features onto wires, and a
:doc:`measurement <api/measurements>`, which turns the final state into numbers.

A :doc:`Trainer <api/trainer>` runs the two together. It seeds, sets up the
data, assembles :doc:`callbacks <api/callbacks>`, and hands off to a training
loop picked by the active backend.

The piece that catches people is that the model, not the user, decides input
shaping. The embedding carries a tag saying what shape the circuit needs, and
``setup()`` reads that tag. Angle encoding pads to one feature per wire and
scales by pi. Amplitude encoding pads to ``2 ** n_qubits`` and L2-normalizes.
You do not reshape anything by hand.

One backend switch, same everything else
========================================

.. code-block:: python

   pyqit.set_backend("torch")     # raises ImportError if torch is missing

That is the whole change. The QNode gets wrapped in a ``qml.qnn.TorchLayer`` and
training runs through Lightning. Your model, callbacks and history are
unchanged.

Backend selection is global and every object reads it once, in its own
``__init__``. So set the backend, and the seed, before you build anything.
:doc:`api/config` covers the ordering and what it affects.

What you get
============

:doc:`Callbacks <api/callbacks>` that work on both backends. ``EarlyStopping``
and ``ModelCheckpoint`` are written once and honoured by the PennyLane loop and
the Lightning loop alike. Lightning callbacks are rejected on purpose, because
the PennyLane loop could only ignore them.

A :doc:`barren-plateau diagnostic <api/diagnostics>`. It samples gradients at
random weights and compares the variance against a theoretical floor, so you
find out that a circuit cannot train before you spend an hour training it.
``Trainer(check_bp=True)`` runs it as a pre-flight check.

:doc:`Pipelines <api/pipeline>` that compose models in sequence or as an
ensemble, including a frozen backbone with a trainable head.

:doc:`Losses <api/losses>` with ``mse``, ``hinge`` and ``cross_entropy`` built
in, and callables accepted anywhere a name is.

Any PennyLane :doc:`device <api/models>`, plugins included. The
PennyLane-Qiskit plugin is tested through its local simulators, and
``Trainer(verbose=2)`` prints which differentiation method a device gets, since
that is what decides whether a run takes seconds or a queue.

Extending any of this means writing a class and tagging it. There is no
registration step, and the test suite picks it up automatically. See
:doc:`contributing`.

Where to go next
================

:doc:`tutorials/index` has three worked notebooks, and the
:doc:`VQC tutorial <tutorials/vqc>` is the place to start. The
:doc:`api_reference` documents every public class, one page per kind of module.

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
