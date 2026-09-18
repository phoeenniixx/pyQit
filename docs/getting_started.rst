===============
Getting started
===============

Installation
============

.. code-block:: bash

   pip install pyqit                 # pennylane and numpy
   pip install "pyqit[pytorch]"      # adds torch and pytorch lightning
   pip install "pyqit[all_extras]"   # adds matplotlib and rich as well
   pip install "pyqit[qiskit]"       # adds the PennyLane-Qiskit plugin

``all_extras`` covers torch, lightning, matplotlib and rich. The Qiskit plugin
is not part of it, because it needs Python 3.11 or newer. Install it through
the ``qiskit`` extra on its own.

PyQit needs Python 3.10 or newer.

First model
===========

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

   print(history.best_epoch, history.best_score)   # 22 0.0947
   preds = trainer.predict(model, dm)              # runs on the test split

``fit`` returns a :class:`~pyqit.core.TrainingHistory` with one entry per epoch
for train and validation loss and accuracy. ``predict`` runs the model on the
test split of the same DataModule.

The DataModule does nothing with the data until the Trainer calls ``setup()``.
That splits it, fits the normalizer on the training split only, and prescales
the features the way the model's embedding expects. You do not reshape
features by hand.

The model draws its weights and reads the backend in ``__init__``. Seed and
pick the backend before you build it.

Training through PyTorch Lightning
==================================

With the ``pytorch`` extra installed, one call moves training to Lightning.

.. code-block:: python

   pyqit.set_backend("torch")     # raises ImportError if torch is missing

Everything after that line is the same code. Lightning settings go through
``Trainer(backend_kwargs=...)``.

Next
====

- :doc:`tutorials/vqc` trains a classifier end to end and then composes two
  models into a pipeline.
- :doc:`tutorials/callbacks` covers early stopping and checkpoints.
- :doc:`tutorials/barren_plateau` shows the gradient-variance check.
- :doc:`api_reference` has one page per kind of object and one page per class.
