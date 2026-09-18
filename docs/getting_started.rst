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

With uv, ``uv add pyqit`` or ``uv pip install "pyqit[pytorch]"`` takes the same
extras. There is no conda package. Inside a conda environment, use pip.

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

With the ``pytorch`` extra installed, the same model trains through Lightning.
Set the backend before you build the model, because every object reads it once
in ``__init__``. A model built earlier stays on the backend it was built with.

.. code-block:: python

   pyqit.set_backend("torch")     # raises ImportError if torch is missing
   pyqit.set_seed(42)

   dm = pyqit.DataModule(X, y, normalize="minmax", batch_size=16)
   model = VQCClassifier(
       n_qubits=4,
       n_layers=3,
       ansatz=SELAnsatz,
       encoder=AngleEmbedding,
   )

   trainer = pyqit.Trainer(
       max_epochs=30,
       learning_rate=0.05,
       backend_kwargs={"enable_model_summary": True},
   )
   history = trainer.fit(model, dm)
   preds = trainer.predict(model, dm, return_format="numpy")

The Trainer, the DataModule, callbacks and the history do not change. Anything
Lightning's own ``Trainer`` accepts goes through ``backend_kwargs``. Training
stays on the CPU unless you ask for an accelerator there. ``predict`` returns
whatever the backend produces by default, so pass ``return_format`` when you
want a numpy array.

:doc:`api/config` covers the backend and seed ordering in detail.

Next
====

- :doc:`tutorials/vqc` trains a classifier end to end and then composes two
  models into a pipeline.
- :doc:`tutorials/callbacks` covers early stopping and checkpoints.
- :doc:`tutorials/barren_plateau` shows the gradient-variance check.
- :doc:`api_reference` has one page per kind of object and one page per class.
