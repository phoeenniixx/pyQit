=======
Trainer
=======

.. currentmodule:: pyqit.core

:class:`Trainer` orchestrates a run but trains nothing itself. It seeds the RNG,
calls ``setup()`` on the datamodule, prints the model table, optionally runs the
barren-plateau pre-flight, assembles the callback list, then hands off to a
training loop chosen by the active backend.

.. code-block:: python

   import pyqit

   trainer = pyqit.Trainer(max_epochs=30, learning_rate=0.05)
   history = trainer.fit(model, dm)

   print(history.best_epoch, history.best_score)
   trainer.validate(model, dm)             # {"val_loss": ..., "val_acc": ...}
   trainer.test(model, dm)                 # {"test_loss": ..., "test_acc": ...}
   preds = trainer.predict(model, dm)      # runs on the test split

:meth:`Trainer.fit` returns a :class:`TrainingHistory` holding ``train_loss``,
``val_loss``, ``train_acc``, ``val_acc`` and ``epoch_times``, one entry per
epoch.

Seeding happens too late for weights
====================================

``Trainer(seed=...)`` covers training and the diagnostics, not weight
initialisation. Models draw their weights in ``__init__``, so seed before you
build one:

.. code-block:: python

   pyqit.set_seed(42)                      # first
   model = VQCClassifier(n_qubits=4)       # then

The same ordering applies to :func:`~pyqit.set_backend`, which each object reads
once in its own ``__init__``.

Settings a backend cannot honour
================================

Each loop declares what it cannot do, and :class:`Trainer` validates that when
the loop is built, before any data is split. The PennyLane loop rejects
``backend_kwargs``, because it has no Lightning trainer to forward them to, and
warns on ``logger``, because metrics still come back in the history. A rejected
setting raises. A warned one trains correctly.

Both checks measure against the signature defaults, so a Trainer built with
defaults never complains.

Passing Lightning settings
==========================

On the torch backend, ``backend_kwargs`` goes straight to
``lightning.pytorch.Trainer``:

.. code-block:: python

   pyqit.Trainer(
       max_epochs=50,
       backend_kwargs={"accelerator": "gpu", "devices": 1, "gradient_clip_val": 0.5},
   )

Lightning's constructor is not mirrored onto :class:`Trainer`. It carries roughly
forty parameters, most of which a PennyLane optimizer loop cannot honour, and it
holds none of ``learning_rate``, ``optimizer``, ``loss_fn`` or ``batch_size``
anyway. The accelerator defaults to ``"cpu"``, so a GPU runs only when you ask
for one.

Related
=======

:class:`Trainer` trains a :doc:`model <models>` against a :doc:`datamodule`.
:doc:`losses` covers ``loss_fn``, :doc:`callbacks` covers ``callbacks``, and
:doc:`config` covers the backend and seeding that must be set before you build
anything. ``check_bp`` runs the check described in :doc:`diagnostics`. The
:doc:`VQC tutorial </tutorials/vqc>` is a full run.

.. autosummary::
   :nosignatures:

   Trainer
   TrainingHistory

.. autoclass:: Trainer
.. autoclass:: TrainingHistory
