=========
Callbacks
=========

.. currentmodule:: pyqit.core.callbacks

A pyqit callback implements up to three hooks, ``on_fit_start``,
``on_epoch_end`` and ``on_fit_end``, each taking one :class:`LoopState`. Write it
once and both backends honour it.

.. code-block:: python

   import pyqit
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

.. code-block:: text

   [EarlyStopping] Stopped at epoch 18 - val_loss did not improve for 3 epoch(s)
   [Checkpoint] Restored best weights from epoch 15 (val_loss: 0.3721)

Why Lightning callbacks are rejected
====================================

They are typed against Lightning's hooks, so the PennyLane loop could only
ignore them. An ignored :class:`EarlyStopping` hands back a fully trained model
without saying so, and that failure is invisible. Rejecting them at the door is
the louder option.

On the torch backend a shim reads Lightning's ``callback_metrics`` into the same
metric names and forwards ``state.stop`` onto ``trainer.should_stop``, so the
same callback object works on both sides.

Checkpointing
=============

:class:`ModelCheckpoint` owns checkpointing on both backends, and Lightning's
own is switched off so a run is never written twice. Only the file format
differs, ``.ckpt`` holding a ``state_dict`` on torch and ``.npz`` on pennylane.
The array keys match ``model.weights`` either way.

Three files can be written independently. ``save_best`` uses the stem from
``filename``, ``save_last`` uses ``last``, and ``every_n_epochs`` uses
``epoch<n>``, numbered from zero to match ``best_epoch``. The best file is
written once after training. Set ``save_on_improve=True`` to write on every
improvement instead, at the cost of extra I/O.

``restore_best`` defaults to whatever ``save_best`` is, not to ``True``, so
``save_best=False, save_last=True`` will not quietly hand you back the best
model when you asked for the last one.

Nothing here resumes a run. These files hold weights only, with no optimizer
state and no epoch counter.

Writing your own
================

.. code-block:: python

   from pyqit.core import BaseCallback

   class StopWhenConverged(BaseCallback):
       def on_epoch_end(self, state):
           if state.metrics["train_loss"] < 0.01:
               state.stop = True

``state`` carries the model, datamodule, history, reporter, epoch index and this
epoch's metrics. ``state.stop`` is the one field a callback may write.

Related
=======

:doc:`trainer` takes the ``callbacks`` list and assembles it. The
:doc:`callbacks tutorial </tutorials/callbacks>` runs both built-in callbacks
together and reloads the checkpoint afterwards.

.. autosummary::
   :nosignatures:

   BaseCallback
   LoopState
   HistoryCallback
   EarlyStopping
   ModelCheckpoint

.. autoclass:: BaseCallback
.. autoclass:: LoopState
.. autoclass:: HistoryCallback
.. autoclass:: EarlyStopping
.. autoclass:: ModelCheckpoint
