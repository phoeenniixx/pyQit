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

Available callbacks
===================

.. autosummary::
   :toctree: generated/
   :nosignatures:

   EarlyStopping
   ModelCheckpoint
   HistoryCallback

Why Lightning callbacks are rejected
====================================

They are typed against Lightning's hooks, so the PennyLane loop could only
ignore them. An ignored :class:`EarlyStopping` hands back a fully trained model
without saying so, and that failure is invisible. Rejecting them at the door is
the louder option.

On the torch backend a shim reads Lightning's ``callback_metrics`` into the same
metric names and forwards ``state.stop`` onto ``trainer.should_stop``, so the
same callback object works on both sides.

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

.. autosummary::
   :toctree: generated/
   :nosignatures:

   BaseCallback
   LoopState

Related
=======

:doc:`trainer` takes the ``callbacks`` list and assembles it. The
:doc:`callbacks tutorial </tutorials/callbacks>` runs both built-in callbacks
together and reloads the checkpoint afterwards.
