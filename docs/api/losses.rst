======
Losses
======

.. currentmodule:: pyqit.core

Pass a loss by name or hand :class:`~pyqit.core.Trainer` a callable:

.. code-block:: python

   import pyqit
   import pennylane.numpy as pnp

   pyqit.Trainer(max_epochs=30, loss_fn="cross_entropy")

   def weighted_mse(preds, targets):
       return pnp.mean((preds - targets) ** 2 * (1 + targets))

   pyqit.Trainer(max_epochs=30, loss_fn=weighted_mse)

Three losses ship built in: ``"mse"``, ``"hinge"`` and ``"cross_entropy"``.

Models emit probabilities, not logits
=====================================

This trips up anyone porting a torch loss. ``F.cross_entropy`` applies its own
log-softmax, so feeding it probabilities scores the wrong thing quietly rather
than raising. :class:`CrossEntropyLoss` takes the log directly instead. Write
new torch losses the same way.

PennyLane ships no ML losses of its own, so the ``pnp`` implementations here are
hand written by design.

Adding a loss
=============

Subclass :class:`BaseLoss` and tag it. Nothing hand-registers it, because
``loss_registry()`` discovers classes by walking the package for an
``object_type`` of ``"loss"``. The ``backends`` tag declares which backends the
class implements, and a backend listed there needs the matching ``_pennylane``
or ``_torch`` method. ``target_dtype`` tells the Lightning adapter whether
targets are class indices.

One caveat costs people an afternoon. skbase's class walk skips modules whose
name starts with an underscore, so a loss defined in a private module never
registers and never says why.

See :doc:`the contributing guide </contributing>`.

Related
=======

:doc:`trainer` takes ``loss_fn``. :doc:`measurements` decides what the model
emits, which is what the loss then scores. The
:doc:`callbacks tutorial </tutorials/callbacks>` trains with
``loss_fn="cross_entropy"``.

.. autosummary::
   :nosignatures:

   BaseLoss
   MSELoss
   HingeLoss
   CrossEntropyLoss
   get_loss_fn

.. autoclass:: BaseLoss
.. autoclass:: MSELoss
.. autoclass:: HingeLoss
.. autoclass:: CrossEntropyLoss
.. autofunction:: get_loss_fn
