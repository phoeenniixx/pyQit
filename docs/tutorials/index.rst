=========
Tutorials
=========

Three worked notebooks. Each one runs end to end and CI executes them on every
change, so the outputs you see are the outputs they produce.

:doc:`vqc`
   Trains a :class:`~pyqit.models.VQCClassifier` on a synthetic dataset using the
   torch backend, plots the loss curve from ``history.as_dict()``, then evaluates
   on the test split. The second half composes two models into a
   :class:`~pyqit.core.QuantumPipeline` with a frozen backbone and a trainable
   head. Start here.

:doc:`callbacks`
   Runs :class:`~pyqit.core.callbacks.EarlyStopping` and
   :class:`~pyqit.core.callbacks.ModelCheckpoint` together, then inspects what each one
   recorded: ``stopped_epoch``, ``stopping_reason``, ``best_epoch`` and
   ``best_path``. Ends by reloading the ``.npz`` checkpoint and confirming the
   weights match.

:doc:`barren_plateau`
   Takes a circuit at 8 qubits and 15 layers, deep enough to plateau, and shows
   the gradient variance collapsing below the baseline. Then runs the same check
   through ``Trainer(check_bp=True)`` on a circuit that trains fine, so you can
   see both verdicts.

.. toctree::
   :hidden:
   :maxdepth: 1

   vqc
   callbacks
   barren_plateau
