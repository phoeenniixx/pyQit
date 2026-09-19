=========
Tutorials
=========

Find some examples on how to use `pyqit` here:


:doc:`vqc`
   Trains a :class:`~pyqit.models.VQCClassifier` on a synthetic dataset using the
   torch backend, plots the loss curve from ``history.as_dict()``, then evaluates
   on the test split. Ends with a :class:`~pyqit.models.VQCRegressor` fitting a
   sine curve. Start here.

:doc:`callbacks`
   Runs :class:`~pyqit.core.callbacks.EarlyStopping` and
   :class:`~pyqit.core.callbacks.ModelCheckpoint` together, then inspects what each one
   recorded: ``stopped_epoch``, ``stopping_reason``, ``best_epoch`` and
   ``best_path``. Ends by resuming the run from ``last.npz`` with
   ``ModelCheckpoint(resume_from=...)`` and predicting new rows through a saved
   :class:`~pyqit.DataModule`.

:doc:`barren_plateau`
   Takes a circuit at 8 qubits and 15 layers, deep enough to plateau, and shows
   the gradient variance collapsing below the baseline. Then runs the same check
   through ``Trainer(check_bp=True)`` on a circuit that trains fine, so you can
   see both verdicts.

:doc:`pipeline`
   Composes two models into a :class:`~pyqit.core.QuantumPipeline` with a frozen
   backbone and a trainable head, then builds a dense, circuit, dense network
   from ``pyqit.models.layers`` and trains it as one model with
   ``fit_mode="joint"``.

.. toctree::
   :hidden:
   :maxdepth: 1

   vqc
   callbacks
   barren_plateau
   pipeline
