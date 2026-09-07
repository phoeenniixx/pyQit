==========
DataModule
==========

.. currentmodule:: pyqit

:class:`DataModule` holds the data and does nothing with it until ``setup()``
runs. :meth:`Trainer.fit <pyqit.core.Trainer.fit>` and
:meth:`Trainer.predict <pyqit.core.Trainer.predict>` call ``setup()`` for you,
which is why properties like ``X_train`` raise before then.

.. code-block:: python

   import pyqit
   from sklearn.datasets import make_moons

   X, y = make_moons(n_samples=200, noise=0.1, random_state=0)
   dm = pyqit.DataModule(X, y, normalize="minmax", batch_size=16)

What setup does, in order
=========================

#. Split into train, val and test.
#. Normalize. This step is stateful, so the normalizer fits on train only and
   then applies to val and test. ``minmax``, ``zscore``, ``l1`` and ``l2`` are
   built in.
#. Prescale for the circuit. This step is stateless and the model's embedding
   drives it, not the user.
#. Apply any ``transform``.

Prescaling explains why feature shaping is not your job. ``AngleEmbedding`` pads
or truncates to ``n_qubits`` and multiplies by pi. ``AmplitudeEmbedding`` pads to
``2 ** n_qubits`` and L2-normalizes. The model class picks the embedding, so the
model class decides the shape.

Repeated setup
==============

``setup()`` returns early if it already ran, unless you pass ``force=True``. Two
things sidestep that early return. ``batch_size`` applies every time, because it
only affects loader construction and never the split or the fitted normalizer,
so ``Trainer(batch_size=...)`` can override a datamodule you set up by hand.
``encoder`` and ``n_qubits`` are overwritten only when actually supplied, so they
survive a later ``setup(force=True)`` that omits them.

Building from tables
====================

.. code-block:: python

   dm = pyqit.DataModule.from_dataframe(df, label_col="target", normalize="zscore")
   dm = pyqit.DataModule.from_csv("data.csv", label_col="target")

Related
=======

:doc:`embeddings` explains the prescaling step and why the model decides input
shaping. :doc:`trainer` calls ``setup()`` for you. :doc:`pipeline` rebuilds a
datamodule between sequential stages. Every
:doc:`tutorial </tutorials/index>` starts by building one.

.. autosummary::
   :nosignatures:

   DataModule

.. autoclass:: DataModule
