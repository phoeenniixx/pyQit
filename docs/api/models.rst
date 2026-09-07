======
Models
======

.. currentmodule:: pyqit.models

A model owns the QNode. It builds the circuit from an ansatz and an embedding,
holds the weights, and exposes ``forward``. :class:`BaseModel` defines the
contract and :class:`BaseQuantumModel` adds the quantum-specific plumbing that
every circuit model shares.

:class:`VQCClassifier` is the concrete classifier that ships today. It takes a
qubit count, a depth, an ansatz class and an embedding class, and wires them
together:

.. code-block:: python

   from pyqit.ansatzes import SELAnsatz
   from pyqit.core import AngleEmbedding
   from pyqit.models import VQCClassifier

   model = VQCClassifier(
       n_qubits=4,
       n_layers=3,
       ansatz=SELAnsatz,
       encoder=AngleEmbedding,
   )

The ansatz and encoder arrive as classes, not instances. The model builds them
with its own ``n_qubits``, so the two cannot disagree about width.

Weights exist before training
=============================

A model draws its weights in ``__init__``, not on the first ``fit``. Two things
follow. Seeding afterwards will not reproduce them, and calling
:func:`~pyqit.set_backend` afterwards will not move the model, because each
object reads the backend once in its own ``__init__`` and caches it.

Weights come back as a flat dict keyed ``"<qnode_name>.<weight_name>"``, the
same on both backends:

.. code-block:: python

   model.weights            # {"main_circuit.weights": array(...)}

``update_weights`` writes that dict back. It is a no-op under torch, where
autograd owns the parameters directly.

Writing a new model
===================

Expose the encoder as ``embedding_obj``. The framework reads that attribute to
decide prescaling, and a mismatched name disables prescaling silently instead of
raising, which is the kind of bug that produces plausible numbers for weeks.

Beyond that, give the class an ``object_type`` tag of ``"model"`` and a
``get_test_params()`` method returning a list of kwarg dicts. There is no
registration step. The suite discovers the class by walking the package and
parametrizes every model test over it.

See :doc:`the contributing guide </contributing>` for the checklist and the PR conventions.

Related
=======

:doc:`ansatzes` and :doc:`embeddings` are the two pieces a model composes, and
:doc:`measurements` decides what comes out of the circuit. :doc:`trainer` runs
the model against a :doc:`datamodule`. The
:doc:`VQC tutorial </tutorials/vqc>` walks through building and training one.

.. autosummary::
   :nosignatures:

   BaseModel
   BaseQuantumModel
   ClassifierMixin
   VQCClassifier

.. autoclass:: BaseModel
.. autoclass:: BaseQuantumModel
.. autoclass:: ClassifierMixin
.. autoclass:: VQCClassifier
