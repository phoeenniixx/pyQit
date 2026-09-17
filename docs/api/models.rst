======
Models
======

.. currentmodule:: pyqit.models

A model owns the QNode. It builds the circuit from an ansatz and an embedding,
holds the weights, and exposes ``forward``. :class:`BaseModel` defines the
contract and :class:`BaseQuantumModel` adds the quantum-specific plumbing that
every circuit model shares.

:class:`VQCClassifier` is the model to start with. It takes a qubit count, a
depth, an ansatz class and an embedding class, and wires them together:

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

:class:`VQCRegressor` is the same circuit read as a value: the parity
``Z ⊗ ... ⊗ Z`` expectation, Qiskit ML's ``VQR`` default, through a trainable
``scale * <Z> + offset`` head that starts at identity, so targets need no
scaling. ``output_scale=False`` reproduces ``VQR``, whose targets must lie in
``[-1, 1]``. The loops record accuracy as NaN for it.

Two models take ``n_features`` instead of an encoder, because they encode the
input themselves and the DataModule leaves it unscaled.
:class:`DataReuploadingClassifier` re-encodes the input inside every layer
(Perez-Salinas et al. 2020), and :class:`DressedQuantumClassifier` sandwiches
the circuit between two dense layers (Mari et al. 2020), the first hybrid. Its
classical weights sit in the same flat dict as the circuit's, under
``pre_net.*`` and ``post_net.*``.

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

Devices
=======

``device=`` goes straight to ``qml.device``, so any PennyLane device name works,
plugins included. ``shots=None`` asks for analytic simulation, and the defaults
assume a local analytic simulator. PennyLane picks the differentiation method
per device. ``default.qubit`` gets backprop, ``lightning.qubit`` adjoint, and
any device with shots, or real hardware, parameter-shift. Parameter-shift runs
``1 + 2 * n_params`` circuits for every gradient, so a model that trains in
seconds locally can take hours on a queue. ``diff_methods`` tells you which
method you are getting, and ``Trainer(verbose=2)`` prints it in the model
summary next to the device.

.. code-block:: python

   model = VQCClassifier(n_qubits=4, device="qiskit.aer", shots=1024)
   model.diff_methods(dm.X_train[:1])   # {"main_circuit": "parameter-shift"}

``pip install pyqit[qiskit]`` adds the Qiskit plugin. Its local simulators
sample even at ``shots=None``, where they run 1024 shots, so expect
parameter-shift and shot noise there. No real QPU has been run against pyqit
yet.

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

Available models
================

.. autosummary::
   :nosignatures:

   VQCClassifier
   VQCRegressor
   DataReuploadingClassifier
   DressedQuantumClassifier

.. autoclass:: VQCClassifier
.. autoclass:: VQCRegressor
.. autoclass:: DataReuploadingClassifier
.. autoclass:: DressedQuantumClassifier

Base classes and mixins
=======================

Subclass these to write a model; you never instantiate them directly.

.. autosummary::
   :nosignatures:

   BaseModel
   BaseQuantumModel
   ClassifierMixin
   RegressorMixin

.. autoclass:: BaseModel
.. autoclass:: BaseQuantumModel
.. autoclass:: ClassifierMixin
.. autoclass:: RegressorMixin
