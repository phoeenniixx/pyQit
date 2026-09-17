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

Available models
================

Each page gives the circuit, the paper it follows, every constructor argument
and a runnable example.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   VQCClassifier
   VQCRegressor
   DataReuploadingClassifier
   DressedQuantumClassifier

Classifiers and regressors
==========================

A mixin decides how the Trainer reads a model's output.
:class:`ClassifierMixin` turns it into class probabilities and hard labels.
:class:`RegressorMixin` passes the raw output through, and the training loops
record accuracy as NaN for it.

Models that encode their own input
==================================

A model that takes ``encoder`` exposes the built embedding as ``embedding_obj``,
and the DataModule prescales the input for it. A model that takes
``n_features`` instead encodes the input inside its own circuit. It has no
``embedding_obj``, so the DataModule normalizes the features and leaves them
unscaled.

Hybrid networks are pipelines
=============================

:class:`DressedQuantumClassifier` is the dressed quantum circuit of Mari et al.
(2020), a circuit between two dense layers. It trains like any other model.
Inside it runs a :doc:`pipeline <pipeline>` of three layers, and their weights
are the model's own, in the same flat dict under ``pre_net.*``, ``quantum.*``
and ``post_net.*``. It takes ``n_features`` instead of an encoder, and the
DataModule leaves its input unscaled.

.. code-block:: python

   from pyqit.models import DressedQuantumClassifier

   model = DressedQuantumClassifier(n_features=8, n_qubits=4, n_layers=6)
   history = pyqit.Trainer(max_epochs=20).fit(model, dm)

Its stages come from ``pyqit.models.layers``, the building blocks for a hybrid
of your own. :class:`~pyqit.models.layers.DenseLayer` and
:class:`~pyqit.models.layers.QuantumLayer` emit features, and
:class:`~pyqit.models.layers.DenseClassifier` is the classical head that turns
the last features into class probabilities. A layer is a pipeline stage and
cannot be fit alone.

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
parametrizes every model test over it. Then add the class name to the list
above.

You subclass these and never instantiate them directly.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   BaseModel
   BaseQuantumModel
   ClassifierMixin
   RegressorMixin

See :doc:`the contributing guide </contributing>` for the checklist and the PR conventions.

Related
=======

:doc:`ansatzes` and :doc:`embeddings` are the two pieces a model composes, and
:doc:`measurements` decides what comes out of the circuit. :doc:`trainer` runs
the model against a :doc:`datamodule`. The
:doc:`VQC tutorial </tutorials/vqc>` walks through building and training one.
