======
Layers
======

.. currentmodule:: pyqit.models.layers

A layer is a reusable block that models are assembled from. It holds its own
weights and runs a batch through ``forward``, and it can be classical or
quantum. Nothing about a layer is specific to hybrid networks.
:class:`BaseVQC` is the embedding, ansatz and measurement block that
:class:`~pyqit.models.VQCClassifier` and :class:`~pyqit.models.VQCRegressor`
share, and :class:`QuantumLayer` is that same block read as ``<Z>`` on every
wire.

There are two ways to use a layer. A model class can build layers in its
``__init__`` and run them in ``forward``, which is how
:class:`~pyqit.models.DressedQuantumClassifier` is written. Or you can compose
layers yourself in a :doc:`pipeline <pipeline>` and train them together with
``fit_mode="joint"``, where the loss after the last layer reaches every layer
before it.

.. code-block:: python

   import pyqit
   from pyqit.core import QuantumPipeline
   from pyqit.models.layers import DenseClassifier, DenseLayer, QuantumLayer

   hybrid = QuantumPipeline(
       [
           ("pre", DenseLayer(n_features=8, n_out=4, activation="tanh")),
           ("circuit", QuantumLayer(n_qubits=4, n_layers=2)),
           ("head", DenseClassifier(n_features=4)),
       ],
       fit_mode="joint",
   )
   history = pyqit.Trainer(max_epochs=20).fit(hybrid, dm)

Available layers
================

Each layer's page gives its inputs, its output shape and its weight names.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   DenseLayer
   QuantumLayer
   DenseClassifier

Feature layers and heads
========================

A feature layer returns a ``(n_samples, width)`` array for the next layer to
read. Its output is not a prediction, so it cannot be fit alone against labels.
A head comes last. It returns predictions and has ``predict_step``, which is
what ``Trainer.predict`` calls for hard labels.

The pipeline prescales the input of every quantum layer for that layer's
embedding, so the layer in front of it needs no scaling of its own. Narrower
input is zero-padded to the layer's width and wider input raises.

What every layer provides
=========================

``forward(X, **custom_weights)`` runs the layer. ``weights`` is a flat dict
keyed ``"<name>.<weight>"``, the same on both backends, and ``update_weights``
writes it back. Inside a pipeline the stage name goes in front of each key.

.. code-block:: python

   DenseLayer(n_features=8, n_out=4).weights
   # {"dense.weight": ..., "dense.bias": ...}
   hybrid.weights
   # {"pre.dense.weight": ..., "circuit.main_circuit.weights": ..., ...}

A layer reads the backend once in its own ``__init__``, as models do. Call
:func:`~pyqit.set_backend` before building it.

Adding a layer
==============

Subclass :class:`~pyqit.models.BaseModel` for a classical layer and register
its weights with ``register_dense``. Subclass
:class:`~pyqit.models.BaseQuantumModel` for a quantum one and use
``register_qnode``. Set the ``object_type`` tag to ``"layer"``, and run the
registered entries with ``execute_qnode`` inside
``forward(X, **custom_weights)``, passing ``custom_weights`` through. The
training loops rely on that argument to route weights. A head also mixes in
:class:`~pyqit.models.ClassifierMixin` or
:class:`~pyqit.models.RegressorMixin` for ``predict_step``.

A quantum layer built from an embedding and an ansatz can subclass
:class:`BaseVQC` and skip the circuit. It then implements ``_resolve_readout``,
which picks the measurement, and ``forward``, which maps the raw output.

Implement ``get_test_params()``. The layer suite finds the class by its tag and
trains it in front of a :class:`DenseClassifier`. It reads the input width from
``n_features`` or ``n_qubits`` and the output width from ``n_out`` or
``n_qubits``, so a feature layer must expose those attributes. Heads are
excluded from that test by name in ``test_all_layers.py``. Then add the class
name to the list above. See :doc:`the contributing guide </contributing>`.

You subclass this and never instantiate it directly.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   BaseVQC

Related
=======

:doc:`pipeline` composes and trains layers. :doc:`models` covers the models
built from them. A quantum layer takes an :doc:`ansatz <ansatzes>` and an
:doc:`embedding <embeddings>` the way a model does.
