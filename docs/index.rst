=====
PyQit
=====

PyQit is a quantum machine learning framework built on PennyLane. It adds a
Trainer, a DataModule and a set of models on top of PennyLane QNodes, so you
train a variational circuit by calling ``trainer.fit(model, dm)``.

.. warning::

   Version |release|. The API is unstable and still changing.

The base install needs only PennyLane and NumPy. PyTorch and PyTorch Lightning
are optional. Installing them adds a second backend that trains through
Lightning.

Compared with plain PennyLane
=============================

Training a variational classifier on PennyLane alone means writing the split,
the scaling, the padding onto wires, the batching and the optimizer loop
yourself.

.. code-block:: python

   import numpy as np
   import pennylane as qml
   from pennylane import numpy as pnp
   from sklearn.datasets import make_moons
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import MinMaxScaler

   X, y = make_moons(n_samples=200, noise=0.1, random_state=0)
   X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
   scaler = MinMaxScaler().fit(X_train)
   X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
   X_train = np.pad(X_train, ((0, 0), (0, 2))) * np.pi   # 2 features onto 4 wires
   X_test = np.pad(X_test, ((0, 0), (0, 2))) * np.pi

   dev = qml.device("default.qubit", wires=4)

   @qml.qnode(dev)
   def circuit(x, weights):
       qml.AngleEmbedding(x, wires=range(4))
       qml.StronglyEntanglingLayers(weights, wires=range(4))
       return qml.expval(qml.PauliZ(0))

   def cost(weights, X, y):
       preds = (1 - circuit(X, weights)) / 2
       return pnp.mean((preds - y) ** 2)

   np.random.seed(42)
   weights = pnp.array(np.random.uniform(size=(3, 4, 3)), requires_grad=True)
   opt = qml.AdamOptimizer(0.05)
   for epoch in range(30):
       for i in range(0, len(X_train), 16):
           batch = slice(i, i + 16)
           weights = opt.step(cost, weights, X=X_train[batch], y=y_train[batch])

   preds = (1 - circuit(X_test, weights)) / 2 > 0.5

The same job in PyQit. The DataModule does the split, the scaling and the
padding, the model builds the circuit, and the Trainer runs the loop.

.. code-block:: python

   from sklearn.datasets import make_moons

   import pyqit
   from pyqit.ansatzes import SELAnsatz
   from pyqit.core import AngleEmbedding
   from pyqit.models import VQCClassifier

   pyqit.set_seed(42)

   X, y = make_moons(n_samples=200, noise=0.1, random_state=0)
   dm = pyqit.DataModule(X, y, normalize="minmax", batch_size=16)

   model = VQCClassifier(
       n_qubits=4,
       n_layers=3,
       ansatz=SELAnsatz,
       encoder=AngleEmbedding,
   )

   trainer = pyqit.Trainer(max_epochs=30, learning_rate=0.05)
   history = trainer.fit(model, dm)

   print(history.best_epoch, history.best_score)   # 22 0.0947
   preds = trainer.predict(model, dm)              # runs on the test split

You also get a validation split, a per-epoch history, callbacks and a second
backend without changing the code above. :doc:`getting_started` covers the
install and walks through this example.

How it works
============

A run involves three objects.

A :doc:`DataModule <api/datamodule>` holds the data and does nothing with it
until ``setup()`` runs, which the Trainer calls for you. It splits the data,
normalizes it with statistics fitted on the training split only, then prescales
the features for the circuit.

A :doc:`model <api/models>` owns the QNode. It composes an
:doc:`embedding <api/embeddings>`, which maps features onto wires, an
:doc:`ansatz <api/ansatzes>`, which holds the trainable weights, and a
:doc:`measurement <api/measurements>`, which turns the final state into numbers.

A :doc:`Trainer <api/trainer>` runs the two together. It seeds, sets up the
data, assembles :doc:`callbacks <api/callbacks>`, and hands off to the training
loop of the active backend.

The model decides how its input is shaped. Each embedding carries a tag naming
the prescaling its circuit needs, such as zero-padding to one feature per wire
and multiplying by pi, and ``setup()`` applies it. You do not reshape features
by hand. Input wider than the embedding takes raises an error.

Switching backends
==================

.. code-block:: python

   pyqit.set_backend("torch")     # raises ImportError if torch is missing

The torch backend wraps the QNode in a ``qml.qnn.TorchLayer`` and trains through
Lightning. Models, callbacks and the returned history work the same way on both
backends.

The backend is a global setting, and each object reads it once in its
``__init__``. Set the backend and the seed before you build a model.
:doc:`api/config` covers the ordering.

Features
========

- :doc:`Callbacks <api/callbacks>` for early stopping and checkpointing that
  run on both backends. PyQit does not accept Lightning callbacks, because the
  PennyLane loop cannot run them.
- A :doc:`barren-plateau diagnostic <api/diagnostics>`. It samples gradients at
  random weights and compares their variance against a theoretical floor.
  ``Trainer(check_bp=True)`` runs it before training starts.
- :doc:`Pipelines <api/pipeline>` that compose models in sequence or as an
  ensemble, including a frozen backbone with a trainable head.
- :doc:`Losses <api/losses>` selected by name. A callable works anywhere a name
  does.
- Any PennyLane :doc:`device <api/models>`, plugins included. The
  PennyLane-Qiskit plugin is tested through its local simulators.
  ``Trainer(verbose=2)`` prints the differentiation method PennyLane picks for
  the device, which sets how many circuits each gradient costs.

To add a model, ansatz, embedding or loss, write a class and tag it. The test
suite finds it by walking the package. See :doc:`contributing`.

Where to go next
================

:doc:`tutorials/index` has three worked notebooks. Start with the
:doc:`VQC tutorial <tutorials/vqc>`. The :doc:`api_reference` has one page per
kind of object and one page per class.

.. toctree::
   :hidden:

   getting_started

.. toctree::
   :hidden:
   :caption: Tutorials
   :maxdepth: 1

   tutorials/index

.. toctree::
   :hidden:
   :caption: Reference
   :maxdepth: 2

   api_reference

.. toctree::
   :hidden:
   :caption: Project

   contributing
   changelog
