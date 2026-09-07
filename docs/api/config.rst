=============
Configuration
=============

.. currentmodule:: pyqit

Backend selection is global, not per object. :func:`set_backend` writes to a
context variable, and every object reads it once in its own ``__init__`` and
caches the answer.

.. code-block:: python

   import pyqit
   from pyqit.models import VQCClassifier

   pyqit.set_backend("torch")               # first
   model = VQCClassifier(n_qubits=4)        # then

Order matters and getting it wrong fails quietly. Setting the backend after you
build a model leaves that model on the old one.

:func:`set_backend` raises :class:`ImportError` when you ask for ``"torch"``
without torch installed. Every torch import in the package sits behind either
this call or a runtime type check, so the one guard covers them all and the
error names its own cause instead of surfacing later as a bare
``ModuleNotFoundError``.

What the backend changes
========================

Three things fork on it. The QNode is wrapped in a ``qml.qnn.TorchLayer`` or
stored as plain ``pnp`` arrays. Training runs through Lightning or through a
PennyLane optimizer loop. Loaders come from ``torch.utils.data`` or from an
internal NumPy loader.

Seeding
=======

:func:`set_seed` seeds NumPy, which covers PennyLane too because
``pennylane.numpy.random`` delegates to it, and seeds torch when it is
installed. :meth:`Trainer.fit <pyqit.core.Trainer.fit>` calls it before anything
stochastic runs.

Weights are drawn at construction, so reproducing them means seeding first:

.. code-block:: python

   pyqit.set_seed(42)
   model = VQCClassifier(n_qubits=4)

``Trainer(seed=...)`` alone covers training and diagnostics, not
initialisation. Note that this mutates global RNG state, the same contract as
Lightning's ``seed_everything``.

Related
=======

:doc:`models` and :doc:`trainer` both read the backend at construction, which is
why the order on this page matters. :doc:`datamodule` picks its loader from the
same setting.

.. autosummary::
   :nosignatures:

   set_backend
   get_backend
   set_seed

.. autofunction:: set_backend
.. autofunction:: get_backend
.. autofunction:: set_seed
