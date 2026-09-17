============
Measurements
============

.. currentmodule:: pyqit.core

A measurement function turns the final quantum state into numbers the loss can
use. Pass one as ``measure_fn``, and name the wires with ``measure_wires``:

.. code-block:: python

   from pyqit.core import measure_probs
   from pyqit.models import VQCClassifier

   model = VQCClassifier(
       n_qubits=4,
       n_classes=2,
       measure_fn=measure_probs,
       measure_wires=[0],
   )

A model picks a default when you leave ``measure_fn`` unset, and its page says
which.

Available measurements
======================

.. autosummary::
   :toctree: generated/
   :nosignatures:

   measure_probs
   measure_expval_z
   measure_expval_x

Writing your own
================

A measurement function takes the list of wires and returns a PennyLane
measurement, or a tuple of them. The model calls it as the last step of the
circuit.

.. code-block:: python

   import pennylane as qml

   def measure_expval_y(wires):
       return tuple(qml.expval(qml.PauliY(w)) for w in wires)

Local and global cost
=====================

The choice also affects the barren-plateau baseline. Measuring fewer wires than
you have qubits counts as a local cost, with a floor of ``1 / 2 ** n_qubits``.
Measuring all of them counts as global, and the floor drops to
``1 / (3 * 4 ** (n_qubits - 1))``, which is far harder to clear.

Related
=======

:doc:`models` takes ``measure_fn`` and ``measure_wires``. :doc:`diagnostics`
explains how the local and global baselines differ, and
:doc:`losses` covers what the outputs then feed into.
