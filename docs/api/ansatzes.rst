========
Ansatzes
========

.. currentmodule:: pyqit.ansatzes

An ansatz is the trainable part of the circuit. A model builds one from the
class you hand it, using the model's own qubit count.

.. code-block:: python

   from pyqit.ansatzes import SELAnsatz
   from pyqit.models import VQCClassifier

   model = VQCClassifier(n_qubits=4, n_layers=3, ansatz=SELAnsatz)

:class:`SELAnsatz` wraps PennyLane's strongly entangling layers. Depth comes from
the model's ``n_layers``. More layers buy expressivity and cost you gradient
variance, which is what the barren-plateau check measures.

To add your own, subclass :class:`BaseAnsatz` and give it an ``object_type`` tag
of ``"ansatz"``. Implement ``get_test_params()`` and the suite enrolls it
automatically. See :doc:`the contributing guide </contributing>`.

Related
=======

A :doc:`model <models>` builds the ansatz you give it. Depth is where gradients
go to die, so pair this with :doc:`diagnostics` and the
:doc:`barren-plateau tutorial </tutorials/barren_plateau>` before reaching for
more layers.

.. autosummary::
   :nosignatures:

   BaseAnsatz
   SELAnsatz

.. autoclass:: BaseAnsatz
.. autoclass:: SELAnsatz
