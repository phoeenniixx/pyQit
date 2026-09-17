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

Available ansatzes
==================

Every ansatz implements a published circuit. Its page names the paper, the
gates in one layer and the shape of its weights.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   SELAnsatz
   BasicEntanglerAnsatz
   CNOTLadderAnsatz
   SimplifiedTwoDesignAnsatz
   RealAmplitudesAnsatz
   EfficientSU2Ansatz

What every ansatz provides
==========================

``get_weight_shapes()`` returns a dict from weight name to shape. The model
draws its initial weights from that dict, so the shapes are all a model needs
to know about an ansatz. ``build_circuit(weights)`` takes a dict with the same
keys and applies the gates.

.. code-block:: python

   SELAnsatz(n_qubits=4, n_layers=3).get_weight_shapes()
   # {"weights": (3, 4, 3)}

Depth comes from the model's ``n_layers``. More layers buy expressivity and cost
you gradient variance, which is what the barren-plateau check measures.

An ansatz that wraps another library's circuit names the package in its
``python_dependencies`` tag. Building one without the package raises an
``ImportError`` that names the extra to install.

Adding an ansatz
================

Subclass :class:`BaseAnsatz` and implement ``build_circuit``,
``get_weight_shapes`` and ``get_test_params()``. The ``object_type`` tag of
``"ansatz"`` is inherited, and the test suite enrolls the class by walking the
package. Then add its name to the list above. See
:doc:`the contributing guide </contributing>`.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   BaseAnsatz

Related
=======

A :doc:`model <models>` builds the ansatz you give it. Gradients shrink as depth
grows, so run :doc:`diagnostics` and read the
:doc:`barren-plateau tutorial </tutorials/barren_plateau>` before adding layers.
