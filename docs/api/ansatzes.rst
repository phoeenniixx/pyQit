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

Every ansatz is a paper's circuit as its reference implementation defines it,
never redesigned here. Depth comes from the model's ``n_layers``. More layers
buy expressivity and cost you gradient variance, which is what the
barren-plateau check measures.

.. list-table::
   :header-rows: 1
   :widths: 22 30 26 22

   * - Ansatz
     - Circuit
     - Weights
     - Reference
   * - :class:`SELAnsatz`
     - PennyLane ``StronglyEntanglingLayers``: three rotations per qubit and a
       CNOT layer with a growing range.
     - ``(n_layers, n_qubits, 3)``
     - Schuld et al. 2020
   * - :class:`BasicEntanglerAnsatz`
     - PennyLane ``BasicEntanglerLayers``: one rotation per qubit (RX by
       default, any gate via ``rotation``) and a CNOT ring.
     - ``(n_layers, n_qubits)``
     - Schuld et al. 2020
   * - :class:`SimplifiedTwoDesignAnsatz`
     - PennyLane ``SimplifiedTwoDesign``: an RY layer, then CZ pairs each
       followed by RY. Needs at least two qubits. The circuit the local-cost
       trainability result was proved on.
     - ``initial_layer_weights (n_qubits,)`` and
       ``weights (n_layers, n_qubits - 1, 2)``
     - Cerezo et al. 2021
   * - :class:`RealAmplitudesAnsatz`
     - Qiskit ``real_amplitudes``: RY layers separated by CX entanglers,
       ``entanglement`` as Qiskit accepts it. Qiskit ML's ``VQC`` default.
     - Qiskit's flat ``θ`` vector, ``(n_qubits * (n_layers + 1),)``
     - Kandala et al. 2017
   * - :class:`EfficientSU2Ansatz`
     - Qiskit ``efficient_su2``: RY and RZ layers separated by CX entanglers,
       same options as above.
     - Qiskit's flat ``θ`` vector, ``(2 * n_qubits * (n_layers + 1),)``
     - Kandala et al. 2017

The two Qiskit ansatzes are Qiskit's own circuit objects converted through the
``pennylane-qiskit`` plugin. They need the ``qiskit`` extra. Together with
:class:`~pyqit.core.ZZFeatureMap` they let a Qiskit user reproduce their numbers
here.

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
   BasicEntanglerAnsatz
   CNOTLadderAnsatz
   SimplifiedTwoDesignAnsatz
   RealAmplitudesAnsatz
   EfficientSU2Ansatz

.. autoclass:: BaseAnsatz
.. autoclass:: SELAnsatz
.. autoclass:: BasicEntanglerAnsatz
.. autoclass:: CNOTLadderAnsatz
.. autoclass:: SimplifiedTwoDesignAnsatz
.. autoclass:: RealAmplitudesAnsatz
.. autoclass:: EfficientSU2Ansatz
