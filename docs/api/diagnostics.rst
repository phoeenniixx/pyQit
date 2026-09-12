===========
Diagnostics
===========

.. currentmodule:: pyqit.utils.diagnostic

A barren plateau is a circuit whose gradients vanish as it widens, which leaves
the optimizer nothing to follow. :func:`check_barren_plateau` samples gradients
at uniformly random weights and compares their variance against a theoretical
floor, so you can find out before spending the compute rather than after.

Run it as a pre-flight check:

.. code-block:: python

   import pyqit

   trainer = pyqit.Trainer(max_epochs=50, check_bp=True, bp_samples=200)
   history = trainer.fit(model, dm)

Or call it directly when you want the :class:`BPResult` without training:

.. code-block:: python

   from pyqit.utils.diagnostic import check_barren_plateau

   result = check_barren_plateau(model, dm, num_samples=200)

.. code-block:: text

              BP Diagnostic Result : BARREN PLATEAU
   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
   ┃ Metric / Layer              ┃    Value ┃         Status ┃
   ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
   │ Qubits                      │        4 │                │
   │ Samples                     │      200 │                │
   │ Circuit Executions          │      200 │                │
   │ Expected Variance           │ 1.56e-02 │       Baseline │
   │ Quantum Variance            │ 3.74e-03 │ BARREN PLATEAU │
   ├─────────────────────────────┼──────────┼────────────────┤
   │ Layer: main_circuit.weights │   0.240x │      ← plateau │
   └─────────────────────────────┴──────────┴────────────────┘

Reading the baseline
====================

The floor depends on how much of the circuit you measure. A local cost, meaning
fewer measured wires than qubits, uses ``1 / 2 ** n_qubits``. A global cost uses
``1 / (3 * 4 ** (n_qubits - 1))``, which shrinks much faster. Classifier models
scale the baseline further through their ``bp_scale_factor`` tag.

Each sample costs one gradient, and what a gradient costs depends on the device.
Under backprop it is one circuit execution. Under parameter-shift, which
shot-based devices and hardware use, it is ``1 + 2 * n_params``. The result
counts the executions the device ran and reports them as ``n_executions``, so
you know what ``bp_samples`` bought. The table renders through rich when it is
installed and falls back to ASCII when it is not.

Related
=======

:doc:`ansatzes` controls depth and :doc:`measurements` controls whether the cost
is local or global. Both move the verdict. :doc:`trainer` runs the check through
``check_bp=True``, and the
:doc:`barren-plateau tutorial </tutorials/barren_plateau>` shows a plateaued
circuit next to a healthy one.

.. autosummary::
   :nosignatures:

   check_barren_plateau
   BPResult

.. autofunction:: check_barren_plateau
.. autoclass:: BPResult
