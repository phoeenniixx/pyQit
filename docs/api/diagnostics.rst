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

This is the result for an 8-qubit, 6-layer ``SELAnsatz`` classifier:

.. code-block:: text

              BP Diagnostic Result : BARREN PLATEAU
   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
   ┃ Metric / Layer              ┃     Value ┃         Status ┃
   ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
   │ Qubits                      │         8 │                │
   │ Samples                     │       200 │                │
   │ Circuit Executions          │       200 │                │
   │ Dead Parameters             │ 37 of 144 │       Excluded │
   │ Expected Variance           │  9.77e-04 │       Baseline │
   │ Quantum Variance            │  4.58e-04 │ BARREN PLATEAU │
   ├─────────────────────────────┼───────────┼────────────────┤
   │ Layer: main_circuit.weights │    0.469x │      ← plateau │
   └─────────────────────────────┴───────────┴────────────────┘

Reading the baseline
====================

The floor depends on how much of the circuit you measure. A local cost, meaning
fewer measured wires than qubits, uses ``1 / 2 ** n_qubits``. A global cost uses
``1 / (3 * 4 ** (n_qubits - 1))``, which shrinks much faster. Classifier models
scale the baseline further through their ``bp_scale_factor`` tag.

The local floor is about twice the gradient variance of a random circuit, which
is ``1 / (2 * (2 ** n_qubits + 1))`` for one Pauli expectation value. A model is
flagged when its variance is within a factor of two of that. The deep circuit
above scores 0.469x, which is where a random circuit lands.

Reading the verdict
===================

A flag says the circuit's gradients are close to a random circuit's, and those
shrink exponentially as the circuit widens. It does not say the model cannot
train at its current width. At 4 qubits a random circuit still has a gradient
standard deviation of about 0.09, and a flagged 4-qubit model can train well.
Read a flag at small width as a warning about widening the same circuit.

Some weights cannot influence the measured wires at all, such as one behind a
final rotation that commutes with the measurement. Their gradient is zero in
every sample. The check leaves them out of the variance and reports them in the
``Dead Parameters`` row, and as ``n_dead_parameters`` on the result. On a
shot-based device their gradient is shot noise, so they are not detected there.

The check evaluates gradients at one input row. Shallow circuits are sensitive
to it, and to zero-padded features in particular. Deep circuits are not.

Only circuit weights are drawn at random. Classical weights, such as a
regressor's output scale or the dense layers of a hybrid, are held at their
current values, so the reported variance is the circuit's and not the product
of the circuit's gradient with a random scale. Their own gradients are still
collected and reported as ``classical_variance``. A jointly trained
:class:`~pyqit.core.pipeline.QuantumPipeline` is sampled the same way, with the
floor taken from its one trainable quantum stage.

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
   :toctree: generated/
   :nosignatures:

   check_barren_plateau
   BPResult
