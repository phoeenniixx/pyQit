========
Datasets
========

.. currentmodule:: pyqit.datasets

Benchmark datasets, so an experiment needs one import. Every function returns
``X, y`` as numpy arrays with class labels in ``0..k-1``, ready for a
:doc:`DataModule <datamodule>`. Nothing touches the network: the ``make_*``
functions generate their rows at call time, and ``load_iris`` and ``load_wine``
read the files that ship inside scikit-learn.

.. code-block:: python

   import pyqit
   from pyqit.datasets import make_two_curves

   X, y = make_two_curves(n_samples=300, n_features=4, seed=0)
   dm = pyqit.DataModule(X, y, normalize="minmax", batch_size=16)

Seeding
=======

Each generator takes ``seed``. Leave it ``None`` and the rows come from numpy's
global state, which :func:`pyqit.set_seed` seeds, so one call at the top of a
script covers the data, the weights and the training loop. Pass an integer and
the set is the same whatever the global state.

Quantum benchmarks
==================

``make_linearly_separable``, ``make_two_curves``, ``make_hidden_manifold`` and
``make_hyperplanes_parity`` are the classification benchmarks of Bowles, Ahmed
and Schuld, *Better than classical? The subtle art of benchmarking quantum
machine learning models* (2024). Each takes ``n_features`` so it maps straight
onto ``n_qubits``. They are reimplemented from the authors' ``qml-benchmarks``
package; the paper's own pre-generated copies are on PennyLane's datasets site.

Available datasets
==================

.. autosummary::
   :toctree: generated/
   :nosignatures:

   make_linearly_separable
   make_two_curves
   make_hidden_manifold
   make_hyperplanes_parity
   make_moons
   make_circles
   make_sine
   load_iris
   load_wine

Related
=======

:doc:`datamodule` is what the arrays go into. The :doc:`VQC tutorial
</tutorials/vqc>` trains on moons and on the sine curve.
