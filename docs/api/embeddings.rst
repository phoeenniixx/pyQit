==========
Embeddings
==========

.. currentmodule:: pyqit.core.embeddings

An embedding maps classical features onto the circuit. It also decides how the
:class:`~pyqit.DataModule` shapes those features, which is the part that
surprises people: the model class picks the embedding, so the model class
controls input shaping.

.. code-block:: python

   from pyqit.core import AmplitudeEmbedding
   from pyqit.models import VQCClassifier

   model = VQCClassifier(n_qubits=4, encoder=AmplitudeEmbedding)

How prescaling is chosen
========================

Every embedding carries a ``prescale`` tag, which ``setup()`` maps to a shaping
function:

``AngleEmbedding``
   Pads or truncates to ``n_qubits`` features, then multiplies by pi. One
   feature per wire.

``HadamardAngleEmbedding``
   One feature per wire, multiplied by pi / 2. The encoding of Mari et al.
   (2020), a Hadamard layer followed by RY.

``AmplitudeEmbedding``
   Pads to ``2 ** n_qubits`` features, then L2-normalizes. Four qubits carry
   sixteen features, so this is the option for wide inputs.

``IQPEmbedding``
   One feature per wire, like the angle case.

``ZZFeatureMap``
   One feature per wire, like the angle case. The Havlicek et al. (2019) map
   that Qiskit's ``VQC`` uses; not the same circuit as ``IQPEmbedding``.

Prescaling is stateless and runs after normalization, which is stateful and fits
on the training split only. Nothing here is fitted, so no information leaks
between splits.

Adding an embedding
===================

Subclass :class:`BaseEmbedding` and set the ``prescale`` tag. ``__init_subclass__``
copies it onto a ``PRESCALE`` class attribute that ``setup()`` maps to a shaping
function, so the tag is the whole registration. See :doc:`the contributing guide </contributing>`.

Related
=======

:doc:`datamodule` runs the prescaling these tags select, and :doc:`models`
chooses the embedding in the first place. :doc:`pipeline` enforces the same
width rules between stages.

.. autosummary::
   :nosignatures:

   BaseEmbedding
   AngleEmbedding
   HadamardAngleEmbedding
   AmplitudeEmbedding
   IQPEmbedding
   ZZFeatureMap

.. autoclass:: BaseEmbedding
.. autoclass:: AngleEmbedding
.. autoclass:: HadamardAngleEmbedding
.. autoclass:: AmplitudeEmbedding
.. autoclass:: IQPEmbedding
.. autoclass:: ZZFeatureMap
