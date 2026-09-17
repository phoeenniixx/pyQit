==========
Embeddings
==========

.. currentmodule:: pyqit.core.embeddings

An embedding maps classical features onto the circuit. It also decides how the
:class:`~pyqit.DataModule` shapes those features. The model class picks the
embedding, so the model class controls input shaping.

.. code-block:: python

   from pyqit.core import AmplitudeEmbedding
   from pyqit.models import VQCClassifier

   model = VQCClassifier(n_qubits=4, encoder=AmplitudeEmbedding)

Available embeddings
====================

Each page gives the circuit, how many features it takes and how the
``DataModule`` prescales them.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   AngleEmbedding
   AmplitudeEmbedding
   IQPEmbedding
   ZZFeatureMap

How prescaling is chosen
========================

Every embedding carries a ``prescale`` tag, and ``setup()`` maps the tag's value
to a shaping function.

``"angle_pi"``
   Zero-pads to ``n_qubits`` features, then multiplies by pi. One feature per
   wire.

``"amplitude"``
   Zero-pads to ``2 ** n_qubits`` features, then L2-normalizes each row.

Input wider than the embedding takes raises instead of being truncated.

Prescaling is stateless and runs after normalization, which is stateful and fits
on the training split only. Nothing here is fitted, so no information leaks
between splits.

Adding an embedding
===================

Subclass :class:`BaseEmbedding`, implement ``forward`` and set the ``prescale``
tag. ``__init_subclass__`` copies the tag onto a ``PRESCALE`` class attribute
that ``setup()`` reads, so the tag is the whole registration. Then add the class
name to the list above. See :doc:`the contributing guide </contributing>`.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   BaseEmbedding

Related
=======

:doc:`datamodule` runs the prescaling these tags select, and :doc:`models`
chooses the embedding in the first place. :doc:`pipeline` enforces the same
width rules between stages.
