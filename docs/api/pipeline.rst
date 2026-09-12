=========
Pipelines
=========

.. currentmodule:: pyqit.core

:class:`QuantumPipeline` composes several models into one object that still
behaves like a model, so the same :class:`~pyqit.Trainer` fits, evaluates and
predicts it. Two things are configurable and they answer different questions.
``mode`` decides how stages relate to each other. ``fit_mode`` decides how they
get trained.

.. code-block:: python

   import pyqit
   from pyqit.core import QuantumPipeline
   from pyqit.models import VQCClassifier

   pipe = QuantumPipeline(
       [
           ("encode", VQCClassifier(n_qubits=4, n_layers=1)),
           ("head", VQCClassifier(n_qubits=4, n_layers=2)),
       ],
       mode="sequential",
   )
   trainer = pyqit.Trainer(max_epochs=20)
   trainer.fit(pipe, dm)            # one TrainingHistory per trained stage
   trainer.test(pipe, dm)           # {"test_loss": ..., "test_acc": ...}
   preds = trainer.predict(pipe, dm)
   preds = trainer.predict(pipe, dm.for_prediction(X_new))   # new raw rows

Steps take either ``(name, model)`` tuples or :class:`PipelineStage` objects. A
bare model gets its class name as the stage name.

Sequential mode
===============

Each stage feeds the next. Stage one's output becomes stage two's input, and the
last stage's output is the pipeline's.

Fitting materializes the intermediate data rather than recomputing it. Once a
stage is fitted, the pipeline runs its ``forward`` across the whole split and
rebuilds a datamodule from the result, so upstream stages run once per fit
instead of once per batch. That is a large saving on a quantum circuit and the
reason sequential fitting is practical at all.

One consequence is worth knowing before you debug it. The rebuilt intermediate
data is unprescaled: the pipeline prescales each stage's input itself, exactly
once, so a one-column output is zero-padded up to the next stage's width and an
output wider than that stage accepts raises instead of being truncated.

Ensemble mode
=============

Every stage sees the same input and the outputs combine. Stages are independent,
so nothing flows between them.

Because they all receive the same ``X``, they must agree on input width and
embedding. The pipeline checks this before fitting and raises if they disagree.
``input_slice`` is rejected outright in this mode, since per-stage column views
would contradict feeding everyone the same input. Use sequential mode if you
need that.

``aggregation`` decides how the outputs combine:

``"mean"``
   Averages the stage outputs. The default, and the right choice for
   probabilities.

``"vote"``
   Rounds each stage's output to a class label and takes the majority. Discards
   confidence, so ties break toward the lower label.

callable
   Receives the list of raw stage outputs and returns whatever you want.

How stages get trained
======================

``fit_mode`` is set on the pipeline and applies to sequential pipelines only.
Ensemble pipelines train each trainable stage independently on the same data, so
there is nothing to sequence. Every trainable stage trains under the one
:class:`~pyqit.Trainer` handed to ``Trainer.fit``.

``"sequential_greedy"`` (default)
   Trains each stage in turn against the data produced by the stages before it.
   Every stage may learn.

``"frozen_backbone"``
   Trains only the final stage. Every non-final stage must have
   ``trainable=False``, and the pipeline raises if one does not. Use this when
   the earlier stages are a pretrained feature map and you only want a new head.

Per-stage options
=================

:class:`PipelineStage` carries the flags that ``(name, model)`` tuples cannot:

``trainable``
   Set ``False`` to freeze a stage. Required on non-final stages under
   ``frozen_backbone``.

``passthrough``
   Concatenates the stage's input onto its output, so the next stage sees both.
   Useful when a stage refines features rather than replacing them, but it
   widens the output, which must still fit the next stage's embedding.

``input_slice``
   Feeds only these input columns to the stage. Sequential mode only.

.. code-block:: python

   from pyqit.core import PipelineStage, QuantumPipeline

   pipe = QuantumPipeline(
       [PipelineStage(backbone, trainable=False), PipelineStage(head)],
       fit_mode="frozen_backbone",
   )
   pyqit.Trainer(max_epochs=20).fit(pipe, dm)

Input width
===========

Each stage's input is prescaled for its embedding on every path, whether you
arrive through ``forward``, ``transform``, ``fit`` or ``predict``. An
amplitude-encoded stage takes up to ``2 ** n_qubits`` features, everything else
up to one per wire; narrower input is zero-padded and wider input raises.

Related
=======

Stages hold :doc:`models <models>`, and :doc:`embeddings` explains the width
rules. :doc:`trainer` covers the :class:`Trainer` that fits the pipeline. The
second half of the :doc:`VQC tutorial </tutorials/vqc>` builds a
frozen-backbone pipeline end to end.

.. autosummary::
   :nosignatures:

   QuantumPipeline
   PipelineStage

.. autoclass:: QuantumPipeline
.. autoclass:: PipelineStage
