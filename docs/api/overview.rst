===============
Object overview
===============

Every public object with its tags. Type in the box to narrow the table: each
word must appear somewhere in the row, so ``model hybrid`` finds the hybrid
models and ``regressor`` finds everything that predicts values.
Inspired from ``sktime`` tag system: https://www.sktime.net/models/

The same search from code:

.. code-block:: python

   from pyqit.base import all_objects, object_overview

   all_objects("model", filter_tags={"model_type": "hybrid"})
   object_overview()   # every row of the table below, as dicts

Each tag's meaning is in ``pyqit/base/_tags.py``.

.. object-overview::
