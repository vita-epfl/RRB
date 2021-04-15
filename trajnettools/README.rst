Tools
=====

* summary table and plots: ``python -m trajnettools.summarize <dataset_files>``
* plot trajectories in a scene: ``python -m trajnettools.trajectories <dataset_file>``


APIs
====

* ``trajnettools.Reader``: class to read the dataset_file
* ``trajnettools.show``: module containing contexts for visualizing ``rows`` and ``paths``
* ``trajnettools.writers``: write a trajnet dataset file
* ``trajnettools.metrics``: contains ``average_l2()`` and ``final_l2()`` functions


Dataset
=======

Datasets are split into ``train``, ``val`` and ``test`` set.
Every line is a self contained JSON string (ndJSON_).

Scene:

.. code-block:: json

    {"scene": {"id": 266, "p": 254, "s": 10238, "e": 10358}}

Track:

.. code-block:: json

    {"track": {"f": 10238, "p": 248, "x": 13.2, "y": 5.85}}

with:

* ``id``: scene id
* ``p``: pedestrian id
* ``s``, ``e``: start and end frame id
* ``f``: frame id
* ``x``, ``y``: x- and y-coordinate in meters

Frame numbers are not recomputed. Rows are resampled to about
2.5 rows per second.


Dev
===

.. code-block:: sh

    pylint trajnettools
    pytest
    mypy trajnettools --disallow-untyped-defs


Dataset Summaries
=================

biwi_hotel:

+----------------------------------------------------+----------------------------------------------------+
| .. image:: docs/train/biwi_hotel.ndjson.theta.png  | .. image:: docs/train/biwi_hotel.ndjson.speed.png  |
+----------------------------------------------------+----------------------------------------------------+

crowds_students001:

+-----------------------------------------------------------+-----------------------------------------------------------+
| .. image:: docs/train/crowds_students001.ndjson.theta.png | .. image:: docs/train/crowds_students001.ndjson.speed.png |
+-----------------------------------------------------------+-----------------------------------------------------------+

crowds_students003:

+-----------------------------------------------------------+-----------------------------------------------------------+
| .. image:: docs/train/crowds_students003.ndjson.theta.png | .. image:: docs/train/crowds_students003.ndjson.speed.png |
+-----------------------------------------------------------+-----------------------------------------------------------+

crowds_zara02:

+-----------------------------------------------------------+-----------------------------------------------------------+
| .. image:: docs/train/crowds_zara02.ndjson.theta.png      | .. image:: docs/train/crowds_zara02.ndjson.speed.png      |
+-----------------------------------------------------------+-----------------------------------------------------------+

crowds_zara03:

+-----------------------------------------------------------+-----------------------------------------------------------+
| .. image:: docs/train/crowds_zara03.ndjson.theta.png      | .. image:: docs/train/crowds_zara03.ndjson.speed.png      |
+-----------------------------------------------------------+-----------------------------------------------------------+

dukemtmc:

+-----------------------------------------------------------+-----------------------------------------------------------+
| .. image:: docs/train/dukemtmc.ndjson.theta.png           | .. image:: docs/train/dukemtmc.ndjson.speed.png           |
+-----------------------------------------------------------+-----------------------------------------------------------+

syi:

+-----------------------------------------------------------+-----------------------------------------------------------+
| .. image:: docs/train/syi.ndjson.theta.png                | .. image:: docs/train/syi.ndjson.speed.png                |
+-----------------------------------------------------------+-----------------------------------------------------------+

wildtrack:

+-----------------------------------------------------------+-----------------------------------------------------------+
| .. image:: docs/train/wildtrack.ndjson.theta.png          | .. image:: docs/train/wildtrack.ndjson.speed.png          |
+-----------------------------------------------------------+-----------------------------------------------------------+


.. _ndJSON: http://ndjson.org/
