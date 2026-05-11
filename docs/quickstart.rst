Quick Start
===========

Overview
--------

This quick start runs the full pipeline end to end:

1. Train Word2Vec and LightGBM models
2. Start the local recommendation API

Why this order
--------------

The serving layer loads both trained model files at startup.
The pipeline must complete before the API can serve recommendations.

Requirements
------------

- Python 3.12+
- Input files prepared:

  - ``data/commerces.csv``
  - ``data/products_v2.csv``
  - ``data/*.csv`` (order data)

Setup
-----

.. code-block:: bash

   python -m venv venv
   venv\Scripts\activate          # Windows
   # source venv/bin/activate     # macOS / Linux
   pip install -r requirements.txt

Choosing a pipeline
-------------------

``run.py`` contains four pipelines. Only one is active at a time; the others are
commented out. Uncomment the desired block and comment out the rest.

.. list-table::
   :header-rows: 1
   :widths: 5 30 35 30

   * - #
     - Name
     - When to use
     - Split
   * - 1
     - **No split** *(active by default)*
     - Exploration, small datasets, or when no test set is needed.
       Trains on all available data.
     - none
   * - 2
     - **External split**
     - Data arrives pre-split as two separate CSV files
       (``train_df_1m.csv`` / ``test_df_1m.csv``).
     - external (two CSVs)
   * - 3
     - **80/20 random split**
     - Single CSV file; random train/test split in Python.
       Maximises training volume.
     - ``data_split()`` — random 80 / 20
   * - 4
     - **Monthly split**
     - Single CSV file; temporally valid evaluation. The last two
       calendar months are held out as the test set.
     - ``data_split_monthly()`` — last 2 months

**Pipeline 1 — No split (active by default):** No changes to ``run.py`` needed.

**Pipeline 2 — External split:** Uncomment the block starting at
``load_data_testTrain_seperated()`` and comment out the pipeline 1 block:

.. code-block:: python

   # Activate pipeline 2:
   data_path_train, data_path_test = load_data_testTrain_seperated()
   # ...
   data_path_test, _ = clean_blocked_products(data_path_test, products_path)
   train_df_path = word2vec_model.build_baskets(data_path_train)

**Pipeline 3 — 80/20 random split:** Uncomment the block starting at ``data_split()``:

.. code-block:: python

   # Activate pipeline 3:
   data_path_train = load_data()
   train_df, test_df = word2vec_model.data_split(data_path_train)
   data_path_train, data_path_test = save_train_test_split(train_df, test_df)
   # ...
   train_df_path = word2vec_model.build_baskets(data_path_train)

**Pipeline 4 — Monthly split:** Uncomment the block starting at
``data_split_monthly()``. The split is applied to the raw order data *before*
basket building:

.. code-block:: python

   # Activate pipeline 4:
   data_path_train = load_data()
   data_path_train, products_path = clean_blocked_products(data_path_train, products_path)
   train_df, test_df = word2vec_model.data_split_monthly(data_path_train)
   data_path_train, data_path_test = save_train_test_split(train_df, test_df)
   train_df_path = word2vec_model.build_baskets(data_path_train)

Step 1 — Training
-----------------

.. code-block:: bash

   python run.py

Step-level caching via ZenML means unchanged steps are skipped on subsequent runs.

Main outputs:

- ``models/word2vec.model``
- ``models/lgbm_ranker.txt``
- ``artifacts/`` — intermediate Parquet files

Step 2 — Local API
------------------

.. code-block:: bash

   cd backend
   uvicorn src.app:app --reload

Endpoints:

- ``GET /health``
- ``GET /recommendations``
- ``POST /recommendations/multi``
- ``GET /docs`` — interactive Swagger UI

Recommendation request format:

.. code-block:: text

   GET /recommendations?kioskId=<kiosk_id>&anchorId=<anchor_product_id>&limit=<N>

Verification
------------

.. code-block:: bash

   curl http://localhost:8000/health
   curl "http://localhost:8000/recommendations?kioskId=<kiosk_id>&anchorId=<anchor_id>&limit=10"

Next steps
----------

- :doc:`pipeline` — detailed description of each training step
- :doc:`data_flow` — input files, intermediate artifacts, and outputs
- :doc:`design` — architecture and technology decisions
