Quick Start
===========

Overview
--------

This quick start runs the full pipeline end to end:

1. Train Word2Vec and LightGBM models
2. Get a bundle recommendation via ``serve_bundle.py``

Why this order
--------------

``serve_bundle.py`` loads the trained model files from ``models/``.
The pipeline must complete before recommendations can be generated.

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

Prepare data
------------

Create the ``data/`` directory in the project root and place the three required
CSV files inside it:

.. code-block:: bash

   mkdir data

Then copy or move the following files into that folder:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - File
     - Description
   * - ``data/products_v2.csv``
     - Product catalogue (product IDs, names, categories, blocked flags)
   * - ``data/commerces.csv``
     - Kiosk/commerce master data
   * - ``data/2024-20250001_part_00-001.csv``
     - Order event data (the filename may differ; any ``*.csv`` matching the
       order export format is accepted)

.. note::

   Only one order CSV is required to get started. You can place additional order files
   in the same ``data/`` folder and pick them later in the process.

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

Step 2 — Get a bundle recommendation
-------------------------------------

Open ``serve_bundle.py`` and edit the ``__main__`` block at the bottom to use
a product ID and kiosk ID from your data:

.. code-block:: python

   # serve_bundle.py — __main__ block
   print(getSingleRec("<anchor_product_id>", "<kiosk_id>", topn=8, addDebugInfo=False))

Then run the script:

.. code-block:: bash

   python serve_bundle.py

The output lists the anchor product followed by a ranked table of recommended
product IDs and names:

.. code-block:: text

   Eingabeprodukt: 000295-999 | <product name>
    productid                          name     score
    000295-003             <product name>  0.92...
    ...

For batch recommendations (multiple anchor–kiosk pairs at once), use
``getMultiRec()`` with a Polars DataFrame that has columns ``anchor_pid`` and
``userid``.

Verification
------------

After running ``python serve_bundle.py``, a non-empty recommendation table
confirms that both models loaded correctly and inference is working.

Next steps
----------

- :doc:`pipeline` — detailed description of each training step
- :doc:`data_flow` — input files, intermediate artifacts, and outputs
- :doc:`design` — architecture and technology decisions
