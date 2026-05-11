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

Pipeline-Modus wählen
---------------------

``run.py`` enthält vier Pipelines. Immer nur eine ist aktiv; die anderen sind
auskommentiert. Den gewünschten Block auskommentieren und den Rest aktivieren.

.. list-table::
   :header-rows: 1
   :widths: 5 30 35 30

   * - #
     - Name
     - Wann verwenden
     - Split
   * - 1
     - **Kein Split** *(aktiv per Default)*
     - Exploration, kleine Datasets oder wenn kein Testset benötigt wird.
       Trainiert auf allen Daten.
     - keiner
   * - 2
     - **Externer Split**
     - Daten kommen extern vorgesplittet als zwei separate CSV-Dateien
       (``train_df_1m.csv`` / ``test_df_1m.csv``).
     - extern (zwei CSVs)
   * - 3
     - **80/20-Zufallssplit**
     - Einzelne CSV-Datei; zufällige Train/Test-Aufteilung in Python.
       Maximiert das Trainingsvolumen.
     - ``data_split()`` — zufällig 80 / 20
   * - 4
     - **Monatlicher Split**
     - Einzelne CSV-Datei; zeitlich valide Evaluation. Die letzten zwei
       Kalendermonate werden als Testset zurückgehalten.
     - ``data_split_monthly()`` — letzte 2 Monate

**Pipeline 1 — Kein Split (aktiv per Default):** Keine Änderungen an ``run.py`` nötig.

**Pipeline 2 — Externer Split:** Den Block ab ``load_data_testTrain_seperated()``
einkommentieren und den Pipeline-1-Block auskommentieren:

.. code-block:: python

   # Pipeline 2 aktivieren:
   data_path_train, data_path_test = load_data_testTrain_seperated()
   # ...
   data_path_test, _ = clean_blocked_products(data_path_test, products_path)
   train_df_path = word2vec_model.build_baskets(data_path_train)

**Pipeline 3 — 80/20-Zufallssplit:** Den Block ab ``data_split()`` einkommentieren:

.. code-block:: python

   # Pipeline 3 aktivieren:
   data_path_train = load_data()
   train_df, test_df = word2vec_model.data_split(data_path_train)
   data_path_train, data_path_test = save_train_test_split(train_df, test_df)
   # ...
   train_df_path = word2vec_model.build_baskets(data_path_train)

**Pipeline 4 — Monatlicher Split:** Den Block ab ``data_split_monthly()``
einkommentieren. Der Split erfolgt auf den Rohdaten *vor* dem Basket-Building:

.. code-block:: python

   # Pipeline 4 aktivieren:
   data_path_train = load_data()
   data_path_train, products_path = clean_blocked_products(data_path_train, products_path)
   train_df, test_df = word2vec_model.data_split_monthly(data_path_train)
   data_path_train, data_path_test = save_train_test_split(train_df, test_df)
   train_df_path = word2vec_model.build_baskets(data_path_train)

Step 1 — Training
-----------------

.. code-block:: bash

   python run.py

The pipeline executes seven steps in order (load → clean → baskets → split →
Word2Vec → LightGBM → evaluate). Step-level caching via ZenML means unchanged
steps are skipped on subsequent runs.

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
