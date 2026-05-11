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

Bevor du ``run.py`` ausführst, entscheide welcher Modus zu deinem Datensatz passt.
Der Modus wird durch Kommentieren bzw. Auskommentieren einzelner Zeilen in
``run.py`` gewählt — es gibt keine einzelne Konfigurationsvariable.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Kriterium
     - Standard-Modus
     - Multi-Month-Modus
   * - Datenmenge
     - Wenige Monate in einer CSV-Datei
     - Mehrere Monate, extern vorgesplittet
   * - Datei laden
     - ``load_data()``
     - ``load_data_testTrain_seperated()``
   * - Basket-Building
     - ``build_baskets()``
     - ``build_baskets_monthly()`` *(bewahrt ``orderdt``)*
   * - Train/Test-Split
     - ``data_split()`` — zufällig 80 / 20
     - ``data_split_monthly()`` — letzte 2 Monate als Test

**Standard-Modus (aktiv per Default):** Keine Änderungen an ``run.py`` nötig.

**Multi-Month-Modus:** Folgende Zeilen in ``run.py`` anpassen:

.. code-block:: python

   # Daten laden — Zeile aktivieren, load_data() auskommentieren:
   data_path_train, data_path_test = load_data_testTrain_seperated()
   # data_path_train = load_data()

   # Baskets mit Datum bauen (statt build_baskets):
   train_df_path = word2vec_model.build_baskets_monthly(data_path_train)

   # Zeitbasierten Split aktivieren:
   train_df, test_df = word2vec_model.data_split_monthly(baskets_path)
   train_df_path, test_df_path = save_train_test_split(train_df, test_df)

   # clean_blocked_products auch auf Testdaten anwenden:
   data_path_test, _ = clean_blocked_products(data_path_test, products_path)

   # Evaluation am Ende aktivieren:
   metrics = test_model(test_df_path, W2Vmodel_path, LGM_model_path)

.. note::

   ``build_baskets_monthly`` ist im Multi-Month-Modus zwingend erforderlich,
   da ``data_split_monthly`` die ``orderdt``-Spalte benötigt, die nur diese
   Funktion bewahrt.

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
