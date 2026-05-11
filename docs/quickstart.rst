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
