Data Flow
=========

Overview
--------

This page documents the input data, the intermediate artifacts produced at each
pipeline stage, and the model files consumed by the serving layer.

Input data
----------

.. list-table::
   :header-rows: 1

   * - File
     - Description
     - Key columns
   * - ``data/*.csv``
     - Order transaction data
     - ``orderid``, ``productid``, ``userid``, ``orderdt``, ``quantity``
   * - ``data/commerces.csv``
     - Kiosk metadata
     - ``userid``, ``channel``, ``subchannel``, ``region``, ``commune``, ``active``
   * - ``data/products_v2.csv``
     - Product catalog
     - ``productid``, ``name``, ``category``, ``subcategory``, ``blocked``

Pipeline flow
-------------

.. code-block:: text

   Raw CSV files
     → load_data()           → data/2024-20250001_part_00-001_short.parquet
     → load_commerces()      → data/commerces.parquet
     → load_products()       → data/products_v2.parquet
     → clean_blocked()       → blocked products and their orders removed
     → build_baskets()       → data/baskets.parquet

   Word2Vec stage:
     data/baskets.parquet
       → train_model()       → models/word2vec.model

   LightGBM stage:
     orders + commerces + products
       → prepare_data()           → artifacts/pop_global.parquet
                                     artifacts/pop_store.parquet
                                     artifacts/pop_region.parquet
                                     artifacts/pop_subch.parquet
                                     artifacts/baskets.parquet
       → generate_candidates()    → artifacts/candidates.parquet
                                     artifacts/neigh_df.csv
       → generate_negatives()     → artifacts/negatives.parquet
       → label_candidates()       → artifacts/train1.parquet
       → build_feature_matrix()   → artifacts/train.parquet
                                     artifacts/groups.npy
       → train_ranker()           → models/lgbm_ranker.txt

   Serving:
     models/word2vec.model + models/lgbm_ranker.txt
       → Docker image → AWS Lambda → API

Intermediate artifacts
----------------------

.. list-table::
   :header-rows: 1

   * - Artifact
     - Purpose
     - Key columns
   * - ``data/baskets.parquet``
     - Products grouped per order (used for Word2Vec training)
     - ``orderid``, ``productid`` (product list)
   * - ``data/train_df.parquet``
     - 80 % training baskets
     - ``orderid``, basket list
   * - ``data/test_df.parquet``
     - 20 % held-out test baskets
     - ``orderid``, basket list
   * - ``artifacts/pop_global.parquet``
     - Global purchase count per product
     - ``productid``, ``pop_global``
   * - ``artifacts/pop_store.parquet``
     - Per-kiosk purchase count
     - ``userid``, ``productid``, ``pop_store``
   * - ``artifacts/pop_region.parquet``
     - Per-region purchase count
     - ``region``, ``productid``, ``pop_region``
   * - ``artifacts/pop_subch.parquet``
     - Per-subchannel purchase count
     - ``subchannel``, ``productid``, ``pop_subch``
   * - ``artifacts/candidates.parquet``
     - Word2Vec nearest-neighbour pairs
     - ``orderid``, ``anchor``, ``candidate``, ``sim_item2vec``
   * - ``artifacts/negatives.parquet``
     - Hard negative samples (top-10 non-purchased)
     - ``orderid``, ``anchor``, ``candidate``, ``sim_item2vec``
   * - ``artifacts/train1.parquet``
     - Labelled candidate pairs
     - ``orderid``, ``anchor``, ``candidate``, ``sim_item2vec``, ``label``
   * - ``artifacts/train.parquet``
     - Full feature matrix for LightGBM (10 features + label)
     - all feature columns, ``label``, ``orderid``, ``anchor``
   * - ``artifacts/groups.npy``
     - Group sizes for LightGBM LambdaRank
     - int32 array, one entry per query

Serving artifacts
-----------------

**Model files**

- ``models/word2vec.model`` — Word2Vec model for nearest-neighbour retrieval
- ``models/lgbm_ranker.txt`` — LightGBM ranker for candidate re-ranking

**Lookup data**

- ``data/commerces.parquet`` — kiosk metadata (region, subchannel, channel, commune)
- ``data/products_v2.parquet`` — product catalog (category per product)
- ``artifacts/pop_global.parquet`` — global popularity lookup
- ``artifacts/pop_store.parquet`` — per-kiosk popularity lookup
- ``artifacts/pop_region.parquet`` — per-region popularity lookup
- ``artifacts/pop_subch.parquet`` — per-subchannel popularity lookup

Why this flow is used
---------------------

- Training and serving are fully decoupled: changing the model does not
  require changes to the serving code.
- Intermediate Parquet files allow individual pipeline steps to be re-run
  independently using ZenML step caching.
- Popularity artifacts are pre-computed once and joined at feature-matrix
  build time, avoiding repeated aggregation across pipeline runs.
