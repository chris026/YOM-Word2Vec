Testing
=======

Overview
--------

The ``tests/`` directory contains three scripts for offline evaluation.
They sit outside the ZenML pipeline and are run manually after training.

.. list-table::
   :header-rows: 1

   * - Script
     - What it tests
     - Anchor strategy
   * - ``W2V_testbench.py``
     - Word2Vec embeddings in isolation
     - Every product in every basket
   * - ``lightGBM_testbench.py``
     - Full pipeline (Word2Vec + LightGBM ranker)
     - Every product in every basket
   * - ``model_test.py``
     - Full serving stack (``getMultiRec``)
     - One random product per order

When to run which script
------------------------

- **After retraining Word2Vec** — run ``W2V_testbench.py`` to measure how
  the raw embeddings changed.
- **After retraining LightGBM** — run ``lightGBM_testbench.py`` to measure
  the effect of the ranker on top of Word2Vec.
- **Before releasing a new model version** — run ``model_test.py`` to test
  the full serving stack end-to-end, including caching and the fallback
  logic in ``getMultiRec``.


W2V_testbench.py
----------------

Evaluates the Word2Vec model in isolation, without the LightGBM ranker.

**What it does**

For every qualifying order, every product in the basket is used as an
anchor. The model's top-K nearest neighbours are retrieved via
``model.wv.most_similar()``. The other basket products are the positives.
Anchors that are out-of-vocabulary (OOV) are skipped and counted
separately so the OOV rate is visible in the output.

Metrics are averaged over all anchors (including OOV), giving a
conservative estimate that penalises low vocabulary coverage.

**Required files**

- ``models/word2vec.model``
- ``data/test_4weeks.csv`` (or any ``.parquet`` with columns
  ``orderid``, ``productid``)

**Usage**

Edit the constants at the top of the file, then run::

   python tests/W2V_testbench.py

Key constants:

.. list-table::
   :header-rows: 1

   * - Constant
     - Default
     - Purpose
   * - ``MODEL_PATH``
     - ``models/word2vec.model``
     - Path to the trained Word2Vec model
   * - ``EVAL_ORDERS_PATH``
     - ``data/test_4weeks.csv``
     - Test order data (CSV or Parquet)
   * - ``KS``
     - ``[5, 10, 20, 50]``
     - Cutoff depths for metric computation
   * - ``RETRIEVAL_TOPK``
     - ``50``
     - Number of neighbours retrieved per anchor; must be >= max(KS)
   * - ``EVAL_MAX_ORDERS``
     - ``0``
     - Limit to N orders; ``0`` means all
   * - ``NUM_ANCHORS``
     - ``0``
     - Randomly subsample to N anchors; ``0`` means all

**Output**

Coverage block (OOV rate, skipped anchors) followed by a metrics table::

   K  HitRate     Recall      MRR         Precision   Positives
   5  0.312000    0.101000    0.154000    0.062400    0.312000
   10 0.421000    0.136000    0.163000    0.042100    0.421000
   ...

**Metrics**

- **HitRate@K** — share of anchors for which at least one positive
  appears in the top-K results.
- **Recall@K** — mean fraction of positives retrieved per anchor.
- **MRR@K** — Mean Reciprocal Rank; rewards placing the first hit high.
- **Precision@K** — mean fraction of top-K slots that contain a positive.


lightGBM_testbench.py
---------------------

Evaluates the full pipeline: Word2Vec for retrieval, LightGBM for
re-ranking. Calls ``recommend_candidates()`` from ``serve_bundle``
directly, bypassing the caching layer.

**What it does**

Like ``W2V_testbench.py``, every product in every basket is used as an
anchor. For each anchor, ``recommend_candidates()`` is called with the
store context (``userid``) and the full set of popularity signals. This
reflects real serving behaviour more closely than the W2V testbench.
Average per-anchor inference time is reported alongside the metrics.

Positives are restricted to products that are also in the Word2Vec
vocabulary; items outside the vocabulary cannot be returned by the ranker
and would inflate the miss count unfairly.

**Required files**

- ``models/word2vec.model``
- ``models/lgbm_ranker.txt``
- ``data/test_4weeks_short.csv`` (or any ``.parquet`` with columns
  ``orderid``, ``productid``, ``userid``)
- ``data/commerces.parquet``
- ``data/products_v2.parquet``
- ``artifacts/pop_global.parquet``
- ``artifacts/pop_store.parquet``
- ``artifacts/pop_region.parquet``
- ``artifacts/pop_subch.parquet``

**Usage**

Edit the constants at the top of the file, then run::

   python tests/lightGBM_testbench.py

Key constants (in addition to the shared ones from W2V_testbench):

.. list-table::
   :header-rows: 1

   * - Constant
     - Default
     - Purpose
   * - ``LGBM_MODEL_PATH``
     - ``models/lgbm_ranker.txt``
     - Path to the trained LightGBM ranker
   * - ``COMMERCES_PATH``
     - ``data/commerces.parquet``
     - Kiosk metadata used for store context features
   * - ``TOPN``
     - ``50``
     - Number of Word2Vec candidates passed to the ranker

**Output**

Coverage block with OOV rate and average inference time, followed by a
metrics table::

   K  Precision   Recall      F1          HitRate     MRR         MAP         NDCG    AvgTruePositives
   5  0.062000    0.101000    0.072000    0.312000    0.154000    0.095000    0.130000  0.310000
   ...

**Metrics**

Includes all metrics from ``W2V_testbench.py``, plus:

- **F1@K** — harmonic mean of Precision@K and Recall@K.
- **MAP@K** — Mean Average Precision; penalises positives found late.
- **NDCG@K** — Normalised Discounted Cumulative Gain; the primary metric
  used during LightGBM training.


model_test.py
-------------

End-to-end test of the full serving stack. Calls ``getMultiRec()`` from
``serve_bundle``, which includes result caching and the global-popularity
fallback for OOV anchors.

**What it does**

For each qualifying order, one product is chosen at random as the anchor.
The remaining basket products are the positives. All anchors are submitted
in a single batched call to ``getMultiRec``. This matches how the backend
calls the serving layer and exercises the batch deduplication and caching
logic that the other two testbenches bypass.

**Required files**

- ``models/word2vec.model``
- ``models/lgbm_ranker.txt``
- ``data/test_df_1m.parquet`` (Parquet with columns ``orderid``,
  ``productid``, ``userid``)
- All ``artifacts/pop_*.parquet`` files (loaded by ``serve_bundle``
  at startup)

**Usage**

Edit the constants at the top of the file, then run::

   python tests/model_test.py

Key constants:

.. list-table::
   :header-rows: 1

   * - Constant
     - Default
     - Purpose
   * - ``TEST_DATA_PATH``
     - ``data/test_df_1m.parquet``
     - Test order data (must be Parquet)
   * - ``KS``
     - ``[5, 10, 20]``
     - Cutoff depths for metric computation
   * - ``MAX_ORDERS``
     - ``0``
     - Limit to N orders; ``0`` means all
   * - ``RANDOM_SEED``
     - ``42``
     - Seed for reproducible anchor selection

**Output**

Configuration block followed by a metrics table::

   K  HitRate     Recall      MRR         Precision   Positives
   5  0.318000    0.106000    0.159000    0.063600    0.318000
   ...

A warning is printed when ``getMultiRec`` returns fewer items than the
largest K, since metrics for those cutoffs are based on shorter lists.
