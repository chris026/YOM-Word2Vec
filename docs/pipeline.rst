Pipeline
========

The YOM Word2Vec Recommender is a two-stage ML pipeline orchestrated with ZenML.
In the first stage, a Word2Vec model learns product co-occurrence embeddings from
purchase baskets. In the second stage, a LightGBM ranker re-ranks the Word2Vec
candidates using contextual features (store metadata, popularity scores, product
categories).

Overview
--------

The following steps are executed in order:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Step
     - Description
   * - ``load_data`` / ``load_data_testTrain_seperated``
     - Load orders, products, and commerces from CSV into Parquet
   * - ``clean_blocked_products``
     - Remove products flagged as blocked and the associated orders
   * - ``data_split`` / ``data_split_monthly`` *(pipeline-dependent)*
     - Split orders into train and test sets — only in pipelines 3 and 4
   * - ``build_baskets``
     - Group order lines into per-order product baskets (min. 2 items)
   * - ``train_model``
     - Train a Word2Vec skip-gram model on training baskets
   * - ``ranker_training_pipeline_fast``
     - Train a LightGBM ranker on top of Word2Vec candidate embeddings


Data Loading
------------

Raw order and product data is read from CSV files and converted to Parquet for
efficient downstream processing. Blocked products (``blocked=True``) are removed
before any modelling step, together with the corresponding order rows.

Two loading functions are available — the choice depends on the active pipeline:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Pipeline
     - Function
     - When to use
   * - 1, 3, 4
     - ``load_data()``
     - Single CSV file containing all orders.
   * - 2
     - ``load_data_testTrain_seperated()``
     - Data arrives pre-split as two separate files.
       Expects ``data/train_df_1m.csv`` and ``data/test_df_1m.csv``
       and returns both paths.

.. note::

   The filenames ``train_df_1m.csv`` and ``test_df_1m.csv`` are the defaults,
   but any CSV files can be used. Update the file paths directly in ``run.py``
   to point to the desired files.

.. autofunction:: steps.load_data.load_data

.. autofunction:: steps.load_data.load_data_testTrain_seperated

.. autofunction:: steps.load_data.load_products

.. autofunction:: steps.load_data.load_commerces

.. autofunction:: steps.load_data.clean_blocked_products

.. autofunction:: steps.load_data.save_train_test_split


Basket Building & Data Split
-----------------------------

Orders are grouped by ``orderid`` to create baskets — lists of products that were
purchased together. Baskets with fewer than two items are discarded because
Word2Vec requires at least two tokens per sequence.

``run.py`` contains four pipelines that differ in how — and whether — the data
is split. The active pipeline is selected by commenting or uncommenting the
relevant lines in ``run.py`` — there is no single flag variable.

.. list-table::
   :header-rows: 1
   :widths: 5 20 35 40

   * - #
     - Name
     - Split function
     - Basket function
   * - 1
     - No split *(default)*
     - —
     - ``build_baskets()``
   * - 2
     - External split
     - external (two CSVs)
     - ``build_baskets()``
   * - 3
     - 80/20 random split
     - ``data_split()``
     - ``build_baskets()``
   * - 4
     - Monthly split
     - ``data_split_monthly()``
     - ``build_baskets()``

.. note::

   In all split pipelines (2, 3, 4), ``clean_blocked_products`` is applied
   **before** the split so that train and test are drawn from the same cleaned
   dataset. In pipeline 4 the last two calendar months are held out as the test
   set. All four pipelines use ``build_baskets()``.

.. autofunction:: steps.train_Word2Vec.build_baskets

.. autofunction:: steps.train_Word2Vec.build_baskets_monthly

.. autofunction:: steps.train_Word2Vec.data_split

.. autofunction:: steps.train_Word2Vec.data_split_monthly


Word2Vec Training
-----------------

A skip-gram Word2Vec model (``sg=1``) is trained on the training baskets.
Each product ID is treated as a token; the embedding space captures co-purchase
similarity. Key hyperparameters: ``vector_size=35``, ``window=100``,
``min_count=2``, ``shrink_windows=False``.

The trained model is saved to ``models/word2vec.model``.

**Why these hyperparameters?**

- ``window=100`` — covers the entire basket as context for every product.
  Within a basket there is no meaningful item order, so all products should
  mutually influence each other's embeddings equally.
- ``sg=1`` (skip-gram) — generalises better to rare products than CBOW because
  it predicts multiple context words from a single target word, generating more
  training signal per occurrence. CBOW works in the opposite direction: it
  averages multiple context words to predict one target word, which dilutes
  the signal for rare products.
- ``vector_size=35`` — chosen based on empirical tests with one month of
  training data across sizes 10, 20, 30, 35, 40, 50, and 100. Size 35
  performed best while remaining small enough to satisfy the mobile offline
  deployment constraint.
- ``min_count=2`` — products appearing only once carry no co-purchase signal
  and are excluded from the vocabulary.
- ``shrink_windows=False`` — disables window size randomisation; every product
  gets the full context window consistently.

.. note::

   The hyperparameters above are not exhaustive. Gensim's Word2Vec exposes
   additional parameters — such as ``epochs``, ``alpha`` (learning rate),
   ``negative`` (number of negative samples), and ``ns_exponent`` — that
   may further improve embedding quality and are worth exploring when more
   compute is available.

.. autofunction:: steps.train_Word2Vec.train_model

.. autofunction:: steps.train_Word2Vec.retrieve_candidates


LightGBM Ranker Training
-------------------------

The ranker pipeline takes the Word2Vec model and raw data as input and produces
a trained LightGBM model (``LambdaRank`` objective) saved to
``models/lgbm_ranker.txt``. It consists of six sub-steps.

**Why LambdaRank?**

The task is ranking candidates within each query group ``(orderid, anchor)``.
LambdaRank directly optimises a ranking metric (NDCG) rather than a
classification or regression loss, which is the correct formulation for a
re-ranker. See :doc:`design` for the full rationale.

**Why** ``topk=10`` **for negatives?**

Generating 10 hard negatives per anchor provides enough negative signal to
train the ranker without making the feature matrix prohibitively large.
Hard negatives — the top-10 Word2Vec neighbours that were *not* purchased —
are more informative than random negatives because they force the ranker to
distinguish truly co-purchased items from merely similar ones.

.. autofunction:: steps.train_lightGBM.ranker_training_pipeline_fast

**Sub-steps:**

.. autofunction:: steps.train_lightGBM.prepare_data

.. autofunction:: steps.train_lightGBM.generate_candidates_fast_to_parquet

.. autofunction:: steps.train_lightGBM.generate_negatives_to_parquet

.. autofunction:: steps.train_lightGBM.label_candidates

.. autofunction:: steps.train_lightGBM.build_feature_matrix

.. autofunction:: steps.train_lightGBM.train_ranker_from_files


Model Evaluation
----------------

Both models are evaluated on the held-out test baskets. For each test basket the
last item is treated as the positive label; the remaining items serve as the
anchor set. The evaluation computes Precision\@k, Recall\@k, F1\@k, and
NDCG\@k for the Word2Vec baseline, the LightGBM ranker, and a blended score.

Evaluation runs outside the ZenML pipeline via standalone scripts in ``tests/``.
See :doc:`testing` for usage instructions.
