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
   * - ``load_data``
     - Load orders, products, and commerces from CSV into Parquet
   * - ``clean_blocked_products``
     - Remove products flagged as blocked and the associated orders
   * - ``build_baskets``
     - Group order lines into per-order product baskets (min. 2 items)
   * - ``data_split``
     - Split baskets 80 / 20 into train and test sets
   * - ``train_model``
     - Train a Word2Vec skip-gram model on training baskets
   * - ``ranker_training_pipeline_fast``
     - Train a LightGBM ranker on top of Word2Vec candidate embeddings
   * - ``test_model``
     - Evaluate both models on the test set (Precision, Recall, NDCG)


Data Loading
------------

Raw order and product data is read from CSV files and converted to Parquet for
efficient downstream processing. Blocked products (``blocked=True``) are removed
before any modelling step, together with the corresponding order rows.

.. autofunction:: steps.load_data.load_data

.. autofunction:: steps.load_data.load_products

.. autofunction:: steps.load_data.load_commerces

.. autofunction:: steps.load_data.clean_blocked_products

.. autofunction:: steps.load_data.save_train_test_split


Basket Building & Data Split
-----------------------------

Orders are grouped by ``orderid`` to create baskets — lists of products that were
purchased together. Baskets with fewer than two items are discarded because
Word2Vec requires at least two tokens per sequence.

Two split strategies are available: a simple random 80 / 20 split and a
time-based split where the last two calendar months are held out for testing.

.. autofunction:: steps.train_Word2Vec.build_baskets

.. autofunction:: steps.train_Word2Vec.data_split

.. autofunction:: steps.train_Word2Vec.data_split_monthly


Word2Vec Training
-----------------

A skip-gram Word2Vec model (``sg=1``) is trained on the training baskets.
Each product ID is treated as a token; the embedding space captures co-purchase
similarity. Key hyperparameters: ``vector_size=35``, ``window=100``,
``min_count=2``, ``shrink_windows=False``.

The trained model is saved to ``models/word2vec.model``.

.. autofunction:: steps.train_Word2Vec.train_model

.. autofunction:: steps.train_Word2Vec.retrieve_candidates


LightGBM Ranker Training
-------------------------

The ranker pipeline takes the Word2Vec model and raw data as input and produces
a trained LightGBM model (``LambdaRank`` objective) saved to
``models/lgbm_ranker.txt``. It consists of six sub-steps:

.. autofunction:: steps.train_lightGBM.ranker_training_pipeline_fast

**Sub-steps:**

.. autofunction:: steps.train_lightGBM.prepare_data

.. autofunction:: steps.train_lightGBM.generate_candidates_fast_to_parquet

.. autofunction:: steps.train_lightGBM.generate_negatives_to_parquet

.. autofunction:: steps.train_lightGBM.train1

.. autofunction:: steps.train_lightGBM.train2

.. autofunction:: steps.train_lightGBM.train_ranker_from_files


Model Evaluation
----------------

Both models are evaluated on the held-out test baskets. For each test basket the
last item is treated as the positive label; the remaining items serve as the
anchor set. The evaluation computes Precision\@k, Recall\@k, F1\@k, and
NDCG\@k for the Word2Vec baseline, the LightGBM ranker, and a blended score.

.. autofunction:: steps.test_model.test_model
