Configuration
=============

Overview
--------

Model hyperparameters and pipeline settings are defined directly in the
training step files under ``steps/``. This page consolidates all configurable
parameters in one place.

Word2Vec hyperparameters
------------------------

Configured in ``steps/train_Word2Vec.py``, function ``train_model``.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Value
     - Purpose
   * - ``vector_size``
     - ``35``
     - Embedding dimension. Small enough for mobile deployment constraints.
   * - ``window``
     - ``100``
     - Context window. Set to 100 so the entire basket is context for every product.
   * - ``sg``
     - ``1``
     - Training algorithm. ``1`` = skip-gram; generalises better to rare products than CBOW (``0``).
   * - ``min_count``
     - ``2``
     - Minimum product occurrences to be included in vocabulary.
   * - ``shrink_windows``
     - ``False``
     - Disables window size randomisation; every product gets the full context window.

LightGBM hyperparameters
------------------------

Configured in ``steps/train_lightGBM.py``, function ``train_ranker_from_files``.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Value
     - Purpose
   * - ``objective``
     - ``lambdarank``
     - Directly optimises a ranking metric (NDCG) instead of classification loss.
   * - ``metric``
     - ``ndcg``
     - Evaluation metric during training, computed at positions 5 and 10.
   * - ``learning_rate``
     - ``0.05``
     - Step size for gradient updates.
   * - ``subsample``
     - ``0.8``
     - Fraction of rows sampled per tree. Reduces overfitting.
   * - ``colsample_bytree``
     - ``0.8``
     - Fraction of features sampled per tree. Reduces overfitting.
   * - ``random_state``
     - ``42``
     - Seed for reproducibility.

Pipeline parameters
-------------------

Configured in ``steps/train_Word2Vec.py`` and ``steps/train_lightGBM.py``.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Value
     - Purpose
   * - Train / test split
     - ``80 / 20``
     - Fraction of baskets used for training vs. evaluation.
   * - Minimum basket size
     - ``2``
     - Baskets with fewer than 2 items are discarded before Word2Vec training.
   * - ``topk`` (negatives)
     - ``10``
     - Number of hard negative candidates sampled per anchor during LightGBM training.

Evaluation settings
-------------------

Configured in ``steps/test_model.py``.

.. list-table::
   :header-rows: 1

   * - Setting
     - Value
     - Purpose
   * - Evaluation metrics
     - Precision\@k, Recall\@k, F1\@k, NDCG\@k
     - Ranking quality metrics computed on held-out test baskets.
   * - Positive label strategy
     - Last item in basket
     - The last item of each test basket is treated as the positive label; remaining items serve as anchors.

Output files
------------

.. list-table::
   :header-rows: 1

   * - File
     - Content
   * - ``models/word2vec.model``
     - Trained Word2Vec embeddings (gensim format)
   * - ``models/lgbm_ranker.txt``
     - Trained LightGBM ranker (plain-text LightGBM format)
