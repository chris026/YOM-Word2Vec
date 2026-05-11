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

Two loading functions are available depending on how much data is being processed:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Modus
     - Funktion
     - Wann verwenden
   * - Standard
     - ``load_data()``
     - Wenige Monate Daten in einer einzigen CSV-Datei
       (``data/2024-20250001_part_00-001_short.csv``).
   * - Multi-Month
     - ``load_data_testTrain_seperated()``
     - Mehrere Monate Daten, die bereits extern aufgeteilt wurden.
       Erwartet ``data/train_df_1m.csv`` und ``data/test_df_1m.csv``
       und gibt beide Pfade zurück.

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

Two strategies are available depending on the volume and time span of the data.
The active variant is selected by commenting or uncommenting the relevant lines
in ``run.py`` — there is no single flag variable:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Schritt
     - Standard-Modus (wenige Monate)
     - Multi-Month-Modus (mehrere Monate)
   * - Basket-Building
     - ``build_baskets()``
       Nur ``orderid`` und ``productid`` werden behalten.
     - ``build_baskets_monthly()``
       Bewahrt zusätzlich ``orderdt`` — zwingend erforderlich
       für den zeitbasierten Split.
   * - Train/Test-Split
     - ``data_split()``
       Zufälliger 80 / 20-Split. Benötigt keine Datumsspalte.
       Maximiert das Trainingsvolumen.
     - ``data_split_monthly()``
       Hält die letzten zwei Kalendermonate als Testset zurück.
       Evaluiert das Modell auf den aktuellsten Kaufmustern.
       Erfordert die ``orderdt``-Spalte aus ``build_baskets_monthly``.

.. note::

   Im Multi-Month-Modus muss **zwingend** ``build_baskets_monthly`` statt
   ``build_baskets`` verwendet werden, da ``data_split_monthly`` die
   ``orderdt``-Spalte für den zeitbasierten Split benötigt. Außerdem sollte
   ``clean_blocked_products`` auch auf den Testdaten-Pfad angewendet werden
   (die entsprechende Zeile ist in ``run.py`` auskommentiert), und der
   ``test_model``-Schritt am Ende des Skripts sollte aktiviert werden, um
   eine zeitlich valide Evaluation zu erhalten.

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
  it trains a dedicated prediction task per target word rather than averaging
  context vectors.
- ``vector_size=35`` — small enough to satisfy the mobile offline deployment
  constraint while capturing sufficient co-purchase signal.
- ``min_count=2`` — products appearing only once carry no co-purchase signal
  and are excluded from the vocabulary.
- ``shrink_windows=False`` — disables window size randomisation; every product
  gets the full context window consistently.

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

.. autofunction:: steps.test_model.test_model
