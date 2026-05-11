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
   * - ``data_split`` / ``data_split_monthly`` *(pipeline-abhängig)*
     - Split orders into train and test sets — nur in Pipeline 3 und 4
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

Zwei Ladefunktionen stehen zur Verfügung — die Wahl hängt von der aktiven Pipeline ab:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Pipeline
     - Funktion
     - Wann verwenden
   * - 1, 3, 4
     - ``load_data()``
     - Einzelne CSV-Datei mit allen Bestellungen.
   * - 2
     - ``load_data_testTrain_seperated()``
     - Daten kommen extern vorgesplittet als zwei separate Dateien.
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

``run.py`` enthält vier Pipelines, die sich darin unterscheiden, wie und ob die
Daten gesplittet werden. Die aktive Pipeline wird durch Kommentieren bzw.
Auskommentieren der jeweiligen Zeilen in ``run.py`` gewählt — es gibt keine
einzelne Konfigurationsvariable.

.. list-table::
   :header-rows: 1
   :widths: 5 20 30 25 20

   * - #
     - Name
     - Split-Funktion
     - Zeitpunkt des Splits
     - Basket-Funktion
   * - 1
     - Kein Split *(Default)*
     - —
     - —
     - ``build_baskets()``
   * - 2
     - Externer Split
     - extern (zwei CSVs)
     - vor dem Pipeline-Start
     - ``build_baskets()``
   * - 3
     - 80/20-Zufallssplit
     - ``data_split()``
     - nach ``clean_blocked_products``
     - ``build_baskets()``
   * - 4
     - Monatlicher Split
     - ``data_split_monthly()``
     - nach ``clean_blocked_products``, **vor** ``build_baskets``
     - ``build_baskets()``

.. note::

   In Pipeline 4 erfolgt der monatliche Split auf den Rohdaten (Bestellzeilen),
   bevor Baskets gebaut werden. Die letzten zwei Kalendermonate werden als
   Testset zurückgehalten. Alle vier Pipelines verwenden ``build_baskets()``.

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
