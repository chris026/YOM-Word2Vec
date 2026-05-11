YOM Word2Vec Recommender
========================

This system recommends products that customers are likely to add to their
basket alongside an anchor product they have already selected. It is designed
for retail kiosks that operate offline and on mobile hardware, where low
latency and no network dependency are hard requirements.

The recommender combines two models trained on historical order data:
**Word2Vec** captures co-purchase patterns across all baskets and is used
for candidate retrieval; a **LightGBM ranker** re-ranks the candidates
using store context and popularity signals to personalise results at the
kiosk level.

Training is orchestrated with ZenML and runs locally. The trained models
are bundled into a Docker image and served via FastAPI on AWS Lambda.
Inference is a nearest-neighbour lookup followed by a single LightGBM
predict call and completes in milliseconds.


Architecture
------------

.. code-block:: text

   Training (ZenML, local)
     Raw CSV → Baskets → Word2Vec embeddings → LightGBM ranker
                                                       ↓
   Serving (FastAPI → AWS Lambda)
     GET /recommendations → Word2Vec lookup → LightGBM predict → Response


Pipeline
--------

The training pipeline runs in seven steps:

- **load_data** — loads order, product, and kiosk data from CSV; writes Parquet
- **clean_blocked_products** — removes blocked products and their orders
- **build_baskets** — groups order items into baskets (min. 2 items)
- **data_split** — splits baskets 80 / 20 into train and test sets
- **train_model** — trains Word2Vec (skip-gram, vector size 35, window 100)
- **ranker_training_pipeline_fast** — trains a LightGBM ranker on Word2Vec embeddings
- **test_model** — evaluates both models on the held-out test set


Backend API
-----------

The FastAPI backend exposes two endpoints:

- ``GET /recommendations?kioskId=<id>&anchorId=<id>&limit=<n>`` — single recommendation
- ``POST /recommendations/multi`` — batch recommendations for multiple anchors


Contents
--------

.. toctree::
   :maxdepth: 2

   quickstart
   data_flow
   pipeline
   features
   configuration
   design
   dashboard
   ab_test
   testing


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
