YOM Word2Vec Recommender
========================

A product recommendation system using Word2Vec for candidate retrieval and LightGBM for ranking, orchestrated with ZenML and deployed via AWS Lambda.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   pipeline
   api
   design
   dashboard

Pipeline
--------

The training pipeline consists of the following steps:

- **load_data** – loads order and product data from Parquet/CSV files
- **build_baskets** – groups order items into baskets (min. 2 items per basket)
- **data_split** – splits baskets into train (80%) and test (20%) sets
- **train_model** – trains a Word2Vec model (skip-gram, vector size 35, window 100)
- **ranker_training_pipeline_fast** – trains a LightGBM ranking model on top of Word2Vec embeddings

Backend API
-----------

The FastAPI backend exposes two endpoints:

- ``GET /recommendations`` – single-product recommendations
- ``POST /recommendations/multi`` – batch recommendations

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
