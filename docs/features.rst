Features
========

Overview
--------

The LightGBM ranker uses 10 features, combining a retrieval signal from
Word2Vec, popularity signals at four granularities, and kiosk context signals.

Feature source: ``steps/train_lightGBM.py``, function ``build_feature_matrix``.

Model feature set
-----------------

+--------------------+-------------------+--------------------------------------------------------------+
| Feature            | Type              | Description                                                  |
+====================+===================+==============================================================+
| ``sim_item2vec``   | Float32           | Word2Vec cosine similarity between anchor and candidate      |
+--------------------+-------------------+--------------------------------------------------------------+
| ``pop_global``     | UInt32            | Purchase count of candidate across all kiosks                |
+--------------------+-------------------+--------------------------------------------------------------+
| ``pop_subch``      | UInt16            | Purchase count of candidate within the kiosk's subchannel    |
+--------------------+-------------------+--------------------------------------------------------------+
| ``pop_region``     | UInt16            | Purchase count of candidate within the kiosk's region        |
+--------------------+-------------------+--------------------------------------------------------------+
| ``pop_store``      | UInt8             | Purchase count of candidate within this specific kiosk       |
+--------------------+-------------------+--------------------------------------------------------------+
| ``channel``        | Category (String) | Channel of the kiosk (from ``commerces.csv``)                |
+--------------------+-------------------+--------------------------------------------------------------+
| ``subchannel``     | Category (String) | Subchannel of the kiosk                                      |
+--------------------+-------------------+--------------------------------------------------------------+
| ``region``         | Category (String) | Geographic region of the kiosk                               |
+--------------------+-------------------+--------------------------------------------------------------+
| ``commune``        | Category (String) | Geographic commune of the kiosk                              |
+--------------------+-------------------+--------------------------------------------------------------+
| ``cand_category``  | Category (String) | Product category of the candidate item                       |
+--------------------+-------------------+--------------------------------------------------------------+

Signal groups
-------------

The 10 features cover three signal types:

**Co-purchase signal**

- ``sim_item2vec`` — the Word2Vec embedding similarity is the primary retrieval
  signal. It captures which products tend to be bought together across all baskets.

**Popularity and personalisation signals**

- ``pop_global``, ``pop_subch``, ``pop_region``, ``pop_store`` — popularity
  counters at increasing levels of granularity. ``pop_store`` provides kiosk-level
  personalisation; ``pop_global`` acts as a prior when local data is sparse.

**Context signals**

- ``channel``, ``subchannel``, ``region``, ``commune``, ``cand_category`` —
  categorical features describing the kiosk and the candidate product. These
  allow the ranker to personalise beyond co-purchase patterns when behaviour
  signals are sparse.

Processing rules
----------------

- The five categorical features (``channel``, ``subchannel``, ``region``,
  ``commune``, ``cand_category``) are passed directly to LightGBM using its
  native categorical support — no manual encoding is required.
- Popularity counters are pre-computed by ``prepare_data()`` and joined into
  the feature matrix by ``build_feature_matrix()``.
- The group sizes array (``artifacts/groups.npy``) encodes the number of
  candidates per query ``(orderid, anchor)`` and is required by LightGBM's
  LambdaRank objective.

Why these features
------------------

The feature set was designed to match the system's deployment constraints
(local, offline, mobile):

- No features require a database lookup or network call at serving time.
- All values can be derived from the order history and kiosk metadata
  available at training time.
- The popularity hierarchy (store → subchannel → region → global) provides
  a natural fallback: when a kiosk has few orders, broader signals still
  provide a meaningful prior.

References
----------

- :doc:`configuration` — hyperparameters for the LightGBM ranker
- :doc:`pipeline` — how the feature matrix is built step by step
- :doc:`data_flow` — source data and intermediate artifacts
