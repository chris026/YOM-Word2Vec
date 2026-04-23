Design Decisions
================

This page explains the architectural and technology choices behind the
YOM Word2Vec Recommender and the reasoning behind each decision.


Two-Stage Architecture
----------------------

The system is built as a two-stage retrieval–ranking pipeline:

1. **Candidate retrieval** — Word2Vec generates a shortlist of similar products
   based on co-purchase patterns.
2. **Re-ranking** — LightGBM re-orders the shortlist using contextual features
   such as store metadata and popularity scores.

This separation keeps retrieval fast (a single vector lookup) while allowing
the ranker to personalise results without having to score the entire product
catalogue. It also enables a direct **A/B comparison** between the Market
Basket Analysis (MBA) baseline system and the Word2Vec approach, and makes
a controlled A/B test in production straightforward to set up.


Stage 1 — Word2Vec for Candidate Retrieval
------------------------------------------

**Why Word2Vec over the MBA baseline?**

The project started with a Market Basket Analysis system as the baseline
recommender. Word2Vec was introduced as an improved second system: instead
of relying on raw co-occurrence counts, it learns a dense embedding space
where products that are frequently bought together end up close to each
other. This produces richer similarity signals and generalises better to
products with sparse purchase histories.

**Treating baskets as sentences**

In natural language processing, Word2Vec learns from word sequences
(sentences). Here, each purchase basket is treated as a sentence and each
product ID as a token. The model therefore learns that products occurring
in the same basket are semantically related — analogous to words that
appear in the same context.

**Why ``window=100``?**

The ``window`` parameter controls how many neighbouring tokens are
considered context for a given token. A window of 100 effectively means
the *entire basket* is context for every product in it, regardless of the
order in which items appear. This is intentional: within a basket there is
no meaningful sequence, so all products should mutually influence each
other's embeddings equally.

**Deployment fit**

A trained Word2Vec model is a compact binary file. Candidate retrieval is
a single nearest-neighbour lookup in an in-memory vector table — no network
call, no GPU, no external service required. This matches the project's
deployment constraint of running entirely offline on a mobile device.


Stage 2 — LightGBM for Re-Ranking
----------------------------------

Word2Vec similarity alone does not capture context: the same product pair
may be highly relevant for one store in one region but irrelevant for
another. The LightGBM ranker closes this gap by combining the Word2Vec
similarity score with contextual features:

- Store-level metadata (channel, subchannel, region, commune)
- Product category of the candidate
- Popularity counts at global, store, region, and subchannel level

**Why LightGBM?**

- **LambdaRank objective** — directly optimises a ranking metric (NDCG)
  rather than a classification or regression loss, which is the right
  formulation for a recommender re-ranker.
- **Native categorical support** — features like channel, region, and
  product category are used as-is without manual one-hot encoding.
- **Fast inference** — prediction is a series of tree lookups with no
  matrix multiplications and no GPU requirement. Latency is negligible
  even on low-end hardware.
- **Small model size** — the model is saved as a plain text file, making
  it easy to bundle with a mobile or embedded application.
- **Wide industry adoption** — LightGBM is a proven choice for
  learning-to-rank tasks in production recommender systems.


Deployment Constraints
----------------------

All inference must run **locally and offline** — including on a mobile
device — with low latency and no internet connection. This ruled out
approaches that require:

- A remote model-serving endpoint
- GPU acceleration
- Large neural network architectures (high memory footprint, slow CPU inference)

Both Word2Vec (in-memory vector table) and LightGBM (tree lookup) satisfy
these constraints. The combined inference path is a nearest-neighbour
lookup followed by a single ``predict`` call, which completes in
milliseconds on commodity hardware.


Technology Choices
------------------

**ZenML**

ZenML was chosen as the pipeline orchestration framework as it is used in
the accompanying university course. Beyond familiarity, it provides:

- Step-level caching to avoid re-running expensive steps during development
- Reproducible, versioned pipeline runs
- Clear separation between pipeline logic and step implementation

**Polars instead of Pandas**

The training data comprises large Parquet files that exceed practical Pandas
memory limits. Polars was chosen because:

- Its **lazy evaluation engine** processes queries in a streaming fashion,
  keeping memory usage bounded regardless of dataset size.
- It is **multi-threaded by default**, making full use of available CPU cores.
- The **Polars query optimizer** rewrites and fuses operations before
  execution, reducing redundant I/O.

Pandas is only used where a library interface strictly requires it
(e.g. passing a DataFrame to LightGBM).