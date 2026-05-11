# YOM Word2Vec Recommender

A product bundle recommendation system for retail kiosks. Uses **Word2Vec** for candidate retrieval and **LightGBM** for re-ranking, orchestrated with ZenML and deployed via AWS Lambda.

**Key principle:** No online ML inference. Both models run locally and offline — inference is a nearest-neighbour lookup followed by a single predict call, completing in milliseconds.

## Architecture

```
Training Pipeline (ZenML)
  Raw CSV → Baskets → Word2Vec embeddings → LightGBM ranker
                                                      ↓
Serving (AWS Lambda / FastAPI)
  GET /recommendations → Word2Vec lookup → LightGBM predict → Response
```

## Documentation

Full documentation: [docs/](docs/)

- [Pipeline](docs/pipeline.rst) — training steps and hyperparameters
- [Design decisions](docs/design.rst) — architecture and technology choices
- [Quickstart](docs/quickstart.rst) — first run in 5 minutes
- [API](backend/API.md) — endpoint reference

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

## Run the training pipeline

```bash
python run.py
```

Outputs: `models/word2vec.model`, `models/lgbm_ranker.txt`

## Run the local API

```bash
cd backend
uvicorn src.app:app --reload
```

API available at `http://localhost:8000`. Endpoints:

- `GET /health`
- `GET /recommendations?kioskId=<id>&anchorId=<id>&limit=10`
- `POST /recommendations/multi`
