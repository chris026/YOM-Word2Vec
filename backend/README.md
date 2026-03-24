# YOM Recommendation Backend

This service provides product recommendations based on a given anchor product and kiosk/user.

It is built using:
- FastAPI (HTTP API)
- Word2Vec (candidate retrieval)
- LightGBM (ranking)
- AWS Lambda (deployment)

## Features

- Single product recommendations (`/recommendations`)
- Batch recommendations (`/recommendations/multi`)
- Fast inference using pre-trained models
- Deployable via Docker to AWS Lambda

## Local Development

### 1. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r backend/requirements.txt
```

### 3. Run the server

```bash
cd backend
python3 -m uvicorn src.app:app --reload --port 8080
```

### Example requests for single and multi recommendations

```bash
curl "http://127.0.0.1:8080/recommendations?anchorId=000295-999&kioskId=9077130ee9894b2d1e6d3341b341e006&limit=5"

curl -X POST "http://127.0.0.1:8080/recommendations/multi" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "anchor_id": "000295-003",
      "kiosk_id": "9077130ee9894b2d1e6d3341b341e006"
    }
  ]'
```

## Deployment

### Build Docker image

```bash
docker build -f backend/Dockerfile -t yom-backend .
```

### Push to AWS ECR

```bash
docker buildx build \
  --platform linux/amd64 \
  --provenance=false \
  -f backend/Dockerfile \
  -t <ECR-URI> \
  --push \
  .
```

## Environment Variables

The backend uses the following environment variables in AWS Lambda:

| Variable | Description | Example |
|----------|-------------|---------|
| `MODEL_ID` | Identifier of the recommendation model | `christian_model_v1` |
| `DEFAULT_RECOMMENDATION_LIMIT` | Default number of recommendations returned by the single recommendation endpoint| `30` |

## Notes

- The first request may be slow due to cold start.
- Models are loaded into memory on first invocation.
- Ensure all required files (`models/`, `artifacts/`, `data/`) are included in the Docker image.
