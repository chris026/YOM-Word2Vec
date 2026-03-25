# API Documentation

This document describes the HTTP API of the YOM Recommendation Backend.

The API provides endpoints for retrieving product recommendations based on:
- an anchor product (`anchor_id`)
- a kiosk/user identifier (`kiosk_id`)

## Base URL

The API is exposed via AWS API Gateway:

https://<api-id>.execute-api.eu-central-1.amazonaws.com

Replace `<api-id>` with your deployed API Gateway ID.

### Example

https://kxqjiw9kq5.execute-api.eu-central-1.amazonaws.com

## Authentication

Currently, no authentication is required to access the API.

## Endpoints

- `GET /health`
- `GET /recommendations`
- `POST /recommendations/multi`

## Health Check

### Endpoint

GET /health

### Description

Returns the health status of the service.

### Request

```bash
curl "https://<api-url>/health"
```

### Example Response

{
  "status": "ok"
}

## Single Recommendation

### Endpoint

GET /recommendations

### Description

Returns the top-N product recommendations for a single `(anchor_id, kiosk_id)` pair.

---

### Query Parameters

| Name       | Type    | Required | Description                  |
|------------|--------|----------|------------------------------|
| anchorId   | string | yes      | Anchor product ID            |
| kioskId    | string | yes      | Kiosk/user identifier        |
| limit      | int    | no       | Number of recommendations    |

---

### Example Request

```bash
curl "https://<api-url>/recommendations?anchorId=000295-999&kioskId=9077130ee9894b2d1e6d3341b341e006&limit=5"
```

### Example Response

[
  {
    "anchor_id": "000295-999",
    "kiosk_id": "9077130ee9894b2d1e6d3341b341e006",
    "product_id": "000332-040",
    "model_id": "christian_model_v1",
    "recommendation_date": "2026-03-23T12:55:01.377487Z"
  }
]

### Response Fields

| Field               | Type     | Description            |
| ------------------- | -------- | ---------------------- |
| anchor_id           | string   | Input anchor product   |
| kiosk_id            | string   | Input kiosk identifier |
| product_id          | string   | Recommended product    |
| model_id            | string   | Model identifier       |
| recommendation_date | datetime | Timestamp (UTC)        |

## Multi Recommendation

### Endpoint

POST /recommendations/multi

### Description

Returns recommendations for multiple `(anchor_id, kiosk_id)` pairs in a single request.

---

### Request Body

```json
[
  {
    "anchor_id": "string",
    "kiosk_id": "string"
  }
]
```

### Example Request

```bash
curl -X POST "https://<api-url>/recommendations/multi" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "anchor_id": "000295-003",
      "kiosk_id": "9077130ee9894b2d1e6d3341b341e006"
    },
    {
      "anchor_id": "000295-999",
      "kiosk_id": "9077130ee9894b2d1e6d3341b341e006"
    }
  ]'
```

### Example Response

[
  {
    "anchor_id": "000295-003",
    "kiosk_id": "9077130ee9894b2d1e6d3341b341e006",
    "recs": [
      "002302-004",
      "000295-010",
      "000617-001"
    ],
    "model_id": "christian_model_v1",
    "recommendation_date": "2026-03-23T15:02:17.567197Z"
  }
]

### Response Fields

| Field               | Type         | Description                    |
| ------------------- | ------------ | ------------------------------ |
| anchor_id           | string       | Input anchor product           |
| kiosk_id            | string       | Input kiosk identifier         |
| recs                | list[string] | Ranked product recommendations |
| model_id            | string       | Model identifier               |
| recommendation_date | datetime     | Timestamp (UTC)                |

## Error Handling

### HTTP Status Codes

| Code | Meaning |
|------|--------|
| 200  | Success |
| 400  | Invalid request |
| 422  | Validation error |
| 500  | Internal server error |
| 503  | Service unavailable (e.g. cold start) |

## Notes

- The first request to the AWS Lambda function may be slower due to cold start.
- Subsequent requests are faster due to in-memory caching.
- Recommendations are generated using a Word2Vec + LightGBM model.