API
===

This section documents the external API of the Word2Vec-based on-the-fly recommendation backend.

Its purpose is to describe how the backend can be accessed by clients and how recommendation requests are handled through a stable HTTP interface.

Overview
--------

The backend exposes a small HTTP API for recommendation serving.

The API is designed to provide:

- a simple integration interface for consumers
- stable request and response patterns
- compatibility with the overall project backend design

The API is exposed through Amazon API Gateway and backed by AWS Lambda.

Base URL
--------

The exact base URL depends on the deployed API Gateway configuration.

Example deployment used during the project:

::

   https://kxqjiw9kq5.execute-api.eu-central-1.amazonaws.com

Endpoints
---------

The backend exposes the following endpoints:

- ``GET /health``
- ``GET /recommendations``
- ``POST /recommendations/multi``

GET /health
-----------

This endpoint is used to check whether the backend service is reachable and running.

Typical request:

.. code-block:: bash

   curl "https://.../health"

Typical response:

.. code-block:: json

   {"status":"ok"}

Purpose:

- service availability check
- deployment validation
- operational monitoring

GET /recommendations
--------------------

This endpoint is used for a single recommendation request.

It accepts the following query parameters:

- ``anchorId``
- ``kioskId``
- ``limit``

Typical request:

.. code-block:: bash

   curl "https://.../recommendations?anchorId=000295-999&kioskId=9077130ee9894b2d1e6d3341b341e006&limit=5"

Typical response structure:

.. code-block:: json

   [
     {
       "anchor_id": "000295-999",
       "kiosk_id": "9077130ee9894b2d1e6d3341b341e006",
       "product_id": "000332-040",
       "model_id": "christian_model_v1",
       "recommendation_date": "2026-03-26T14:05:18.978737Z"
     }
   ]

Purpose:

- request recommendations for a single anchor and kiosk combination
- return ranked recommendation items
- support direct client-side integration

POST /recommendations/multi
---------------------------

This endpoint is used to request multiple recommendation sets in a single API call.

The request body contains a JSON array of request objects with:

- ``anchor_id``
- ``kiosk_id``

Typical request:

.. code-block:: bash

   curl -X POST "https://.../recommendations/multi" \
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

Typical response structure:

.. code-block:: json

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
       "recommendation_date": "2026-03-26T14:03:15.717366Z"
     }
   ]

Purpose:

- submit multiple recommendation queries in one request
- reduce request overhead for batch-like client usage
- provide a consistent multi-query interface

API Design Considerations
-------------------------

The API was designed to be lightweight and easy to consume.

Important design characteristics:

- small number of endpoints
- simple request structure
- JSON-based responses
- clear distinction between health checks, single requests, and multi requests

The API focuses on serving recommendation results and does not expose internal model details directly.

Response Consistency
--------------------

The backend returns structured recommendation results with explicit metadata, including:

- input identifiers
- returned product identifiers
- model identifier
- recommendation timestamp

This makes the responses easier to trace and compare across experiments or deployments.

Relation to the Overall Project
-------------------------------

Within the overall project, this API follows the same external structure as the other backend variant.

This was an intentional design decision because it allows:

- easier client-side integration
- consistent testing
- side-by-side comparison of different backend implementations
- support for A/B testing scenarios

Even though the internal serving logic differs, the external API remains stable.

Operational Usage
-----------------

The API can be used for:

- manual testing via ``curl``
- integration into client applications
- deployment validation
- service monitoring through the health endpoint

For routine operation, the most important checks are:

- verify ``/health`` after deployment
- test known requests against ``/recommendations``
- use ``/recommendations/multi`` for grouped request scenarios
