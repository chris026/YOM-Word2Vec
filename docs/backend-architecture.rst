Backend Architecture
====================

This section describes the backend architecture of the Word2Vec-based on-the-fly recommendation system.

Its purpose is to explain how the backend is structured, which technical components are involved, and how requests are processed at runtime.

Overview
--------

The backend is designed as a lightweight, containerized API service that exposes recommendation functionality through HTTP endpoints.

It is based on the following main components:

- FastAPI as the web framework
- Docker for containerization
- AWS Lambda as the execution environment
- Amazon API Gateway as the public access layer
- GitHub Actions for automated deployment

At a high level, the backend architecture can be described as:

::

   Client -> API Gateway -> AWS Lambda -> FastAPI application -> recommendation logic -> response

Core Design Idea
----------------

The backend follows an **on-the-fly serving approach**.

This means that recommendation results are computed directly during request processing instead of being loaded from a precomputed artifact.

High-level request flow:

::

   Request -> live recommendation computation -> response

This approach keeps the serving logic closely coupled to the deployed application code.

Application Layer
-----------------

The backend application is implemented with FastAPI.

FastAPI is responsible for:

- defining the API endpoints
- validating incoming requests
- calling the recommendation logic
- formatting the response

This provides a clean and structured interface between the external API and the internal recommendation logic.

Serving Logic
-------------

The Word2Vec backend generates recommendations dynamically at request time.

This means that the backend does not depend on a separate prediction file for serving.

Instead, the recommendation logic is executed when a request reaches the backend.

From an architectural perspective, this has the following consequences:

- no separate batch prediction artifact is required at runtime
- recommendation behavior is defined by the currently deployed code
- updates to recommendation logic are primarily deployment-driven

Container-Based Runtime
-----------------------

The backend is packaged as a Docker image and deployed to AWS Lambda.

This design was chosen to:

- ensure reproducibility of the runtime environment
- package dependencies together with the backend code
- simplify deployment through a standard container workflow

The Docker image contains:

- application source code
- Python dependencies
- Lambda entry point configuration

Execution Environment
---------------------

AWS Lambda is used as the runtime environment for the backend.

Lambda is responsible for:

- starting the backend application
- receiving requests forwarded by API Gateway
- running the recommendation logic
- returning responses

This provides a serverless execution model with low operational overhead.

Public Access Layer
-------------------

Amazon API Gateway exposes the backend through public HTTP endpoints.

From the client perspective, API Gateway is the entry point to the service.

It forwards incoming requests to the corresponding Lambda function and returns the backend response.

This makes the backend accessible without exposing internal implementation details directly.

Deployment Integration
----------------------

The backend architecture is integrated into an automated deployment process.

The main deployment path is:

::

   GitHub -> GitHub Actions -> Docker build -> Amazon ECR -> AWS Lambda -> API Gateway

This ensures that architecture, runtime, and deployment remain aligned.

Operational Characteristics
---------------------------

The most important characteristic of this backend architecture is that recommendation generation happens at runtime.

As a result:

- no separate serving artifact has to be maintained
- recommendation behavior changes when the deployed code changes
- backend logic and serving logic are tightly connected

This makes the architecture relatively direct and compact, but also means that runtime behavior depends strongly on the deployed application version.

Comparison to Batch-Based Serving
---------------------------------

Within the overall project, this backend represents the on-the-fly serving approach.

Compared to a batch-based backend, it differs mainly in one architectural aspect:

- the on-the-fly backend computes recommendations during the request
- the batch-based backend serves precomputed recommendation results

This document focuses only on the architecture of the Word2Vec-based on-the-fly backend.
