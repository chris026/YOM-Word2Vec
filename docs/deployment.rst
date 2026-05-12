Deployment
==========

This section documents the deployment and infrastructure setup of the Word2Vec-based on-the-fly recommendation backend.

Its purpose is to provide a practical technical reference for understanding how the backend is deployed, exposed, configured, and maintained in a production-oriented environment.

Overview
--------

The Word2Vec backend is deployed as a containerized FastAPI service on AWS Lambda.

The deployment setup is based on the following components:

- GitHub for source code management
- GitHub Actions for CI/CD
- Docker for containerization
- Amazon ECR for image storage
- AWS Lambda for backend execution
- Amazon API Gateway for public HTTP access

At a high level, the deployment flow is:

::

   GitHub -> GitHub Actions -> Docker -> Amazon ECR -> AWS Lambda -> API Gateway

Deployment Goal
---------------

The deployment architecture was designed to achieve the following goals:

- reproducible backend deployment
- lightweight serverless operation
- public API accessibility
- minimal manual deployment effort
- easy integration into YOM's infrastructure

Backend Runtime
---------------

The backend is implemented using FastAPI and deployed as a Lambda-compatible Docker container.

The backend serves recommendations on-the-fly, meaning recommendation results are computed directly during request processing.

High-level runtime behavior:

::

   Request -> recommendation computation -> response

This means that no separate prediction artifact is required during serving.

AWS Resources
-------------

The deployment relies on the following AWS services:

Amazon ECR
~~~~~~~~~~

Amazon ECR is used to store the backend container image.

Typical resource:

- ECR repository for the Word2Vec backend image

AWS Lambda
~~~~~~~~~~

AWS Lambda is used as the runtime environment for the backend service.

Typical resource:

- Lambda function for the Word2Vec backend

Lambda is responsible for:

- starting the FastAPI application
- processing incoming requests
- returning recommendation results

Amazon API Gateway
~~~~~~~~~~~~~~~~~~

Amazon API Gateway exposes the backend through public HTTP endpoints.

It acts as the interface between the consumer and the deployed backend.

IAM
~~~

IAM is used for:

- Lambda execution permissions
- GitHub Actions deployment permissions through OIDC-based role assumption

CloudWatch Logs
~~~~~~~~~~~~~~~

CloudWatch Logs is used for runtime monitoring and debugging.

It should be the first place to inspect in case of startup or runtime failures.

Containerization
----------------

The backend is packaged as a Docker image.

Containerization was chosen in order to:

- ensure a reproducible runtime environment
- package dependencies together with the application
- simplify deployment to AWS Lambda

A typical Dockerfile setup for Lambda-based deployment looks like:

.. code-block:: dockerfile

   FROM public.ecr.aws/lambda/python:3.12

   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   COPY src /var/task/src

   CMD ["src.app.handler"]

The exact entry point may differ depending on the repository structure and Lambda handler implementation.

CI/CD Pipeline
--------------

Deployment is automated using GitHub Actions.

Typical workflow steps include:

1. checkout the repository
2. configure AWS credentials
3. authenticate against Amazon ECR
4. build the Docker image
5. push the image to ECR
6. update the Lambda function image

This allows backend changes to be deployed automatically after a push to the configured branch.

Authentication with AWS
~~~~~~~~~~~~~~~~~~~~~~~

GitHub Actions authenticates against AWS by assuming an IAM role via OIDC.

This avoids storing long-lived AWS credentials in the repository.

Typical requirement:

- GitHub secret: ``AWS_ROLE_ARN``

The corresponding IAM role must trust:

- ``token.actions.githubusercontent.com``

and restrict access to the intended repository and deployment branch.

API Exposure
------------

The backend is exposed through API Gateway.

Typical endpoints include:

- ``GET /health``
- ``GET /recommendations``
- ``POST /recommendations/multi``

The exact base URL depends on the deployed API Gateway configuration.

The purpose of the API layer is to provide a stable interface for the consumer while keeping the internal backend logic independent from the external contract.

Configuration
-------------

The exact runtime configuration depends on the repository implementation.

Typical configuration areas include:

- Lambda function settings
- API Gateway integration
- IAM execution role
- environment variables
- image URI in ECR

If the backend depends on model-specific configuration, these values should be documented together with the corresponding codebase.

Operational Notes
-----------------

The following checks are recommended after deployment:

- verify that the GitHub Actions workflow completed successfully
- confirm that Lambda references the updated container image
- call the ``/health`` endpoint
- test recommendation endpoints with known request examples
- inspect CloudWatch logs in case of unexpected runtime behavior

Since this backend computes recommendations at request time, operational maintenance is mainly focused on:

- backend code changes
- deployment validation
- runtime monitoring

No separate prediction artifact refresh is required for serving.

Troubleshooting
---------------

Common issues include:

Lambda startup failures
~~~~~~~~~~~~~~~~~~~~~~~

Possible causes:

- missing Python modules
- incorrect handler path
- incomplete Docker image contents

Recommended action:

- inspect CloudWatch logs
- verify Dockerfile copy instructions
- confirm that all required files are committed and included in the image

GitHub Actions deployment failures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Possible causes:

- incorrect ``AWS_ROLE_ARN``
- invalid OIDC trust relationship
- missing ECR or Lambda permissions

Recommended action:

- inspect the GitHub Actions logs
- verify the IAM role trust policy
- verify repository and branch restrictions

API returns internal server errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Possible causes:

- runtime import errors
- invalid environment configuration
- backend startup failure

Recommended action:

- test ``/health``
- inspect CloudWatch logs
- verify Lambda configuration


