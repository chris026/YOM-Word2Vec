AWS Configuration
=================

This section documents the AWS resources and configuration required to run the Word2Vec-based on-the-fly backend.

Region
------

The backend is deployed in:

- ``eu-central-1``

Core AWS Resources
------------------

The deployment uses the following AWS services:

- Amazon ECR
- AWS Lambda
- Amazon API Gateway
- IAM
- Amazon CloudWatch Logs

Amazon ECR
----------

Amazon ECR stores the Docker image of the backend.

Typical resource:

- ECR repository for the Word2Vec backend image

This repository is used by the CI/CD pipeline to push new container versions.

AWS Lambda
----------

AWS Lambda runs the backend service.

Typical resource:

- Lambda function for the Word2Vec backend

The Lambda function is configured to use the backend image stored in ECR.

Relevant Lambda settings include:

- image-based deployment
- execution role
- memory configuration
- timeout configuration
- environment variables, if required by the implementation

Amazon API Gateway
------------------

Amazon API Gateway exposes the backend through public HTTP endpoints.

Typical responsibility:

- receive incoming HTTP requests
- forward them to the Lambda function
- return the backend response to the client

Example deployed endpoint used during the project:

::

   https://kxqjiw9kq5.execute-api.eu-central-1.amazonaws.com

IAM
---

IAM is used in two different places:

- Lambda execution role
- deployment role for GitHub Actions

Lambda execution role
~~~~~~~~~~~~~~~~~~~~~

The Lambda execution role is used at runtime.

It should allow at least:

- writing logs to CloudWatch

Depending on the backend implementation, it may also need access to additional AWS resources.

Deployment role
~~~~~~~~~~~~~~~

The deployment role is used by GitHub Actions.

It must allow:

- authentication via OIDC
- pushing images to ECR
- updating the Lambda function

This role is different from the Lambda execution role.

CloudWatch Logs
---------------

CloudWatch Logs is used for runtime logging and debugging.

It should be the first place to inspect when:

- the backend fails during startup
- the API returns internal server errors
- deployment succeeds but the service does not work correctly

Environment Variables
---------------------

The exact environment variables depend on the implementation of the backend.

Typical configuration areas may include:

- model identifiers
- service-level configuration
- optional external integration settings

For handover, the Lambda configuration in AWS should be reviewed directly to confirm which variables are currently required.

Configuration Checks
--------------------

The following points should be verified during setup or handover:

- correct AWS region
- existing ECR repository
- correct Lambda function
- valid API Gateway integration
- attached Lambda execution role
- working CloudWatch logging
- correct deployment role for GitHub Actions

Summary
-------

The Word2Vec backend uses a standard serverless AWS setup based on:

- ECR for image storage
- Lambda for execution
- API Gateway for public access
- IAM for runtime and deployment permissions
- CloudWatch Logs for monitoring and debugging