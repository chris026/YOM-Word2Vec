CI/CD
=====

This section documents the CI/CD setup of the Word2Vec-based on-the-fly backend.

Overview
--------

Deployment is automated through GitHub Actions.

The pipeline follows this flow:

::

   Git push -> GitHub Actions -> Docker build -> ECR push -> Lambda update

This allows backend code changes to be deployed automatically without manual image upload or manual Lambda updates.

CI/CD Platform
--------------

GitHub Actions is used as the CI/CD platform.

The workflow is responsible for:

- checking out the repository
- authenticating against AWS
- building the Docker image
- pushing the image to Amazon ECR
- updating the Lambda function

Workflow Steps
--------------

A typical deployment workflow includes the following steps:

1. checkout the repository
2. configure AWS credentials
3. log in to Amazon ECR
4. build the backend image
5. push the image to ECR
6. update the Lambda function image
7. wait for the Lambda update to complete

Trigger
-------

The exact trigger depends on the repository setup.

Typical triggers are pushes to specific branches, for example:

- ``main``
- a dedicated backend branch

For handover, YOM should define which branch is the deployment source of truth.

AWS Authentication
------------------

GitHub Actions authenticates against AWS via OIDC.

This avoids storing long-lived AWS credentials in the repository.

Required GitHub Secret
~~~~~~~~~~~~~~~~~~~~~~

The workflow typically expects:

- ``AWS_ROLE_ARN``

This secret contains the ARN of the IAM role that GitHub Actions is allowed to assume.

OIDC Trust Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

The IAM role used for deployment must trust:

- ``token.actions.githubusercontent.com``

The trust relationship should also restrict access to the correct GitHub repository and branch.

Docker Build
------------

The backend is built as a Lambda-compatible Docker image.

Important build settings may include:

- ``--platform linux/amd64``
- ``--provenance=false``

These settings are relevant because AWS Lambda requires a compatible image format.

ECR Push
--------

After the image is built, it is pushed to the corresponding Amazon ECR repository.

Typical tagging strategy:

- a commit-based tag
- ``latest``

This allows both traceability and easy access to the newest image version.

Lambda Update
-------------

After the image is pushed, the workflow updates the Lambda function to use the new image.

This connects the CI/CD pipeline directly to the deployed backend runtime.

Validation After Deployment
---------------------------

After a successful deployment, the following checks are recommended:

- verify that the GitHub Actions run completed successfully
- confirm that Lambda references the updated image
- call ``/health``
- test a known recommendation request

A successful pipeline run does not automatically guarantee a successful runtime startup.

Common Failure Points
---------------------

Typical CI/CD issues include:

- incorrect ``AWS_ROLE_ARN`` secret
- invalid OIDC trust relationship
- missing ECR permissions
- missing Lambda update permissions
- incompatible Docker image settings

These issues should be checked in the GitHub Actions logs first.

