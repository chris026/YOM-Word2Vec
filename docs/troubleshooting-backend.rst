Troubleshooting
===============

This section lists common issues that may occur when deploying or running the Word2Vec-based on-the-fly backend.

Recommended Debugging Order
---------------------------

When a problem occurs, use this order:

1. test whether the API is reachable
2. call ``/health``
3. inspect the latest CloudWatch logs
4. check the GitHub Actions run
5. verify Lambda configuration
6. verify IAM / OIDC settings if deployment failed

API Returns Internal Server Error
---------------------------------

Symptom
~~~~~~~

.. code-block:: json

   {"message":"Internal Server Error"}

Likely causes
~~~~~~~~~~~~~

- Lambda startup failure
- import error
- missing dependency
- invalid runtime configuration
- application error during request processing

Checks
~~~~~~

- test ``/health``
- inspect CloudWatch logs
- verify the latest deployment state
- verify Lambda configuration

Health Endpoint Fails
---------------------

Symptom
~~~~~~~

``/health`` also returns an error.

Likely causes
~~~~~~~~~~~~~

- backend did not start successfully
- missing Python module
- incorrect handler path
- incomplete Docker image contents

Checks
~~~~~~

- inspect CloudWatch logs
- look for ``ImportModuleError`` or other startup errors
- verify Dockerfile copy instructions
- verify the Lambda handler path

Runtime Import Errors
---------------------

Symptom
~~~~~~~

CloudWatch shows an error such as:

.. code-block:: text

   Runtime.ImportModuleError: Unable to import module ...

Likely causes
~~~~~~~~~~~~~

- required files are missing in the image
- source code was not committed
- Dockerfile copied the wrong directory
- handler path does not match the actual package structure

Checks
~~~~~~

- verify repository contents
- confirm that required files were committed and pushed
- inspect Dockerfile ``COPY`` statements
- verify the handler path in Lambda

GitHub Actions Cannot Assume AWS Role
-------------------------------------

Symptom
~~~~~~~

GitHub Actions fails with:

.. code-block:: text

   Not authorized to perform sts:AssumeRoleWithWebIdentity

Likely causes
~~~~~~~~~~~~~

- incorrect ``AWS_ROLE_ARN`` secret
- invalid OIDC trust relationship
- repository or branch not allowed in IAM trust policy

Checks
~~~~~~

- verify ``AWS_ROLE_ARN`` in GitHub secrets
- inspect IAM trust relationship
- confirm trust for ``token.actions.githubusercontent.com``
- confirm correct repository and branch restrictions

Lambda Rejects the Image
------------------------

Symptom
~~~~~~~

Lambda reports that the image manifest or media type is not supported.

Likely causes
~~~~~~~~~~~~~

- wrong build architecture
- incompatible image manifest
- unsupported build metadata

Checks
~~~~~~

- build for ``linux/amd64``
- use Docker Buildx
- disable provenance metadata if needed

Typical working settings:

- ``--platform linux/amd64``
- ``--provenance=false``

Deployment Succeeds but Service Fails
-------------------------------------

Symptom
~~~~~~~

GitHub Actions finishes successfully, but the API still fails afterward.

Likely causes
~~~~~~~~~~~~~

- runtime error not visible during build
- missing source file in the deployed image
- invalid Lambda configuration
- incorrect handler or environment settings

Checks
~~~~~~

- inspect CloudWatch logs immediately after deployment
- test ``/health``
- verify Lambda configuration
- verify that the correct image is active

Local Code Works, Deployed Code Fails
-------------------------------------

Symptom
~~~~~~~

The backend works locally, but fails in AWS.

Likely causes
~~~~~~~~~~~~~

- local files were never committed
- files were ignored by ``.gitignore``
- deployment branch does not contain the expected code

Checks
~~~~~~

- run ``git status``
- run ``git check-ignore -v <path>``
- verify the active branch
- verify the deployed branch contents

CloudWatch Logs
---------------

CloudWatch Logs should be the first place to look for runtime issues.

Use them to diagnose:

- startup failures
- import errors
- runtime exceptions
- unexpected request failures

Summary
-------

Most issues in this backend setup are caused by one of the following:

- deployment state
- missing source files
- incorrect handler or Docker setup
- IAM / OIDC problems
- runtime startup failures

In most cases, the fastest path to diagnosis is:

- test ``/health``
- inspect CloudWatch logs
- then verify deployment and repository state