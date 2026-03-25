FROM public.ecr.aws/lambda/python:3.12

WORKDIR /var/task

RUN dnf install -y libgomp && dnf clean all

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/src/ ./src
COPY serve_bundle.py ./serve_bundle.py
COPY models/ ./models
COPY artifacts/ ./artifacts
COPY data/ ./data

CMD ["src.app.handler"]