# Agent-RL Credit Auditor — CPU-only release image
#
# docker build -t credit-auditor .
# docker run --rm -v $PWD/artifacts:/artifacts credit-auditor \
#   credit-auditor validate-protocol /app/configs/protocols/m0_regression_v1.json
#
# CPU-only by design (design §20.1: 0 GPU·h for v0.1): the image never
# installs CUDA or model weights.
FROM python:3.12-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir /app \
    && rm -rf /app/.venv /app/artifacts

WORKDIR /app
ENTRYPOINT ["credit-auditor"]
