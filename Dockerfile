FROM python:3.12-slim

ARG INSTALL_TEST_DEPS=false

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgomp1 tesseract-ocr && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements-test.txt ./
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.6.0
RUN pip install --no-cache-dir -r requirements.txt
RUN if [ "$INSTALL_TEST_DEPS" = "true" ]; then pip install --no-cache-dir -r requirements-test.txt; fi
COPY . .
RUN useradd --create-home --uid 10001 appuser && mkdir -p /data/uploads && chown -R appuser:appuser /app /data
USER appuser
