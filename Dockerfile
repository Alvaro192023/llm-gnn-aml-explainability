# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencias primero: maximiza el aprovechamiento de la cache de capas
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Codigo fuente
COPY . .

# Ejecutar como usuario sin privilegios
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

# Smoke test por defecto: el paquete importa y expone su version
CMD ["python", "-c", "import codigo; print('llm-gnn-aml-explainability', codigo.__version__)"]
