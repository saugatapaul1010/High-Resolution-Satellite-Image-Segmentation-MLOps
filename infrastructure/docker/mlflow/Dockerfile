# infrastructure/docker/mlflow/Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install MLflow
RUN pip install --no-cache-dir mlflow psycopg2-binary

# Create directory for MLflow artifacts
RUN mkdir -p /mlflow

# Expose port for MLflow UI
EXPOSE 5000

# Start MLflow server
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "/mlflow"]