# High-Resolution Satellite Image Segmentation MLOps Pipeline

![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/your-org/satellite-segmentation-mlops/ci.yaml?branch=main)
![Python Version](https://img.shields.io/badge/python-3.9-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.9-orange)
![License](https://img.shields.io/badge/license-MIT-green)

This repository contains a production-grade MLOps pipeline for satellite image segmentation. The system is designed to handle the full lifecycle of machine learning models including data versioning, experiment tracking, model training, inference, and monitoring - all on-premise.

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
  - [Python Development Environment](#python-development-environment)
  - [Docker Production Environment](#docker-production-environment)
- [Usage Guide](#usage-guide)
  - [Data Management](#data-management)
  - [Model Training](#model-training)
  - [Model Registry](#model-registry)
  - [Inference](#inference)
  - [Monitoring](#monitoring)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Contributing](#contributing)

## Overview

This MLOps pipeline is specifically designed for satellite image segmentation, addressing the unique challenges:

- Processing gigapixel images that don't fit in memory
- Handling geospatial projections
- Managing limited labeled data
- Detecting drift in temporal satellite data
- Maintaining reproducibility in the ML workflow

The system uses Data Version Control (DVC) for data management, MLflow for experiment tracking, FastAPI for serving, and TensorFlow/Segmentation Models for the machine learning components.

## System Architecture

The system follows a layered architecture:

![Architecture Diagram](docs/images/system-architecture.png)

Components include:

1. **Data Management Layer**: Handles data versioning, preprocessing, and storage
2. **Training Pipeline**: Manages model training, hyperparameter tuning, and evaluation
3. **Model Registry**: Stores model versions and metadata
4. **Inference Pipeline**: Provides real-time and batch inference capabilities
5. **Monitoring System**: Tracks model performance and data drift
6. **CI/CD Pipeline**: Ensures code quality and automates deployments

## Prerequisites

Before getting started, ensure you have:

- Python 3.9+
- Git
- Docker and Docker Compose (for production setup)
- NVIDIA GPU (recommended for training)
- 16GB+ RAM

## Setup Instructions

### Python Development Environment

Setting up a development environment allows you to run the system locally and debug components.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-org/satellite-segmentation-mlops.git
   cd satellite-segmentation-mlops
   ```

2. **Create a virtual environment**:
   ```bash
   # Create a new virtual environment
   python -m venv venv
   
   # Activate the environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit the .env file to set your configuration
   # Example (modify paths as needed):
   # DATA_DIR=/path/to/data
   # MODELS_DIR=/path/to/models
   # MLFLOW_TRACKING_URI=http://localhost:5000
   ```

5. **Create required directories**:
   ```bash
   mkdir -p data/raw data/processed models logs
   ```

6. **Run services individually**:
   
   Start MLflow tracking server:
   ```bash
   mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri ./mlflow
   ```
   
   Start Redis (in a separate terminal):
   ```bash
   redis-server
   ```
   
   Start Celery worker (in a separate terminal):
   ```bash
   celery -A tasks.celery_app worker --loglevel=info
   ```
   
   Start FastAPI service (in a separate terminal):
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

7. **Verify installation**:
   
   Access FastAPI docs at: http://localhost:8000/docs
   
   Access MLflow UI at: http://localhost:5000

### Docker Production Environment

For production deployment, we use Docker Compose to manage all services.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-org/satellite-segmentation-mlops.git
   cd satellite-segmentation-mlops
   ```

2. **Configure environment variables**:
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit the .env file as needed
   # The paths will be handled by Docker volumes
   ```

3. **Start all services with Docker Compose**:
   ```bash
   docker-compose up -d
   ```
   
   This starts the following services:
   - FastAPI (API service)
   - Celery Worker (background processing)
   - Redis (message broker)
   - MLflow (experiment tracking)
   - SonarQube (code quality, optional)

4. **Verify deployment**:
   
   Check service status:
   ```bash
   docker-compose ps
   ```
   
   Access FastAPI docs at: http://localhost:8000/docs
   
   Access MLflow UI at: http://localhost:5000
   
   Access SonarQube at: http://localhost:9000 (default login: admin/admin)

5. **View logs**:
   ```bash
   # All services
   docker-compose logs -f
   
   # Specific service
   docker-compose logs -f api
   ```

6. **Stop services**:
   ```bash
   docker-compose down
   ```

## Usage Guide

### Data Management

The system uses DVC to track data versions and a standardized structure for processing satellite imagery.

#### Uploading Data

1. Upload annotation files (JSON) first:

   ```bash
   curl -X POST -F "file=@path/to/annotations.json" http://localhost:8000/data/upload
   ```

2. Then upload corresponding satellite images:

   ```bash
   curl -X POST -F "file=@path/to/image.png" http://localhost:8000/data/upload
   ```

   Or use the FastAPI Swagger UI at http://localhost:8000/docs

#### Processing Data

Process raw data into training patches:

```bash
curl -X POST "http://localhost:8000/data/process?dataset_name=your_dataset_name"
```

### Model Training

Training a model involves:

1. Launch a training job:

   ```bash
   curl -X POST \
     -H "Content-Type: application/json" \
     -d '{
       "model_type": "linknet",
       "dataset_name": "your_dataset_name",
       "parameters": {
         "batch_size": 4,
         "learning_rate": 0.001,
         "epochs": 10,
         "backbone": "efficientnetb0"
       },
       "experiment_name": "My Experiment"
     }' \
     http://localhost:8000/training/start
   ```

2. Check training status (using task_id from the previous response):

   ```bash
   curl -X GET "http://localhost:8000/training/status/{task_id}"
   ```

3. View experiment details in MLflow UI at http://localhost:5000

### Model Registry

After training, register models for deployment:

```bash
curl -X POST "http://localhost:8000/training/register/{run_id}?model_name=my-model"
```

List available models:

```bash
curl -X GET "http://localhost:8000/training/models"
```

### Inference

The system supports both real-time inference and batch processing:

For inference with images, use the FastAPI endpoint:

```bash
curl -X POST \
  -F "file=@path/to/image.png" \
  -F "model_name=my-model" \
  -F "model_version=1" \
  http://localhost:8000/inference/predict
```

### Monitoring

Monitor your models for performance and drift:

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "reference_dataset": "baseline_dataset",
    "current_dataset": "new_dataset",
    "threshold": 0.05
  }' \
  http://localhost:8000/monitoring/check-drift
```

View monitoring dashboards in MLflow at http://localhost:5000

## API Reference

The FastAPI application provides the following endpoints:

### Health

- `GET /health` - Check system health status

### Data Management

- `GET /data/datasets` - List available datasets
- `POST /data/upload` - Upload satellite image or annotation data
- `POST /data/process` - Process raw data into training patches
- `GET /data/versions` - List data versions tracked by DVC
- `POST /data/checkout/{version}` - Checkout a specific data version

### Training

- `POST /training/start` - Start a new training job
- `GET /training/status/{task_id}` - Get training job status
- `GET /training/models` - List registered models
- `POST /training/register/{run_id}` - Register a model from an MLflow run

See the full API documentation at http://localhost:8000/docs

## Project Structure

```
satellite-segmentation-mlops/
├── .github/workflows/      # CI/CD pipeline definitions
├── api/                    # FastAPI service
│   ├── routers/            # API endpoints by resource
│   ├── models/             # Pydantic data models
│   └── services/           # Business logic
├── core/                   # Core ML functionality
│   ├── config/             # Configuration management
│   ├── data/               # Data processing
│   ├── models/             # Model training and evaluation
│   └── utils/              # Shared utilities
├── infrastructure/         # Deployment components
│   ├── docker/             # Dockerfile definitions
│   └── scripts/            # Deployment scripts
├── tasks/                  # Asynchronous tasks
├── tests/                  # Test suite
│   ├── test_api/           # API tests
│   ├── test_core/          # Core functionality tests
│   └── test_tasks/         # Async task tests
├── docker-compose.yml      # Local service definitions
└── requirements.txt        # Python dependencies
```

## Development Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Run linting and tests before committing**:
   ```bash
   # Format code with Black
   black .
   
   # Sort imports
   isort .
   
   # Check style
   flake8
   
   # Run tests
   pytest
   ```

3. **Run code quality analysis (requires SonarQube running)**:
   ```bash
   ./infrastructure/scripts/run_sonar_analysis.sh
   ```

4. **Create a pull request**:
   Push your branch to GitHub and create a pull request. The CI pipeline will automatically run tests.

## Troubleshooting

### Common Issues

**"Cannot connect to Redis" error**:
- Check if Redis is running: `redis-cli ping`
- Verify Redis host/port in .env file

**"MLflow tracking URI not accessible" error**:
- Check if MLflow server is running
- Verify MLflow URI in .env file

**Out of Memory (OOM) errors during training**:
- Reduce batch size in training parameters
- Use smaller patch sizes
- Try a more memory-efficient model architecture (LinkNet)

**Slow data processing**:
- Check disk I/O performance
- Consider using batch processing
- Reduce image size if possible

### Getting Help

If you encounter issues:
1. Check the logs: `docker-compose logs` or individual service logs
2. Consult the FastAPI documentation: http://localhost:8000/docs
3. Open an issue on GitHub with detailed reproduction steps

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing

We welcome contributions! Please follow our development workflow and ensure all tests pass before submitting a pull request.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

For major changes, please open an issue first to discuss what you would like to change.