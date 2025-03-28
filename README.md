# High-Resolution Satellite Image Segmentation MLOps Pipeline

![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/your-org/satellite-segmentation-mlops/ci.yaml?branch=main)
![Python Version](https://img.shields.io/badge/python-3.9-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Table of Contents

- [Introduction](#introduction)
- [Project Vision](#project-vision)
- [System Architecture](#system-architecture)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
- [Core Components](#core-components)
  - [Data Management](#data-management)
  - [Training Pipeline](#training-pipeline)
  - [Model Registry](#model-registry)
  - [API Service](#api-service)
  - [Monitoring System](#monitoring-system)
- [Continuous Integration](#continuous-integration)
  - [Pipeline Overview](#pipeline-overview)
  - [Code Quality Checks](#code-quality-checks)
  - [Testing Strategy](#testing-strategy)
  - [Artifact Management](#artifact-management)
- [Development Workflow](#development-workflow)
- [Contributing](#contributing)
- [License](#license)

## Introduction

In the age of Earth observation satellites, we're swimming in a sea of high-resolution imagery. This abundance of visual data offers unprecedented opportunities for monitoring our planet – from detecting deforestation to urban planning and disaster response. But the real challenge? Transforming these raw pixels into actionable insights, at scale.

Satellite image segmentation – the task of pixel-wise classification of satellite imagery – stands at the frontier of this challenge. Manual annotation is slow, expensive, and simply cannot keep pace with the volume of incoming data. Machine learning offers a solution, but implementing a production-ready ML system comes with its own set of complex engineering challenges.

This project implements a robust, production-grade MLOps system for high-resolution satellite image segmentation. Built for on-premise deployment, it integrates the best practices of software engineering with the specialized needs of machine learning workflows. From data versioning to continuous integration, from experiment tracking to model deployment and monitoring – I've created a comprehensive solution for organizations that need reliable, reproducible, and maintainable AI systems for satellite imagery analysis.

> "In the world of satellite imagery, the difference between a prototype and a production system is the difference between interesting research and real-world impact."

## Project Vision

When I started this project, I asked myself: what would it take to bridge the gap between a working segmentation model and a system that delivers value in production, year after year? The answer wasn't just about model accuracy – it was about engineering discipline, operational excellence, and designing for change.

The vision for this system encompasses five key principles:

1. **Reproducibility First**: Every experiment, dataset version, and model must be tracked and reproducible. There's no place for "it worked on my machine" in production ML.

2. **Continuous Improvement**: The system should evolve through continuous integration, testing, and deployment of both code and models.

3. **Self-Monitoring**: A production system needs to watch itself, detecting when data distributions shift or model performance degrades.

4. **Scalable Processing**: Satellite images are BIG. The system must efficiently handle preprocessing, training, and inference on high-resolution imagery.

5. **Quality Through Automation**: Tests, code quality checks, and documented processes ensure the system remains maintainable as it grows.

This repository represents the realization of that vision – a complete MLOps pipeline that transforms raw satellite images and annotations into continuously improving segmentation models that deliver reliable results in production.

## System Architecture

Our architecture cleanly separates concerns while maintaining a coherent workflow from data to deployment:

![System Architecture Diagram](docs/images/system-architecture.png)

### Data Management Layer

The foundation of our system, responsible for:
- Versioning raw satellite imagery and annotations (DVC)
- Data validation and quality assurance
- Preprocessing and patch generation
- Maintaining a data registry with metadata

### Training Pipeline

The experimental heart of our system, handling:
- Feature engineering from preprocessed data
- Model training with various architectures (LinkNet, UNet, MANet)
- Hyperparameter optimization
- Evaluation against standard metrics (IoU, Dice coefficient)
- Experiment tracking (MLflow)

### Model Registry

Our model management layer, providing:
- Versioned storage of trained models
- Model metadata and performance metrics
- Staging and production environment transitions
- Model lineage tracking

### Inference Pipeline

The operational core, offering:
- Real-time prediction via REST API (FastAPI)
- Batch prediction for large datasets
- Post-processing of segmentation masks
- Coordinate transformation to geographic systems

### Monitoring System

The vigilant observer, featuring:
- Performance monitoring of deployed models
- Data drift detection
- Concept drift identification
- Alerting systems
- Centralized logging

### CI/CD Pipeline

The engineering backbone, ensuring:
- Code quality through automated checks
- Comprehensive testing
- Continuous integration of new features
- Automated deployment (future enhancement)

### Containerization

The deployment foundation, providing:
- Isolated, reproducible environments
- Service orchestration
- Resource management
- Consistent deployment across environments

## Getting Started

### Prerequisites

Before diving in, ensure you have the following installed:

- Python 3.9+
- Docker and Docker Compose
- Git
- (Optional) NVIDIA drivers and Docker GPU support for accelerated training

### Installation

1. Clone the repository:

```bash
git clone https://github.com/your-org/satellite-segmentation-mlops.git
cd satellite-segmentation-mlops
```

2. Set up the environment:

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

3. Create required directories:

```bash
mkdir -p data/raw data/processed models logs
```

4. Start the services with Docker Compose:

```bash
docker-compose up -d
```

This launches:
- FastAPI service on port 8000
- MLflow tracking server on port 5000
- Redis for task queue
- Celery workers for background processing
- SonarQube for code quality (port 9000)

5. Verify the installation:

```bash
# Check API health
curl http://localhost:8000/health

# Access MLflow UI
open http://localhost:5000

# Access SonarQube
open http://localhost:9000
```

### Configuration

Configuration is managed through a combination of:

1. Environment variables (loaded from `.env` file)
2. DVC parameters (in `params.yaml`)
3. Application settings (in `core/config/settings.py`)

Create a local configuration by copying the example:

```bash
cp .env.example .env
```

Key configuration options include:

```
# API settings
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True

# Data paths
DATA_DIR=/data
MODELS_DIR=/models
LOG_DIR=/logs

# MLflow configuration
MLFLOW_TRACKING_URI=http://mlflow:5000

# Redis & Celery
REDIS_HOST=redis
REDIS_PORT=6379

# DVC
DVC_REMOTE=local
```

For different environments (development, testing, production), you can create different `.env` files and switch between them.

## Core Components

### Data Management

The data management system handles the unique challenges of satellite imagery:

#### Data Upload and Organization

Raw data can be uploaded via the API:

```bash
# Upload an annotation file
curl -X POST -F "file=@sample.json" http://localhost:8000/data/upload

# Upload the corresponding image
curl -X POST -F "file=@sample.png" http://localhost:8000/data/upload
```

Uploaded files are automatically organized by dataset name, extracted from the filename.

#### Data Versioning

We will use DVC to track changes to datasets, ensuring reproducibility across experiments:

```bash
# List available data versions
curl http://localhost:8000/data/versions

# Checkout a specific version
curl -X POST http://localhost:8000/data/checkout/v1.0
```

Behind the scenes, this initializes a DVC repository in the data directory and tracks changes as data is processed:

```python
# From tasks/data_tasks.py
os.chdir(data_dir)
os.system(f"dvc add processed/{dataset_name}")
os.system(f"dvc push")
```

#### Data Preprocessing

The preprocessing pipeline transforms raw satellite images and annotations into training-ready data:

1. **Binary Mask Creation**: Converts polygon annotations to binary masks
2. **Patch Generation**: Divides large satellite images into manageable patches
3. **Balance Enhancement**: Generates multiple versions of patches containing objects of interest
4. **Train/Validation Split**: Creates stratified splits for model training

This process is implemented in the `DataPreparation` class:

```python
# From core/data/data_preparation.py
class DataPreparation:
    def prepare_patches(self, json_files):
        """Prepare patches from satellite images."""
        for filename in json_files:
            # Load image and annotations
            png_image = cv2.imread(png_filename)
            shape_dicts = self.get_poly(annotation_path)
            
            # Create binary mask
            im_binary = self.create_binary_masks(png_image, shape_dicts)
            
            # Create patches
            patches_mask = patchify(pad_mask, (self.patch_size, self.patch_size), self.patch_size)
            patches_img = patchify(pad_img, (self.patch_size, self.patch_size, 3), self.patch_size)
            
            # Process each patch
            for i in range(patches_mask.shape[0]):
                for j in range(patches_mask.shape[1]):
                    # Save patches with special handling for positive samples
                    ...
```

Processing is executed as a background task to handle large images efficiently:

```bash
# Trigger data processing
curl -X POST "http://localhost:8000/data/process?dataset_name=dataset1"
```

### Training Pipeline

The training pipeline implements state-of-the-art segmentation models for satellite imagery:

#### Model Architecture

We support multiple segmentation architectures:

- **LinkNet**: Efficient architecture with skip connections
- **UNet**: The classic U-shaped network
- **MANet**: Multi-scale attention network

These are implemented using the Segmentation Models library with TensorFlow/Keras:

```python
# From core/models/trainer.py
def build_model(self):
    """Build segmentation model."""
    img_shape = (*self.img_size, 3)  # (width, height, channels)
    
    if self.model_type.lower() == "linknet":
        model = sm.Linknet(
            backbone_name=self.backbone,
            input_shape=img_shape,
            activation="sigmoid",
            classes=1,
            encoder_weights="imagenet"
        )
    # Other model types...
```

#### Training Workflow

The training workflow includes:

1. **Data Generation**: Creating TensorFlow data generators with augmentation
2. **Model Compilation**: Setting up loss functions, optimizers, and metrics
3. **Callback Configuration**: Early stopping, learning rate scheduling, checkpoints
4. **Training Execution**: Running the training loop
5. **Evaluation**: Calculating performance metrics

Training can be triggered via API:

```bash
curl -X POST -H "Content-Type: application/json" -d '{
  "model_type": "linknet",
  "dataset_name": "dataset1",
  "parameters": {
    "batch_size": 4,
    "learning_rate": 0.001,
    "epochs": 50,
    "backbone": "efficientnetb0"
  },
  "experiment_name": "satellite_segmentation"
}' http://localhost:8000/training/start
```

This runs as a Celery task, allowing for long-running training jobs:

```python
# From tasks/training_tasks.py
@shared_task(bind=True)
def train_model(self, model_type, dataset_name, parameters, experiment_name=None):
    """Train a model as a background task."""
    # Set up MLflow tracking
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    
    # Start MLflow run
    with mlflow.start_run(experiment_id=experiment_id) as run:
        # Log parameters
        mlflow.log_params(parameters)
        
        # Configure and run training
        trainer = ModelTrainer(config)
        model, history = trainer.train()
        
        # Log metrics from history
        for metric_name, values in history.history.items():
            for step, value in enumerate(values):
                mlflow.log_metric(metric_name, value, step=step)
        
        # Log model to MLflow
        mlflow.tensorflow.log_model(model, "model")
```

#### Experiment Tracking

Every training run is tracked in MLflow, recording:

- Hyperparameters
- Training and validation metrics
- Model artifacts
- Runtime information

This creates a comprehensive record of experiments, enabling:

- Comparison between runs
- Reproduction of results
- Model lineage tracking
- Performance analysis

The MLflow UI provides a visual interface for exploring experiments:

![MLflow Experiments](docs/images/mlflow-experiments.png)

### Model Registry

The Model Registry provides a central repository for trained models:

#### Model Storage

Models are stored in the MLflow Model Registry, which maintains:

- Model versions
- Stage transitions (None, Staging, Production)
- Model metadata and tags
- Performance metrics

Models can be registered via API:

```bash
curl -X POST "http://localhost:8000/training/register/run_id?model_name=satellite_segmentation" 
```

#### Model Retrieval

Models can be retrieved for inference or analysis:

```python
# From the Python client
import mlflow.tensorflow

model = mlflow.tensorflow.load_model("models:/satellite_segmentation/Production")
```

The API service automatically loads the latest production model for inference.

### API Service

The API service provides a RESTful interface to the entire system:

#### Health Checks

Monitor system health with:

```bash
curl http://localhost:8000/health
```

This checks the status of all components (Redis, MLflow, disk space) and returns a consolidated health report.

#### Data Management Endpoints

```bash
# List datasets
curl http://localhost:8000/data/datasets

# Upload data
curl -X POST -F "file=@sample.json" http://localhost:8000/data/upload

# Process data
curl -X POST "http://localhost:8000/data/process?dataset_name=dataset1"

# List data versions
curl http://localhost:8000/data/versions

# Checkout version
curl -X POST http://localhost:8000/data/checkout/v1.0
```

#### Training Endpoints

```bash
# Start training
curl -X POST -H "Content-Type: application/json" -d '{...}' http://localhost:8000/training/start

# Check training status
curl http://localhost:8000/training/status/task_id

# List models
curl http://localhost:8000/training/models

# Register model
curl -X POST "http://localhost:8000/training/register/run_id?model_name=model_name"
```

The API is implemented using FastAPI, providing automatic documentation:

```bash
# Access API documentation
open http://localhost:8000/docs
```

### Monitoring System

The monitoring system tracks model performance and data distributions:

#### Performance Monitoring

Tracks metrics like IoU and Dice coefficient over time, detecting performance degradation.

#### Data Drift Detection

Analyzes statistical properties of input data to detect when distributions shift:

```python
# From tasks/monitoring_tasks.py
@shared_task
def check_data_drift(reference_dataset, current_dataset, threshold=0.05):
    """Check for data drift between reference and current datasets."""
    from scipy import stats
    
    # Load datasets
    reference_samples = []
    current_samples = []
    
    # Extract image statistics
    # ...
    
    # Calculate drift for each feature
    for i in range(reference_array.shape[1]):
        ks_statistic, p_value = stats.ks_2samp(
            reference_array[:, i],
            current_array[:, i]
        )
        
        drift_metrics[feature_name] = {
            "ks_statistic": float(ks_statistic),
            "p_value": float(p_value),
            "drift_detected": p_value < threshold
        }
```

#### Alerting System

When issues are detected, the system can:
- Log warnings
- Trigger alerts (future enhancement)
- Initiate retraining (future enhancement)

## Continuous Integration

Our CI pipeline ensures code quality and reliability through automated checks and tests.

### Pipeline Overview

The CI pipeline runs on every pull request and push to the main branch, performing:

1. Code linting and formatting checks
2. Static type checking
3. Unit and integration tests
4. Code coverage analysis
5. SonarQube code quality scan

The pipeline is implemented in GitHub Actions:

```yaml
# From .github/workflows/ci.yaml
name: CI

on:
  push:
    branches:
      - main
  pull_request:
    branches:
      - main

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt

    - name: Run tests
      run: |
        pytest tests/

    - name: SonarQube Scan
      uses: sonarsource/sonarqube-scan-action@master
      env:
        SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}
        SONAR_HOST_URL: ${{ secrets.SONAR_HOST_URL }}
```

### Code Quality Checks

We use multiple tools to ensure code quality:

#### Linting and Formatting

- **Black**: Automatic code formatting
- **isort**: Import sorting
- **flake8**: Style guide enforcement

Configuration is in `pyproject.toml`:

```toml
[tool.black]
line-length = 88
target-version = ['py39']

[tool.isort]
profile = "black"
multi_line_output = 3
```

#### Static Type Checking

We use MyPy to catch type-related errors before runtime:

```toml
[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
```

#### SonarQube Analysis

SonarQube provides comprehensive code quality analysis:

```properties
# sonar-project.properties
sonar.projectKey=satellite-segmentation-mlops
sonar.projectName=Satellite Segmentation MLOps
sonar.projectVersion=1.0

# Source code location
sonar.sources=api,core,tasks
sonar.tests=tests

# Python specifics
sonar.python.coverage.reportPaths=coverage.xml
sonar.python.xunit.reportPath=test-results.xml
```

To run a local analysis:

```bash
./infrastructure/scripts/run_sonar_analysis.sh
```

### Testing Strategy

Our testing strategy covers multiple levels:

#### Unit Tests

Test individual components in isolation:

```python
# From tests/test_core/test_data_preparation.py
def test_create_binary_masks(sample_image):
    """Test binary mask creation."""
    # Create shape dictionaries
    shape_dicts = [
        {
            "points": [[64, 64], [192, 64], [192, 192], [64, 192]]
        }
    ]
    
    # Create binary mask
    mask = DataPreparation.create_binary_masks(sample_image, shape_dicts)
    
    # Check mask shape and values
    assert mask.shape == (sample_image.shape[0], sample_image.shape[1])
    assert np.sum(mask[64:192, 64:192]) > 0
    assert np.sum(mask[0:64, 0:64]) == 0
```

#### API Tests

Test API endpoints using FastAPI's TestClient:

```python
# From tests/test_api/test_health.py
def test_health_endpoint(api_client):
    """Test the health check endpoint."""
    with patch('api.services.health.HealthService._check_redis', return_value='ok'), \
         patch('api.services.health.HealthService._check_mlflow', return_value='ok'), \
         patch('api.services.health.HealthService._check_disk_space', return_value='ok'):
        
        response = api_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
```

#### Integration Tests

Test interactions between components:

```python
# Example integration test
def test_end_to_end_training_flow(api_client, sample_dataset):
    """Test the complete training flow from data to model."""
    # Process dataset
    response = api_client.post(f"/data/process?dataset_name={sample_dataset}")
    assert response.status_code == 200
    process_response = response.json()
    
    # Start training
    training_request = {
        "model_type": "linknet",
        "dataset_name": sample_dataset,
        "parameters": {"batch_size": 2, "epochs": 1}
    }
    response = api_client.post("/training/start", json=training_request)
    assert response.status_code == 200
    # ... check training results
```

#### Test Fixtures

We use pytest fixtures to set up test environments:

```python
# From tests/conftest.py
@pytest.fixture
def api_client():
    """FastAPI test client."""
    return TestClient(app)

@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    # Create a simple pattern
    img[64:192, 64:192] = [255, 0, 0]
    return img
```

#### Test Coverage

We track code coverage to ensure comprehensive testing:

```ini
# pytest.ini
[pytest]
testpaths = tests
addopts = --cov=api --cov=core --cov=tasks --cov-report=xml:coverage.xml --cov-report=term
```

### Artifact Management

CI pipeline artifacts include:

- Test results
- Coverage reports
- SonarQube analysis results

These artifacts are stored with each CI run, providing a history of code quality and test results.

## Development Workflow

Our development workflow follows these principles:

1. **Feature Branches**: Develop new features in dedicated branches
2. **Pull Requests**: Submit PRs for code review and CI validation
3. **Code Review**: All changes require review before merging
4. **CI Validation**: PRs must pass CI checks before merging
5. **Documentation**: Update documentation alongside code changes

A typical workflow:

```bash
# Create a feature branch
git checkout -b feature/new-monitoring-component

# Make changes and commit
git add .
git commit -m "Add new data drift detection algorithm"

# Push changes and create PR
git push origin feature/new-monitoring-component
```

The PR triggers the CI pipeline, which validates the changes.

## Contributing

We welcome contributions to improve the system! Here's how:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests locally
5. Submit a pull request

Please follow our coding standards:
- PEP 8 style guide
- Type annotations for all functions
- Comprehensive docstrings
- Unit tests for new features

## Future Roadmap

While the current implementation focuses on the CI pipeline and core functionality, future enhancements will include:

- Continuous Deployment (CD) pipeline
- Advanced monitoring and alerting
- A/B testing framework
- Model fairness and bias detection
- Support for additional model architectures
- Integration with cloud storage options
- Web-based annotation tool

Stay tuned for these exciting developments!

---

This project is licensed under the MIT License - see the LICENSE file for details.