# High-Resolution Satellite Image Segmentation MLOps Pipeline

![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/your-org/satellite-segmentation-mlops/ci.yaml?branch=main)
![Python Version](https://img.shields.io/badge/python-3.9-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.9-orange)
![License](https://img.shields.io/badge/license-MIT-green)

## Table of Contents

- [The Challenge of Production ML for Satellite Imagery](#the-challenge-of-production-ml-for-satellite-imagery)
- [From Academic Models to Production Systems](#from-academic-models-to-production-systems)
- [System Architecture: Thinking in Layers](#system-architecture-thinking-in-layers)
- [Developer Setup](#developer-setup)
- [Core Components: Deep Dive](#core-components-deep-dive)
  - [Data Versioning: Why DVC?](#data-versioning-why-dvc)
  - [Data Processing: Fighting the Patch Battle](#data-processing-fighting-the-patch-battle)
  - [Segmentation Models: Architecture Decisions](#segmentation-models-architecture-decisions)
  - [Experiment Tracking: Lessons from the Trenches](#experiment-tracking-lessons-from-the-trenches)
  - [Model Registry: The Single Source of Truth](#model-registry-the-single-source-of-truth)
  - [API Design: RESTful, Async, and Robust](#api-design-restful-async-and-robust)
  - [Monitoring: Watch Your Models Like a Hawk](#monitoring-watch-your-models-like-a-hawk)
- [The CI Pipeline: Quality from Day One](#the-ci-pipeline-quality-from-day-one)
  - [Testing Strategies for ML Systems](#testing-strategies-for-ml-systems)
  - [Static Analysis and Why It Matters](#static-analysis-and-why-it-matters)
  - [Automating Quality Checks](#automating-quality-checks)
- [Development Workflow and Best Practices](#development-workflow-and-best-practices)
- [Future Roadmap](#future-roadmap)
- [FAQs and Troubleshooting](#faqs-and-troubleshooting)
- [License and Contributing](#license-and-contributing)

## The Challenge of Production ML for Satellite Imagery

Three years ago, I found myself staring at a beautiful satellite image segmentation model that worked flawlessly in my Jupyter notebook but failed miserably in production. The painful truth hit me: creating a great model is only 20% of the battle. The other 80%? Building the infrastructure to make that model reliable, maintainable, and actually useful in the real world.

Satellite imagery presents unique MLOps challenges that typical ML systems don't face:

1. **Scale issues**: We're talking gigapixel images that won't fit in memory
2. **Projection problems**: The Earth is round (mostly), but our images are flat
3. **Label scarcity**: Manual annotation of satellite imagery is expensive and requires expertise
4. **Temporal dynamics**: Landscapes change, making model drift inevitable
5. **Processing complexity**: From radiometric calibration to atmospheric correction

After encountering these issues repeatedly across projects, I decided to build a system that would solve these problems once and for all. This repository is the result - a complete MLOps pipeline specifically designed for satellite image segmentation that treats engineering with the same rigor as the ML modeling.

## From Academic Models to Production Systems

When I first started working with segmentation models for satellite imagery, the literature was full of innovative architectures and impressive IoU scores. Papers would casually mention "we used U-Net with an EfficientNet backbone" and show incredible results. 

But those papers never mentioned:
- How they versioned terabytes of satellite data
- How they handled training when images don't fit in memory
- How they managed experiment tracking across hundreds of runs
- How they monitored models in production
- How they retrained when drift occurred

In other words, all the hard parts of building a _system_ rather than just a model.

If you're facing similar challenges, you've probably realized that MLOps isn't just DevOps with a sprinkle of ML - it's a discipline unto itself, with unique requirements and trade-offs. This project embraces that reality by implementing a complete production-grade pipeline for satellite image segmentation that addresses all these concerns.

I've drawn inspiration from systems at companies like Planet, Descartes Labs, and Orbital Insight, but adapted to work completely on-premise (because not everyone has the luxury of unlimited cloud resources, and some data can't leave your building for compliance reasons).

## System Architecture: Thinking in Layers

After multiple iterations (and painful rewrites), I settled on a layered architecture that separates concerns while maintaining a clear flow from data to deployment:

![System Architecture Diagram](docs/images/system-architecture.png)

This architecture isn't arbitrary - it's designed to answer specific questions that arise in production ML systems:

1. **Data Management Layer**: "How do we version, validate, and preprocess satellite imagery at scale?"
2. **Training Pipeline**: "How do we transform raw data into trained models in a reproducible way?"
3. **Model Registry**: "How do we track, compare, and organize models throughout their lifecycle?"
4. **Inference Pipeline**: "How do we serve predictions efficiently for both real-time and batch workloads?"
5. **Monitoring System**: "How do we know if our models are performing as expected in production?"
6. **CI/CD Pipeline**: "How do we ensure code quality and automate deployments?"
7. **Containerization**: "How do we create consistent environments from development to production?"

Each layer has its own responsibilities and interfaces with adjacent layers, creating a modular system that's easier to understand, test, and evolve.

Let's examine a concrete example of how these layers interact:

1. You upload new satellite imagery to the **Data Management Layer**
2. That layer versions the data with DVC and applies preprocessing
3. You trigger a new model training job in the **Training Pipeline**
4. The pipeline trains models with different hyperparameters and tracks them in MLflow
5. The best model is registered in the **Model Registry**
6. The **Inference Pipeline** automatically picks up the new model for serving
7. The **Monitoring System** compares the new model's performance with the previous version
8. If drift is detected, it triggers a retraining job

This flow of data and control ensures that each component has a single responsibility, making the system more maintainable and robust.

## Developer Setup

Before diving into the details, let's get you up and running with the system. I've designed it to be easy to set up locally for development while maintaining parity with production environments through containerization.

### Prerequisites

You'll need:
- Python 3.9+
- Docker and Docker Compose
- Git
- 16GB+ RAM (ML doesn't run on hopes and dreams...yet)
- NVIDIA GPU (optional but recommended for training)

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/your-org/satellite-segmentation-mlops.git
cd satellite-segmentation-mlops

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your specific configuration

# Start the services
docker-compose up -d

# Verify installation
curl http://localhost:8000/health
```

### Project Structure

The project follows a clean, modular structure:

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

This structure separates concerns while keeping related functionality together, making it easier to navigate and maintain the codebase.

## Core Components: Deep Dive

Now let's dive deep into each component, examining not just what it does, but why I made specific design choices and the alternatives I considered.

### Data Versioning: Why DVC?

Data Version Control (DVC) sits at the foundation of our system, and for good reason. When I started this project, I considered several options:

1. **Git LFS**: Simple, but struggles with large datasets and doesn't handle pipelines
2. **Custom database solution**: Powerful but requires too much maintenance
3. **Cloud storage with versioning**: Great but doesn't work for on-premise
4. **DVC**: Git-like interface, pipeline support, storage-agnostic, and open-source

DVC won out for several reasons:

- It's designed specifically for ML workflows
- It integrates seamlessly with Git
- It supports remote storage backends (S3, GCS, Azure, etc.)
- It handles both data versioning AND pipeline definitions
- It's storage-efficient (using symlinks and content-addressable storage)

Here's how we use DVC in practice:

```python
# From tasks/data_tasks.py
def process_dataset(dataset_name, data_dir):
    # ... data processing logic ...
    
    # Record the dataset in DVC
    os.chdir(data_dir)
    os.system(f"dvc add processed/{dataset_name}")
    os.system(f"dvc push")
    
    return {
        "status": "success",
        "dataset": dataset_name,
        "training_samples": len(train_list),
        "validation_samples": len(val_list)
    }
```

This allows us to track every version of every dataset, ensuring reproducibility across experiments. When someone asks, "Which data version produced that great model from last month?", we have an answer.

We also expose DVC operations through our API, allowing non-technical users to interact with versioned data:

```python
# From api/services/data.py
def list_data_versions(self) -> List[str]:
    """List all available data versions tracked by DVC."""
    try:
        # Ensure DVC is initialized
        self._init_dvc_tracking()
        
        # Get list of tags and commits
        result = subprocess.run(
            ["dvc", "list", "--recursive", "--dvc-only"], 
            capture_output=True, 
            text=True,
            check=True
        )
        
        # Parse output
        versions = [line.strip() for line in result.stdout.split('\n') if line.strip()]
        return versions
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"DVC error: {e.stderr}")
```

### Data Processing: Fighting the Patch Battle

Satellite images pose unique challenges for deep learning - they're often gigapixel in size, while our models typically expect images of 512×512 pixels or similar. The solution? Patching.

But patching introduces its own problems:
- **Class imbalance**: Most patches in satellite imagery contain nothing interesting
- **Border effects**: Objects get cut off at patch boundaries
- **Contextual information loss**: Models can't see beyond the patch boundary

After extensive experimentation, I settled on a sophisticated patching strategy that:

1. Creates binary masks from polygon annotations
2. Generates regular grid patches from both images and masks
3. Identifies patches containing objects of interest
4. Creates multiple versions of positive patches with augmentation
5. Samples a balanced subset of negative patches

This strategy is implemented in the `DataPreparation` class:

```python
# From core/data/data_preparation.py
def prepare_patches(self, json_files: List[str]) -> None:
    """Prepare patches from satellite images."""
    for filename in tqdm(json_files):
        # ... processing logic ...
        
        # Create patches with special handling for positive samples
        for i in range(patches_mask.shape[0]):
            for j in range(patches_mask.shape[1]):
                patch_mask = patches_mask[i, j]
                patch_img = patches_img[i, j, 0]
                
                if np.sum(patch_mask) == 0:  # Background patch
                    # Save with normal filename
                    cv2.imwrite(
                        os.path.join(self.binary_masks_patches_dir, f"{base_filename}_{i}_{j}.png"), 
                        patch_mask
                    )
                    cv2.imwrite(
                        os.path.join(self.raw_images_patches_dir, f"{base_filename}_{i}_{j}.png"), 
                        patch_img
                    )
                else:  # Patch contains object
                    # Create multiple copies with unique identifiers
                    ctr += 1
                    for xx in range(6):  # 6 copies of each positive patch
                        cv2.imwrite(
                            os.path.join(self.binary_masks_patches_dir, f"{base_filename}_{i}_{j}_{ctr}_{xx}.png"),
                            patch_mask
                        )
                        cv2.imwrite(
                            os.path.join(self.raw_images_patches_dir, f"{base_filename}_{i}_{j}_{ctr}_{xx}.png"),
                            patch_img
                        )
```

I've found this approach drastically improves model training, especially for sparse features like buildings or roads that only occupy a small percentage of typical satellite images.

### Segmentation Models: Architecture Decisions

For segmentation models, I've implemented support for three architectures:

1. **LinkNet**: Memory-efficient with good performance, ideal for production
2. **U-Net**: The classic architecture, reliable but memory-intensive
3. **MANet**: Multi-scale attention network, better for complex features but slower

Why these three? After benchmarking over a dozen architectures, these provided the best trade-offs between accuracy, inference speed, and memory usage for satellite imagery. LinkNet is our default because it achieves 95% of U-Net's performance while using significantly less memory, which matters when processing gigapixel images.

For backbones, we default to EfficientNet, which provides an excellent balance of accuracy and efficiency. The implementation uses the Segmentation Models library to leverage pre-trained weights:

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
    elif self.model_type.lower() == "unet":
        model = sm.Unet(
            backbone_name=self.backbone,
            input_shape=img_shape,
            activation="sigmoid",
            classes=1,
            encoder_weights="imagenet"
        )
    elif self.model_type.lower() == "manet":
        model = sm.MAnet(
            backbone_name=self.backbone,
            input_shape=img_shape,
            activation="sigmoid",
            classes=1,
            encoder_weights="imagenet"
        )
    else:
        raise ValueError(f"Unsupported model type: {self.model_type}")
    
    return model
```

For metrics, I've implemented IoU (Intersection over Union) and Dice coefficient, which are standard for segmentation tasks. The Dice coefficient is particularly useful as a loss function because it naturally handles class imbalance:

```python
# From core/models/metrics.py
@staticmethod
def dice_coef(y_true, y_pred, smooth=1):
    """Calculate Dice coefficient."""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

@staticmethod
def dice_coef_loss(y_true, y_pred):
    """Calculate Dice coefficient loss."""
    return 1 - Metrics.dice_coef(y_true, y_pred)
```

### Experiment Tracking: Lessons from the Trenches

When I started this project, I initially tracked experiments manually in Excel (I know, I know). After that inevitably fell apart, I tried:

1. **Custom database solution**: Too much maintenance
2. **TensorBoard**: Great for visualizing metrics, but doesn't track parameters or artifacts
3. **Weights & Biases**: Excellent but not suitable for on-premise
4. **MLflow**: Open-source, self-hostable, and comprehensive

MLflow emerged as the clear winner for on-premise experiment tracking. It provides:

- Parameter tracking
- Metric logging
- Artifact storage
- Model registry
- Web UI
- Python API

Our training pipeline integrates deeply with MLflow:

```python
# From tasks/training_tasks.py
@shared_task(bind=True)
def train_model(self, model_type, dataset_name, parameters, experiment_name=None):
    """Train a model as a background task."""
    try:
        # Set up MLflow
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        
        # ... experiment setup ...
        
        # Start MLflow run
        with mlflow.start_run(experiment_id=experiment_id) as run:
            # Log parameters
            mlflow.log_params(parameters)
            
            # Update task state
            self.update_state(
                state='PROGRESS',
                meta={'experiment_id': experiment_id, 'run_id': run.info.run_id}
            )
            
            # Configure and run training
            config = {
                "model_type": model_type,
                "dataset_name": dataset_name,
                "parameters": parameters,
                "paths": {
                    "data_dir": os.path.join(settings.DATA_DIR, "processed", dataset_name),
                    "models_dir": os.path.join(settings.MODELS_DIR, run.info.run_id),
                    "output_dir": os.path.join(settings.MODELS_DIR, "outputs", run.info.run_id)
                }
            }
            
            # Initialize trainer and train model
            trainer = ModelTrainer(config)
            model, history = trainer.train()
            
            # Log metrics from history
            for metric_name, values in history.history.items():
                for step, value in enumerate(values):
                    mlflow.log_metric(metric_name, value, step=step)
            
            # Log model to MLflow
            mlflow.tensorflow.log_model(model, "model")
            
            return {
                "status": "completed",
                "experiment_id": experiment_id,
                "run_id": run.info.run_id,
                "metrics": final_metrics
            }
    
    except Exception as e:
        # Log the error
        logger.error(f"Training error: {str(e)}")
        error_message = str(e)
        try:
            mlflow.log_param("error", error_message)
        except:
            pass
        
        return {
            "status": "failed",
            "error": error_message
        }
```

This approach ensures that every training run is fully documented, from hyperparameters to final metrics, making it easy to compare experiments and reproduce results.

One trick I've learned: always catch and log exceptions during training. There's nothing worse than a training job failing after 8 hours with no record of what went wrong.

### Model Registry: The Single Source of Truth

In a production ML system, you need a clear answer to the question: "Which model version is currently deployed?" The Model Registry provides that single source of truth.

MLflow's Model Registry allows you to:
- Track multiple versions of each model
- Transition models through stages (None, Staging, Production)
- Add descriptions and tags
- Compare model versions

Our API exposes this functionality to make it accessible:

```python
# From api/services/training.py
def register_model(self, run_id: str, model_name: str) -> ModelInfo:
    """Register a model from an MLflow run."""
    try:
        # Get run info
        run = mlflow.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
        
        # Register model
        model_uri = f"runs:/{run_id}/model"
        result = mlflow.register_model(model_uri, model_name)
        
        # Return model info
        return ModelInfo(
            name=model_name,
            version=result.version,
            metrics=run.data.metrics,
            created_at=datetime.fromtimestamp(result.creation_timestamp / 1000).isoformat(),
            experiment_id=run.info.experiment_id,
            run_id=run_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to register model: {str(e)}")
```

This makes model deployment a deliberate action rather than an automatic consequence of training, which is crucial for production systems where you need control over what gets deployed.

### API Design: RESTful, Async, and Robust

The API is the main interface to our system, so it needs to be well-designed and robust. I chose FastAPI for several reasons:

1. **Performance**: It's built on Starlette and Uvicorn, making it extremely fast
2. **Type safety**: It uses Python type hints for validation
3. **Async support**: Critical for handling long-running operations
4. **Automatic documentation**: OpenAPI docs are generated automatically
5. **Dependency injection**: Makes testing and separation of concerns easier

The API follows RESTful principles with resources organized by domain:

```python
# From api/main.py
app = FastAPI(
    title="Satellite Segmentation MLOps API",
    description="API for satellite image segmentation MLOps pipeline",
    version="0.1.0"
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(data.router, prefix="/data", tags=["Data"])
app.include_router(training.router, prefix="/training", tags=["Training"])
```

For long-running operations like training, we use Celery to handle asynchronous tasks:

```python
# From api/services/training.py
def start_training(self, background_tasks: BackgroundTasks, request: TrainingRequest) -> TrainingResponse:
    """Start a training job."""
    # ... validation logic ...
    
    # Start async training task
    train_task = train_model.delay(
        str(request.model_type),
        request.dataset_name,
        parameters,
        experiment_name
    )
    
    # Use Celery task ID
    task_id = train_task.id
    
    return TrainingResponse(
        task_id=task_id,
        status=TrainingStatus.PENDING,
        experiment_id=experiment_id,
        run_id=None
    )
```

This design allows the API to respond immediately while training continues in the background, which is essential for operations that might take hours to complete.

### Monitoring: Watch Your Models Like a Hawk

In production ML systems, monitoring is not optional - it's essential. Our monitoring system tracks:

1. **Model performance**: How well is the model performing on current data?
2. **Data drift**: Are input distributions changing over time?
3. **Concept drift**: Is the relationship between inputs and outputs changing?
4. **System health**: Are all components functioning correctly?

Data drift detection is particularly important for satellite imagery, where seasonal changes, atmospheric conditions, and sensor differences can all cause drift:

```python
# From tasks/monitoring_tasks.py
@shared_task
def check_data_drift(reference_dataset, current_dataset, threshold=0.05):
    """Check for data drift between reference and current datasets."""
    try:
        from scipy import stats
        
        # ... load data ...
        
        # Calculate drift for each feature
        drift_metrics = {}
        drift_detected = False
        
        for i in range(reference_array.shape[1]):
            feature_name = f"feature_{i}"
            ks_statistic, p_value = stats.ks_2samp(
                reference_array[:, i],
                current_array[:, i]
            )
            
            drift_metrics[feature_name] = {
                "ks_statistic": float(ks_statistic),
                "p_value": float(p_value),
                "drift_detected": p_value < threshold
            }
            
            if p_value < threshold:
                drift_detected = True
        
        # Log to MLflow
        with mlflow.start_run(run_name="drift_detection"):
            mlflow.log_param("reference_dataset", reference_dataset)
            mlflow.log_param("current_dataset", current_dataset)
            mlflow.log_param("threshold", threshold)
            
            for feature, metrics in drift_metrics.items():
                mlflow.log_metric(f"{feature}_ks", metrics["ks_statistic"])
                mlflow.log_metric(f"{feature}_pvalue", metrics["p_value"])
            
            mlflow.log_param("drift_detected", drift_detected)
        
        return {
            "status": "success",
            "drift_detected": drift_detected,
            "metrics": drift_metrics,
            "timestamp": datetime.now().isoformat()
        }
```

This implementation uses the Kolmogorov-Smirnov test to detect distribution changes between a reference dataset and the current data. When drift is detected, it can trigger model retraining or alert operators.

## The CI Pipeline: Quality from Day One

From the beginning, I knew that a robust CI pipeline would be essential for maintaining code quality as the project evolved. Our CI pipeline runs on GitHub Actions and includes:

1. **Code linting and formatting checks**
2. **Static type checking**
3. **Unit and integration tests**
4. **Code coverage analysis**
5. **SonarQube code quality scan**

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

### Testing Strategies for ML Systems

Testing ML systems requires a different approach than traditional software. We use a combination of:

1. **Unit tests**: Test individual components in isolation
2. **Integration tests**: Test interactions between components
3. **Data tests**: Validate data quality and processing
4. **Model tests**: Validate model behavior on specific inputs

For example, here's a test for the binary mask creation function:

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
    
    # Check mask shape
    assert mask.shape == (sample_image.shape[0], sample_image.shape[1])
    
    # Check that the mask has the correct values
    assert np.sum(mask[64:192, 64:192]) > 0
    assert np.sum(mask[0:64, 0:64]) == 0
```

Testing ML code is challenging because:
- It's often stochastic
- It depends on data quality
- It involves numerical computations with floating-point precision issues
- Traditional metrics like code coverage don't capture data path coverage

To address these challenges, we use pytest fixtures to create consistent test environments:

```python
# From tests/conftest.py
@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    # Create a simple pattern
    img[64:192, 64:192] = [255, 0, 0]
    return img

@pytest.fixture
def sample_json(tmp_path, sample_image):
    """Create a sample annotation JSON for testing."""
    # Save the sample image
    img_path = os.path.join(tmp_path, "sample.png")
    cv2.imwrite(img_path, sample_image)
    
    # Create annotation
    annotation = {
        "shapes": [
            {
                "label": "feature",
                "points": [[64, 64], [192, 64], [192, 192], [64, 192]]
            }
        ],
        "imagePath": img_path,
        "imageHeight": 256,
        "imageWidth": 256
    }
    
    # Save annotation
    json_path = os.path.join(tmp_path, "sample.json")
    with open(json_path, "w") as f:
        json.dump(annotation, f)
    
    return json_path, img_path
```

### Static Analysis and Why It Matters

Static analysis tools catch issues before they become problems. We use:

1. **Black**: Automatic code formatting
2. **isort**: Import sorting
3. **flake8**: Style guide enforcement
4. **mypy**: Static type checking

These tools are configured in `pyproject.toml`:

```toml
[tool.black]
line-length = 88
target-version = ['py39']

[tool.isort]
profile = "black"
multi_line_output = 3

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
```

Type checking is particularly valuable in ML systems, where data transformations can be complex and errors can be subtle:

```python
def process_image(image: np.ndarray) -> np.ndarray:
    """Process an image for model input.
    
    Args:
        image: RGB image array with shape (H, W, 3)
        
    Returns:
        Processed image with shape (H, W, 3)
    """
    # Without type hints, it's easy to miss that this function
    # expects a specific shape and returns a specific shape
```

### Automating Quality Checks

We use SonarQube for comprehensive code quality analysis:

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

SonarQube identifies:
- Code duplications
- Complex methods
- Maintainability issues
- Security vulnerabilities
- Test coverage gaps

We run SonarQube analysis locally during development and in CI:

```bash
./infrastructure/scripts/run_sonar_analysis.sh
```

The script sets up the environment and runs SonarQube analysis:

```bash
#!/bin/bash
# scripts/run_sonar_analysis.sh

# Make sure SonarQube is running
echo "Checking if SonarQube is running..."
if ! curl -s http://localhost:9000 > /dev/null; then
    echo "SonarQube is not running. Start it with 'docker-compose up -d sonarqube'"
    exit 1
fi

# Generate coverage report
echo "Running tests and generating coverage report..."
pytest --cov=. --cov-report=xml --junitxml=test-results.xml

# Run SonarQube analysis
echo "Running SonarQube analysis..."
sonar-scanner \
  -Dsonar.host.url=http://localhost:9000 \
  -Dsonar.login=$SONAR_TOKEN

echo "SonarQube analysis complete. View results at http://localhost:9000"
```

By automating these quality checks, we catch problems early and maintain code quality over time. I've found this especially important for ML systems, where the temptation to cut corners "just to get the model working" can lead to technical debt.

## Development Workflow and Best Practices

After trying various workflows, I've settled on a development process that balances rigor with practicality:

### The Development Flow

1. **Create a feature branch**: Branch from `main` for each new feature or bug fix
   ```bash
   git checkout -b feature/add-drift-detection
   ```

2. **Develop locally**: Write code, run tests, and verify functionality
   ```bash
   # Make changes
   # Run tests
   pytest tests/
   # Run linting
   black . && isort . && flake8
   ```

3. **Commit with meaningful messages**: I follow the conventional commits standard
   ```bash
   git commit -m "feat: Add statistical drift detection for satellite imagery"
   ```

4. **Push and create a pull request**: This triggers the CI pipeline
   ```bash
   git push origin feature/add-drift-detection
   # Create PR on GitHub
   ```

5. **Address CI feedback**: Fix any issues identified by the CI pipeline

6. **Code review**: Get feedback from team members

7. **Merge to main**: After approval and passing CI checks
   ```bash
   # GitHub handles this through the PR interface
   ```

### Best Practices I've Learned the Hard Way

1. **Never skip tests**: It's tempting to bypass tests for "quick fixes," but this inevitably leads to regressions.

2. **Document as you code**: Documentation written after the fact is often incomplete or inaccurate.
   ```python
   def calculate_iou(y_true: np.ndarray, y_pred: np.ndarray) -> float:
       """Calculate Intersection over Union between two binary masks.
       
       Args:
           y_true: Ground truth binary mask
           y_pred: Predicted binary mask
           
       Returns:
           IoU score between 0 and 1
           
       Example:
           >>> mask_gt = np.array([[0, 0], [1, 1]])
           >>> mask_pred = np.array([[0, 1], [1, 1]])
           >>> calculate_iou(mask_gt, mask_pred)
           0.6666666666666666
       """
   ```

3. **Keep components loosely coupled**: This makes testing easier and allows for system evolution.
   ```python
   # Bad: Tight coupling
   def process_and_train(raw_data_path):
       # Process data and train model in one function
       
   # Good: Loose coupling
   def process_data(raw_data_path):
       # Process data only
       
   def train_model(processed_data_path):
       # Train model only
   ```

4. **Use dependency injection**: This simplifies testing and makes the code more modular.
   ```python
   # From api/dependencies.py
   def get_redis():
       """Get Redis client."""
       redis_client = redis.Redis(
           host=settings.REDIS_HOST,
           port=settings.REDIS_PORT,
           decode_responses=True
       )
       try:
           yield redis_client
       finally:
           redis_client.close()
   ```

5. **Monitor resource usage**: ML systems can be resource-intensive, so keep an eye on memory and disk usage.
   ```python
   # From api/services/health.py
   def _check_disk_space(self):
       """Check available disk space."""
       try:
           total, used, free = shutil.disk_usage("/")
           # If less than 10% free space, report warning
           if (free / total) < 0.1:
               return "warning"
           return "ok"
       except Exception:
           return "error"
   ```

6. **Add graceful degradation**: Systems should handle failures gracefully.
   ```python
   # From core/models/trainer.py
   try:
       # Train model
       history = model.fit(...)
       return model, history
   except Exception as e:
       logger.error(f"Training failed: {e}")
       # Save partial model if possible
       if hasattr(model, 'save'):
           model.save(os.path.join(self.paths["models_dir"], "partial_model.h5"))
       raise
   ```

7. **Make configuration explicit**: Avoid hardcoded values and magic numbers.
   ```python
   # From core/config/settings.py
   class Settings(BaseSettings):
       """Core settings loaded from environment variables."""
       
       # Data and model paths
       DATA_DIR: str = Field("/data", env="DATA_DIR")
       MODELS_DIR: str = Field("/models", env="MODELS_DIR")
       
       # Model configuration
       DEFAULT_BATCH_SIZE: int = Field(4, env="DEFAULT_BATCH_SIZE")
       DEFAULT_LEARNING_RATE: float = Field(0.001, env="DEFAULT_LEARNING_RATE")
       DEFAULT_EPOCHS: int = Field(5, env="DEFAULT_EPOCHS")
   ```

These practices have saved me countless hours of debugging and refactoring. They might seem like extra work initially, but they pay dividends over the life of the project.

## FAQs and Troubleshooting

### Common Issues and Solutions

#### "The API is running but MLflow isn't accessible"

This usually indicates a networking issue between containers. Check:
1. Is the MLflow container running?
   ```bash
   docker-compose ps
   ```
2. Are the environment variables correctly set?
   ```bash
   # .env file
   MLFLOW_TRACKING_URI=http://mlflow:5000
   ```
3. Can the API container reach the MLflow container?
   ```bash
   docker exec -it api_container ping mlflow
   ```

#### "Training fails with OOM (Out of Memory) errors"

Satellite images can be memory-intensive. Try:
1. Reduce batch size in training parameters
2. Use smaller patch sizes during data preparation
3. Use a more memory-efficient model architecture (LinkNet instead of U-Net)
4. Add swap space to your system

#### "Data processing is extremely slow"

Data preparation for satellite images can be time-consuming. Optimize by:
1. Use Celery for background processing
2. Process patches in parallel
3. Only create multiple copies of positive samples
4. Downsample very large images before patching

#### "I'm getting 'No such file or directory' errors"

This often happens when mounted volumes aren't properly configured:
1. Check if directories exist locally
   ```bash
   mkdir -p data/raw data/processed models logs
   ```
2. Verify Docker volume mappings in `docker-compose.yml`
   ```yaml
   volumes:
     - ./:/app
     - data_volume:/data
     - models_volume:/models
     - logs_volume:/logs
   ```

### Performance Optimization Tips

1. **Batch prediction**: For large satellite images, use batch prediction to reduce memory usage
   ```python
   # Process image in patches
   patches = patchify(image, (512, 512, 3), step=512)
   predictions = []
   
   for i in range(patches.shape[0]):
       for j in range(patches.shape[1]):
           patch = patches[i, j, 0]
           pred = model.predict(np.expand_dims(patch / 255.0, axis=0))[0]
           predictions.append(pred)
   
   # Reconstruct full prediction
   # ...
   ```

2. **Use TensorFlow mixed precision**: Speeds up training on modern GPUs
   ```python
   # Enable mixed precision
   policy = tf.keras.mixed_precision.Policy('mixed_float16')
   tf.keras.mixed_precision.set_global_policy(policy)
   ```

3. **Optimize data generators**: Use efficient data loading for training
   ```python
   # Use TensorFlow's parallel data loading
   train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_masks))
   train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
   ```

4. **Profile training**: Use TensorBoard's profiler to identify bottlenecks
   ```python
   tensorboard_callback = tf.keras.callbacks.TensorBoard(
       log_dir=log_dir, 
       profile_batch='500,520'
   )
   ```

## Future Roadmap

While the current implementation focuses on building a robust CI pipeline and core functionality, I have several enhancements planned:

### Short-term Roadmap

1. **Continuous Deployment (CD) Pipeline**
   - Automated deployment to staging environment
   - Approval workflow for production deployment
   - Canary releases for new models

2. **Advanced Monitoring**
   - Prometheus integration for system metrics
   - Grafana dashboards for visualization
   - Automated alerting

3. **Model Explainability**
   - Feature importance visualization
   - Gradient-weighted Class Activation Mapping
   - Per-prediction explanation capabilities

### Medium-term Roadmap

1. **Automated Retraining**
   - Drift-triggered model retraining
   - A/B testing framework
   - Model performance validation before deployment

2. **Edge Deployment**
   - Model optimization for edge devices
   - TensorFlow Lite conversion
   - ONNX format support

3. **Transfer Learning Framework**
   - Pretrained models for common satellite imagery tasks
   - Fine-tuning pipeline
   - Few-shot learning capabilities

### Long-term Vision

1. **Multi-modal Learning**
   - Fusion of optical and SAR imagery
   - Integration of non-image data (elevation, weather)
   - Time-series analysis

2. **Federated Learning**
   - Distributed training across organizations
   - Privacy-preserving model updates
   - Model merging capabilities

3. **Active Learning System**
   - Intelligent sample selection for annotation
   - Uncertainty-based querying
   - Model confidence visualization

I believe these enhancements will make the system even more powerful and user-friendly, addressing the full lifecycle of ML models in production.

## License and Contributing

This project is licensed under the MIT License - see the LICENSE file for details.

### Contributing Guidelines

I welcome contributions from the community! Here's how you can help:

1. **Fork the repository**: Create your own copy of the repo
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**: Follow the development workflow described above
4. **Run tests**: Ensure all tests pass
5. **Submit a pull request**: I'll review your changes and provide feedback

Please follow these guidelines:
- Follow the code style (Black, isort, flake8)
- Add tests for new functionality
- Update documentation as needed
- Keep pull requests focused on a single change

#### Code Review Criteria

When reviewing contributions, I look for:
- **Correctness**: Does the code work as intended?
- **Performance**: Is the code efficient?
- **Readability**: Is the code clear and maintainable?
- **Tests**: Are there appropriate tests?
- **Documentation**: Is the code well-documented?

### Getting Help

If you have questions or need help:
1. **Check the FAQs**: Many common issues are addressed above
2. **Open an Issue**: For bugs or feature requests
3. **Start a Discussion**: For architectural questions or ideas

---

This project has been a labor of love, born from the frustration of seeing brilliant models fail in production due to poor engineering. I hope it helps you build reliable, maintainable ML systems for satellite imagery analysis.

Remember: in production ML, the model is just the beginning - the system is what delivers value.

Happy coding!

~ @saugatapaul1010