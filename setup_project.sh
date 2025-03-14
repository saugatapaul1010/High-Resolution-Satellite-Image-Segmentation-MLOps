# Create main project directory
mkdir -p satellite-segmentation-mlops

# Change to project directory
cd satellite-segmentation-mlops

# Create main subdirectories
mkdir -p .github/workflows
mkdir -p api/routers api/models api/services
mkdir -p core/config core/data core/models core/utils
mkdir -p tasks
mkdir -p infrastructure/docker/api infrastructure/docker/worker infrastructure/docker/mlflow infrastructure/scripts
mkdir -p tests/test_api tests/test_core tests/test_tasks

# Create __init__.py files to make Python recognize directories as packages
touch api/__init__.py
touch api/routers/__init__.py
touch api/models/__init__.py
touch api/services/__init__.py
touch core/__init__.py
touch core/config/__init__.py
touch core/data/__init__.py
touch core/models/__init__.py
touch core/utils/__init__.py
touch tasks/__init__.py
touch tests/__init__.py
touch tests/test_api/__init__.py
touch tests/test_core/__init__.py
touch tests/test_tasks/__init__.py

# Create API files
touch api/main.py
touch api/config.py
touch api/dependencies.py
touch api/routers/health.py
touch api/routers/data.py
touch api/routers/training.py
touch api/models/data.py
touch api/models/training.py
touch api/services/health.py
touch api/services/data.py
touch api/services/training.py

# Create core files
touch core/config/settings.py
touch core/data/data_preparation.py
touch core/models/trainer.py
touch core/models/metrics.py
touch core/utils/logging.py

# Create task files
touch tasks/celery_app.py
touch tasks/data_tasks.py
touch tasks/training_tasks.py
touch tasks/monitoring_tasks.py

# Create Docker files
touch infrastructure/docker/api/Dockerfile
touch infrastructure/docker/worker/Dockerfile
touch infrastructure/docker/mlflow/Dockerfile
touch infrastructure/scripts/run_sonar_analysis.sh

# Create test files
touch tests/conftest.py
touch tests/test_api/test_health.py
touch tests/test_api/test_data_routes.py
touch tests/test_api/test_training_routes.py
touch tests/test_core/test_data_preparation.py
touch tests/test_core/test_model_trainer.py
touch tests/test_tasks/test_data_tasks.py
touch tests/test_tasks/test_training_tasks.py

# Create configuration files
touch docker-compose.yml
touch pyproject.toml
touch requirements.txt
touch pytest.ini
touch sonar-project.properties
touch .env.example
touch .gitignore
touch .dvcignore
touch README.md

# Make shell script executable
chmod +x infrastructure/scripts/run_sonar_analysis.sh
