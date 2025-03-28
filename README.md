satellite-segmentation-mlops/
├── .github/workflows/
│   └── ci.yml                     # CI pipeline configuration
├── api/
│   ├── main.py                    # API entry point
│   ├── config.py                  # API configuration
│   ├── dependencies.py            # Dependency injection
│   ├── routers/
│   │   ├── __init__.py            # Make routers a package
│   │   ├── health.py              # Health check endpoints
│   │   ├── data.py                # Data management endpoints
│   │   └── training.py            # Training endpoints
│   ├── models/
│   │   ├── __init__.py            # Make models a package
│   │   ├── data.py                # Data Pydantic models
│   │   └── training.py            # Training Pydantic models
│   └── services/
│       ├── __init__.py            # Make services a package
│       ├── health.py              # Health check service
│       ├── data.py                # Data service
│       └── training.py            # Training service
├── core/
│   ├── __init__.py                # Make core a package
│   ├── config/
│   │   ├── __init__.py            # Make config a package
│   │   └── settings.py            # Configuration settings
│   ├── data/
│   │   ├── __init__.py            # Make data a package
│   │   └── data_preparation.py    # Data processing
│   ├── models/
│   │   ├── __init__.py            # Make models a package
│   │   ├── trainer.py             # Model training
│   │   └── metrics.py             # Model evaluation metrics
│   └── utils/
│       ├── __init__.py            # Make utils a package
│       └── logging.py             # Logging utilities
├── tasks/
│   ├── __init__.py                # Make tasks a package
│   ├── celery_app.py              # Celery configuration
│   ├── data_tasks.py              # Data processing tasks
│   ├── training_tasks.py          # Training tasks
│   └── monitoring_tasks.py        # Monitoring tasks
├── infrastructure/
│   ├── docker/
│   │   ├── api/
│   │   │   └── Dockerfile         # API service Dockerfile
│   │   ├── worker/
│   │   │   └── Dockerfile         # Celery worker Dockerfile
│   │   └── mlflow/
│   │       └── Dockerfile         # MLflow server Dockerfile
│   └── scripts/
│       └── run_sonar_analysis.sh  # SonarQube analysis script
├── tests/
│   ├── __init__.py                # Make tests a package
│   ├── conftest.py                # Pytest fixtures
│   ├── test_api/
│   │   ├── __init__.py
│   │   ├── test_health.py
│   │   ├── test_data_routes.py
│   │   └── test_training_routes.py
│   ├── test_core/
│   │   ├── __init__.py
│   │   ├── test_data_preparation.py
│   │   └── test_model_trainer.py
│   └── test_tasks/
│       ├── __init__.py
│       ├── test_data_tasks.py
│       └── test_training_tasks.py
├── docker-compose.yml             # Service orchestration
├── pyproject.toml                 # Project metadata and dependencies
├── requirements.txt               # Python dependencies
├── pytest.ini                     # Pytest configuration
├── sonar-project.properties       # SonarQube configuration
├── .env.example                   # Example environment variables
├── .gitignore                     # Git ignore patterns
├── .dvcignore                     # DVC ignore patterns
└── README.md                      # Project documentation