# tests/test_api/test_health.py
import pytest
from unittest.mock import patch, MagicMock

def test_health_endpoint(api_client):
    """Test the health check endpoint."""
    with patch('api.services.health.HealthService._check_redis', return_value='ok'), \
         patch('api.services.health.HealthService._check_mlflow', return_value='ok'), \
         patch('api.services.health.HealthService._check_disk_space', return_value='ok'):
        
        response = api_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["redis"] == "ok"
        assert data["mlflow"] == "ok"
        assert data["disk"] == "ok"

def test_health_endpoint_with_warnings(api_client):
    """Test the health check endpoint with warnings."""
    with patch('api.services.health.HealthService._check_redis', return_value='ok'), \
         patch('api.services.health.HealthService._check_mlflow', return_value='ok'), \
         patch('api.services.health.HealthService._check_disk_space', return_value='warning'):
        
        response = api_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"  # overall status is error if any component is not ok
        assert data["redis"] == "ok"
        assert data["mlflow"] == "ok"
        assert data["disk"] == "warning"

def test_health_endpoint_with_errors(api_client):
    """Test the health check endpoint with errors."""
    with patch('api.services.health.HealthService._check_redis', return_value='error'), \
         patch('api.services.health.HealthService._check_mlflow', return_value='ok'), \
         patch('api.services.health.HealthService._check_disk_space', return_value='ok'):
        
        response = api_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"
        assert data["redis"] == "error"
        assert data["mlflow"] == "ok"
        assert data["disk"] == "ok"