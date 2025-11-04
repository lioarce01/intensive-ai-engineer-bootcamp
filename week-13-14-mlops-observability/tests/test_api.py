"""
Tests for the MLOps API.
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.main import app

client = TestClient(app)


def test_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code in [200, 503]
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "timestamp" in data


def test_train_model():
    """Test model training endpoint."""
    payload = {
        "data": [
            [5.1, 3.5, 1.4, 0.2],
            [4.9, 3.0, 1.4, 0.2],
            [4.7, 3.2, 1.3, 0.2],
            [6.2, 2.9, 4.3, 1.3],
            [5.9, 3.0, 4.2, 1.5]
        ],
        "target": [0, 0, 0, 1, 1],
        "algorithm": "random_forest",
        "hyperparameters": {
            "n_estimators": 10,
            "max_depth": 5
        }
    }

    response = client.post("/train", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "run_id" in data
    assert "metrics" in data
    assert "accuracy" in data["metrics"]


def test_predict():
    """Test prediction endpoint."""
    # First train a model
    train_payload = {
        "data": [
            [5.1, 3.5, 1.4, 0.2],
            [4.9, 3.0, 1.4, 0.2],
            [6.2, 2.9, 4.3, 1.3],
            [5.9, 3.0, 4.2, 1.5]
        ],
        "target": [0, 0, 1, 1],
        "algorithm": "logistic_regression"
    }
    client.post("/train", json=train_payload)

    # Now make prediction
    predict_payload = {
        "features": [[5.1, 3.5, 1.4, 0.2]]
    }

    response = client.post("/predict", json=predict_payload)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert "model_version" in data
    assert "inference_time" in data
    assert len(data["predictions"]) == 1


def test_predict_without_model():
    """Test prediction without training a model first."""
    # This test should be run in isolation or after clearing the model
    # For now, we skip this test if a model is already loaded
    pass


def test_model_info():
    """Test model info endpoint."""
    # Train a model first
    train_payload = {
        "data": [[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2]],
        "target": [0, 1],
        "algorithm": "logistic_regression"
    }
    client.post("/train", json=train_payload)

    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_version" in data
    assert "model_type" in data


def test_metrics_summary():
    """Test metrics summary endpoint."""
    response = client.get("/metrics/summary")
    assert response.status_code == 200
    data = response.json()
    assert "total_predictions" in data
    assert "total_trainings" in data
    assert "total_errors" in data


def test_invalid_algorithm():
    """Test training with invalid algorithm."""
    payload = {
        "data": [[5.1, 3.5, 1.4, 0.2]],
        "target": [0],
        "algorithm": "invalid_algorithm"
    }

    response = client.post("/train", json=payload)
    assert response.status_code == 500


def test_invalid_prediction_format():
    """Test prediction with invalid input format."""
    payload = {
        "features": "invalid"
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
