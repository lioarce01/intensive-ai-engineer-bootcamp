"""
Example script for training a model using the MLOps API.
"""

import requests
import json
from sklearn.datasets import load_iris

# Load sample data
iris = load_iris()
X = iris.data.tolist()[:100]  # Use subset for demo
y = iris.target.tolist()[:100]

# API endpoint
API_URL = "http://localhost:8000"

def train_model():
    """Train a model using the API."""
    print("Training model...")

    payload = {
        "data": X,
        "target": y,
        "algorithm": "random_forest",
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": 10
        }
    }

    response = requests.post(f"{API_URL}/train", json=payload)

    if response.status_code == 200:
        result = response.json()
        print(f"\n✓ Training successful!")
        print(f"  Run ID: {result['run_id']}")
        print(f"  Accuracy: {result['metrics']['accuracy']:.4f}")
        print(f"  Training time: {result['training_time']:.2f}s")
        print(f"  Model path: {result['model_path']}")
        return result['run_id']
    else:
        print(f"✗ Training failed: {response.status_code}")
        print(response.text)
        return None


def make_prediction(features):
    """Make a prediction using the API."""
    print("\nMaking prediction...")

    payload = {
        "features": features
    }

    response = requests.post(f"{API_URL}/predict", json=payload)

    if response.status_code == 200:
        result = response.json()
        print(f"\n✓ Prediction successful!")
        print(f"  Predictions: {result['predictions']}")
        print(f"  Model version: {result['model_version']}")
        print(f"  Inference time: {result['inference_time']:.4f}s")
        return result
    else:
        print(f"✗ Prediction failed: {response.status_code}")
        print(response.text)
        return None


def get_metrics():
    """Get metrics summary."""
    print("\nFetching metrics...")

    response = requests.get(f"{API_URL}/metrics/summary")

    if response.status_code == 200:
        metrics = response.json()
        print(f"\n✓ Metrics Summary:")
        print(f"  Total predictions: {metrics['total_predictions']}")
        print(f"  Total trainings: {metrics['total_trainings']}")
        print(f"  Total errors: {metrics['total_errors']}")
        print(f"  Avg inference time: {metrics['avg_inference_time']:.4f}s")
        print(f"  Avg training time: {metrics['avg_training_time']:.2f}s")
        return metrics
    else:
        print(f"✗ Failed to get metrics: {response.status_code}")
        return None


if __name__ == "__main__":
    print("=" * 60)
    print("MLOps API - Training Example")
    print("=" * 60)

    # Train model
    run_id = train_model()

    if run_id:
        # Make predictions
        test_features = [[5.1, 3.5, 1.4, 0.2], [6.2, 2.9, 4.3, 1.3]]
        make_prediction(test_features)

        # Get metrics
        get_metrics()

    print("\n" + "=" * 60)
    print("Check the following URLs:")
    print(f"  API Docs: {API_URL}/docs")
    print(f"  MLflow: http://localhost:5000")
    print(f"  Prometheus: http://localhost:9090")
    print(f"  Grafana: http://localhost:3000")
    print(f"  Jaeger: http://localhost:16686")
    print("=" * 60)
