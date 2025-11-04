"""
Machine Learning service for training and prediction with MLflow integration.
"""

import os
import time
from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import joblib

from .observability import setup_logging

logger = setup_logging()


class MLService:
    """Service for ML operations with MLflow tracking."""

    def __init__(self):
        """Initialize ML service."""
        self.model = None
        self.model_version = None
        self.model_loaded_at = None

        # Configure MLflow
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("mlops-production")

        logger.info(f"MLflow tracking URI: {mlflow_uri}")

    def _get_algorithm(self, algorithm: str, hyperparameters: Optional[Dict] = None):
        """Get ML algorithm instance."""
        algorithms = {
            "random_forest": RandomForestClassifier,
            "gradient_boosting": GradientBoostingClassifier,
            "logistic_regression": LogisticRegression
        }

        if algorithm not in algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm}. Available: {list(algorithms.keys())}")

        algo_class = algorithms[algorithm]
        params = hyperparameters or {}

        # Set default parameters
        if algorithm == "random_forest" and not params:
            params = {"n_estimators": 100, "max_depth": 10, "random_state": 42}
        elif algorithm == "gradient_boosting" and not params:
            params = {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42}
        elif algorithm == "logistic_regression" and not params:
            params = {"max_iter": 1000, "random_state": 42}

        return algo_class(**params)

    def train(
        self,
        data: List[List[float]],
        target: List[int],
        algorithm: str = "random_forest",
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Train a model and log to MLflow.

        Args:
            data: Training features
            target: Training labels
            algorithm: Algorithm to use
            hyperparameters: Model hyperparameters

        Returns:
            Dictionary with training results
        """
        with mlflow.start_run() as run:
            logger.info(f"Starting training with {algorithm}")

            # Convert to numpy arrays
            X = np.array(data)
            y = np.array(target)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Log parameters
            mlflow.log_param("algorithm", algorithm)
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_param("n_features", X.shape[1])

            if hyperparameters:
                for param, value in hyperparameters.items():
                    mlflow.log_param(param, value)

            # Train model
            start_time = time.time()
            model = self._get_algorithm(algorithm, hyperparameters)
            model.fit(X_train, y_train)
            training_time = time.time() - start_time

            # Evaluate
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred_test),
                "precision": precision_score(y_test, y_pred_test, average="weighted", zero_division=0),
                "recall": recall_score(y_test, y_pred_test, average="weighted", zero_division=0),
                "f1": f1_score(y_test, y_pred_test, average="weighted", zero_division=0),
                "train_accuracy": accuracy_score(y_train, y_pred_train),
                "training_time": training_time
            }

            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Log model
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name=f"{algorithm}_model"
            )

            # Save model locally
            model_dir = "/app/models"
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"model_{run.info.run_id}.joblib")
            joblib.dump(model, model_path)
            mlflow.log_artifact(model_path)

            # Update current model
            self.model = model
            self.model_version = run.info.run_id
            self.model_loaded_at = time.time()

            logger.info(f"Training completed. Run ID: {run.info.run_id}, Accuracy: {metrics['accuracy']:.4f}")

            return {
                "run_id": run.info.run_id,
                "metrics": metrics,
                "model_path": model_path
            }

    def predict(self, features: List[List[float]]) -> np.ndarray:
        """
        Make predictions using the loaded model.

        Args:
            features: Input features

        Returns:
            Model predictions
        """
        if self.model is None:
            raise ValueError("No model loaded")

        X = np.array(features)
        predictions = self.model.predict(X)

        logger.info(f"Made predictions for {len(features)} samples")

        return predictions

    def load_model(self, run_id: Optional[str] = None, model_path: Optional[str] = None):
        """
        Load a model from MLflow or local path.

        Args:
            run_id: MLflow run ID
            model_path: Local model path
        """
        try:
            if run_id:
                # Load from MLflow
                model_uri = f"runs:/{run_id}/model"
                self.model = mlflow.sklearn.load_model(model_uri)
                self.model_version = run_id
                logger.info(f"Model loaded from MLflow: {run_id}")
            elif model_path:
                # Load from local path
                self.model = joblib.load(model_path)
                self.model_version = os.path.basename(model_path)
                logger.info(f"Model loaded from path: {model_path}")
            else:
                # Try to load latest model
                model_dir = "/app/models"
                if os.path.exists(model_dir):
                    models = [f for f in os.listdir(model_dir) if f.endswith(".joblib")]
                    if models:
                        latest_model = sorted(models)[-1]
                        model_path = os.path.join(model_dir, latest_model)
                        self.model = joblib.load(model_path)
                        self.model_version = latest_model
                        logger.info(f"Loaded latest model: {latest_model}")
                    else:
                        raise FileNotFoundError("No models found")
                else:
                    raise FileNotFoundError("Models directory not found")

            self.model_loaded_at = time.time()

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def is_healthy(self) -> bool:
        """Check if service is healthy."""
        # Check MLflow connectivity
        try:
            mlflow.get_tracking_uri()
            return True
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False
