"""
Pydantic models for request and response validation.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request model for predictions."""
    features: List[List[float]] = Field(
        ...,
        description="List of feature vectors for prediction",
        example=[[5.1, 3.5, 1.4, 0.2], [6.2, 2.9, 4.3, 1.3]]
    )


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predictions: List[int] = Field(..., description="Model predictions")
    model_version: str = Field(..., description="Version of the model used")
    inference_time: float = Field(..., description="Inference time in seconds")


class TrainingRequest(BaseModel):
    """Request model for training."""
    data: List[List[float]] = Field(
        ...,
        description="Training data features",
        example=[[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2]]
    )
    target: List[int] = Field(
        ...,
        description="Training data labels",
        example=[0, 0]
    )
    algorithm: str = Field(
        default="random_forest",
        description="ML algorithm to use",
        example="random_forest"
    )
    hyperparameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Hyperparameters for the algorithm",
        example={"n_estimators": 100, "max_depth": 10}
    )


class TrainingResponse(BaseModel):
    """Response model for training."""
    success: bool = Field(..., description="Training success status")
    run_id: str = Field(..., description="MLflow run ID")
    metrics: Dict[str, float] = Field(..., description="Training metrics")
    model_path: str = Field(..., description="Path to saved model")
    training_time: float = Field(..., description="Training time in seconds")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Health status")
    model_loaded: bool = Field(..., description="Whether a model is loaded")
    timestamp: float = Field(..., description="Timestamp of health check")


class MetricsResponse(BaseModel):
    """Response model for metrics summary."""
    total_predictions: int = Field(..., description="Total number of predictions")
    total_trainings: int = Field(..., description="Total number of training runs")
    total_errors: int = Field(..., description="Total number of errors")
    avg_inference_time: float = Field(..., description="Average inference time")
    avg_training_time: float = Field(..., description="Average training time")
