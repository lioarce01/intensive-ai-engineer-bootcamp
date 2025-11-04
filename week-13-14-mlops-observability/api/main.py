"""
FastAPI application with ML endpoints, observability, and monitoring.
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Any

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

from .models import (
    PredictionRequest,
    PredictionResponse,
    TrainingRequest,
    TrainingResponse,
    HealthResponse,
    MetricsResponse
)
from .ml_service import MLService
from .observability import setup_logging, get_metrics_collector

# Setup logging
logger = setup_logging()

# Initialize ML service
ml_service = MLService()

# Setup OpenTelemetry
def setup_tracing():
    """Configure OpenTelemetry distributed tracing."""
    trace.set_tracer_provider(TracerProvider())
    otlp_exporter = OTLPSpanExporter(
        endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://jaeger:4317"),
        insecure=True
    )
    trace.get_tracer_provider().add_span_processor(
        BatchSpanProcessor(otlp_exporter)
    )

setup_tracing()
tracer = trace.get_tracer(__name__)

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("Starting MLOps API...")

    # Startup: Load default model if available
    try:
        ml_service.load_model()
        logger.info("Default model loaded successfully")
    except Exception as e:
        logger.warning(f"No default model found: {e}")

    yield

    # Shutdown
    logger.info("Shutting down MLOps API...")

# Create FastAPI app
app = FastAPI(
    title="MLOps API",
    description="Production ML API with observability and monitoring",
    version="1.0.0",
    lifespan=lifespan
)

# Setup Prometheus metrics
instrumentator = Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_respect_env_var=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics"],
    env_var_name="ENABLE_METRICS",
    inprogress_name="http_requests_inprogress",
    inprogress_labels=True,
)
instrumentator.instrument(app).expose(app, include_in_schema=False)

# Setup FastAPI instrumentation for OpenTelemetry
FastAPIInstrumentor.instrument_app(app)

# Get metrics collector
metrics = get_metrics_collector()


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "MLOps API with Observability",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    with tracer.start_as_current_span("health_check"):
        is_healthy = ml_service.is_healthy()
        model_loaded = ml_service.model is not None

        status_code = status.HTTP_200_OK if is_healthy else status.HTTP_503_SERVICE_UNAVAILABLE

        return JSONResponse(
            status_code=status_code,
            content={
                "status": "healthy" if is_healthy else "unhealthy",
                "model_loaded": model_loaded,
                "timestamp": time.time()
            }
        )


@app.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest):
    """
    Train a new model with the provided data.
    Logs experiment to MLflow.
    """
    with tracer.start_as_current_span("train_model") as span:
        try:
            logger.info(f"Starting model training with algorithm: {request.algorithm}")
            span.set_attribute("ml.algorithm", request.algorithm)

            start_time = time.time()

            # Train model
            result = ml_service.train(
                data=request.data,
                target=request.target,
                algorithm=request.algorithm,
                hyperparameters=request.hyperparameters
            )

            training_time = time.time() - start_time

            # Record metrics
            metrics.record_training(
                algorithm=request.algorithm,
                duration=training_time,
                accuracy=result["metrics"]["accuracy"]
            )

            span.set_attribute("ml.accuracy", result["metrics"]["accuracy"])
            span.set_attribute("ml.training_time", training_time)

            logger.info(f"Training completed in {training_time:.2f}s with accuracy: {result['metrics']['accuracy']:.4f}")

            return TrainingResponse(
                success=True,
                run_id=result["run_id"],
                metrics=result["metrics"],
                model_path=result["model_path"],
                training_time=training_time
            )

        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            span.set_attribute("error", True)
            metrics.record_error("training")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Training failed: {str(e)}"
            )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make predictions using the loaded model.
    """
    with tracer.start_as_current_span("predict") as span:
        try:
            if ml_service.model is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No model loaded. Train a model first."
                )

            logger.info(f"Making prediction for {len(request.features)} samples")
            span.set_attribute("ml.num_samples", len(request.features))

            start_time = time.time()

            # Make prediction
            predictions = ml_service.predict(request.features)

            inference_time = time.time() - start_time

            # Record metrics
            metrics.record_prediction(
                num_samples=len(request.features),
                inference_time=inference_time
            )

            span.set_attribute("ml.inference_time", inference_time)

            logger.info(f"Prediction completed in {inference_time:.4f}s")

            return PredictionResponse(
                predictions=predictions.tolist(),
                model_version=ml_service.model_version,
                inference_time=inference_time
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}", exc_info=True)
            span.set_attribute("error", True)
            metrics.record_error("prediction")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed: {str(e)}"
            )


@app.get("/model/info", response_model=Dict[str, Any])
async def get_model_info():
    """Get information about the currently loaded model."""
    with tracer.start_as_current_span("get_model_info"):
        if ml_service.model is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No model loaded"
            )

        return {
            "model_version": ml_service.model_version,
            "model_type": type(ml_service.model).__name__,
            "loaded_at": ml_service.model_loaded_at
        }


@app.get("/metrics/summary", response_model=MetricsResponse)
async def get_metrics_summary():
    """Get summary of application metrics."""
    with tracer.start_as_current_span("get_metrics_summary"):
        summary = metrics.get_summary()
        return MetricsResponse(**summary)


@app.post("/model/load/{run_id}")
async def load_model_by_run(run_id: str):
    """Load a specific model by MLflow run ID."""
    with tracer.start_as_current_span("load_model") as span:
        try:
            span.set_attribute("ml.run_id", run_id)
            ml_service.load_model(run_id=run_id)
            logger.info(f"Model loaded successfully: {run_id}")

            return {
                "success": True,
                "message": f"Model {run_id} loaded successfully",
                "model_version": ml_service.model_version
            }
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}", exc_info=True)
            span.set_attribute("error", True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load model: {str(e)}"
            )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
