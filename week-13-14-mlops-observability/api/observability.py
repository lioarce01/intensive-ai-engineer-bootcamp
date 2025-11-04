"""
Observability module for logging, metrics, and tracing.
"""

import logging
import sys
from typing import Dict
from pythonjsonlogger import jsonlogger


class MetricsCollector:
    """Collector for application metrics."""

    def __init__(self):
        """Initialize metrics collector."""
        self.metrics = {
            "predictions": {
                "count": 0,
                "total_time": 0.0,
                "total_samples": 0
            },
            "trainings": {
                "count": 0,
                "total_time": 0.0,
                "algorithms": {}
            },
            "errors": {
                "prediction": 0,
                "training": 0,
                "total": 0
            }
        }

    def record_prediction(self, num_samples: int, inference_time: float):
        """Record prediction metrics."""
        self.metrics["predictions"]["count"] += 1
        self.metrics["predictions"]["total_time"] += inference_time
        self.metrics["predictions"]["total_samples"] += num_samples

    def record_training(self, algorithm: str, duration: float, accuracy: float):
        """Record training metrics."""
        self.metrics["trainings"]["count"] += 1
        self.metrics["trainings"]["total_time"] += duration

        if algorithm not in self.metrics["trainings"]["algorithms"]:
            self.metrics["trainings"]["algorithms"][algorithm] = {
                "count": 0,
                "total_time": 0.0,
                "best_accuracy": 0.0
            }

        algo_metrics = self.metrics["trainings"]["algorithms"][algorithm]
        algo_metrics["count"] += 1
        algo_metrics["total_time"] += duration
        algo_metrics["best_accuracy"] = max(algo_metrics["best_accuracy"], accuracy)

    def record_error(self, error_type: str):
        """Record error metrics."""
        if error_type in self.metrics["errors"]:
            self.metrics["errors"][error_type] += 1
        self.metrics["errors"]["total"] += 1

    def get_summary(self) -> Dict:
        """Get metrics summary."""
        pred_count = self.metrics["predictions"]["count"]
        train_count = self.metrics["trainings"]["count"]

        return {
            "total_predictions": pred_count,
            "total_trainings": train_count,
            "total_errors": self.metrics["errors"]["total"],
            "avg_inference_time": (
                self.metrics["predictions"]["total_time"] / pred_count
                if pred_count > 0 else 0.0
            ),
            "avg_training_time": (
                self.metrics["trainings"]["total_time"] / train_count
                if train_count > 0 else 0.0
            )
        }


# Global metrics collector instance
_metrics_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    return _metrics_collector


def setup_logging() -> logging.Logger:
    """
    Setup structured JSON logging.

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("mlops")

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # JSON formatter
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger
