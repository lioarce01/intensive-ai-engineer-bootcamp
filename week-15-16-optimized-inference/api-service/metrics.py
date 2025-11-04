"""
Prometheus Metrics for vLLM API
================================

Implements monitoring and observability using Prometheus metrics:
- Request counts
- Latency histograms (P50, P95, P99)
- Token throughput
- Cache hit rates
- Error rates

Installation:
    pip install prometheus-client

View metrics:
    curl http://localhost:8000/metrics
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
import time
import logging
from typing import Callable
from fastapi import Request, Response

logger = logging.getLogger(__name__)

# ============================================================================
# Prometheus Metrics
# ============================================================================

# Request counter
REQUEST_COUNT = Counter(
    "vllm_requests_total",
    "Total number of requests",
    ["method", "endpoint", "status"],
)

# Request latency histogram
REQUEST_LATENCY = Histogram(
    "vllm_request_duration_seconds",
    "Request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

# Token generation metrics
TOKENS_GENERATED = Counter(
    "vllm_tokens_generated_total",
    "Total number of tokens generated",
)

TOKEN_GENERATION_RATE = Gauge(
    "vllm_tokens_per_second",
    "Current token generation rate",
)

# Cache metrics
CACHE_HITS = Counter(
    "vllm_cache_hits_total",
    "Total number of cache hits",
)

CACHE_MISSES = Counter(
    "vllm_cache_misses_total",
    "Total number of cache misses",
)

# Error metrics
ERROR_COUNT = Counter(
    "vllm_errors_total",
    "Total number of errors",
    ["error_type"],
)

# Model metrics
MODEL_LOAD_TIME = Gauge(
    "vllm_model_load_seconds",
    "Time taken to load the model",
)

GPU_MEMORY_USAGE = Gauge(
    "vllm_gpu_memory_bytes",
    "GPU memory usage in bytes",
)

# Batch size metrics
BATCH_SIZE = Histogram(
    "vllm_batch_size",
    "Distribution of batch sizes",
    buckets=[1, 2, 4, 8, 16, 32, 64, 128],
)


# ============================================================================
# Middleware
# ============================================================================

async def metrics_middleware(request: Request, call_next: Callable) -> Response:
    """
    Middleware to track request metrics.

    Args:
        request: FastAPI request
        call_next: Next middleware/handler

    Returns:
        Response with metrics tracked
    """
    # Skip metrics endpoint itself
    if request.url.path == "/metrics":
        return await call_next(request)

    # Track request
    start_time = time.time()

    try:
        # Process request
        response = await call_next(request)

        # Record metrics
        duration = time.time() - start_time
        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=request.url.path,
        ).observe(duration)

        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code,
        ).inc()

        return response

    except Exception as e:
        # Record error
        ERROR_COUNT.labels(error_type=type(e).__name__).inc()
        logger.error(f"Request failed: {e}")
        raise


# ============================================================================
# Metric Collectors
# ============================================================================

class MetricsCollector:
    """
    Collects and exposes custom metrics.
    """

    def __init__(self):
        self.start_time = time.time()

    def record_generation(self, num_tokens: int, duration_seconds: float):
        """
        Record token generation metrics.

        Args:
            num_tokens: Number of tokens generated
            duration_seconds: Time taken to generate
        """
        TOKENS_GENERATED.inc(num_tokens)

        if duration_seconds > 0:
            tokens_per_second = num_tokens / duration_seconds
            TOKEN_GENERATION_RATE.set(tokens_per_second)

    def record_cache_hit(self):
        """Record cache hit."""
        CACHE_HITS.inc()

    def record_cache_miss(self):
        """Record cache miss."""
        CACHE_MISSES.inc()

    def record_batch_size(self, size: int):
        """
        Record batch size.

        Args:
            size: Size of the batch
        """
        BATCH_SIZE.observe(size)

    def record_gpu_memory(self, bytes_used: int):
        """
        Record GPU memory usage.

        Args:
            bytes_used: GPU memory in bytes
        """
        GPU_MEMORY_USAGE.set(bytes_used)

    def get_cache_hit_rate(self) -> float:
        """
        Calculate cache hit rate.

        Returns:
            Hit rate as percentage (0-100)
        """
        hits = CACHE_HITS._value.get()
        misses = CACHE_MISSES._value.get()
        total = hits + misses

        if total == 0:
            return 0.0

        return (hits / total) * 100

    def get_uptime_seconds(self) -> float:
        """
        Get service uptime in seconds.

        Returns:
            Uptime in seconds
        """
        return time.time() - self.start_time


# Global metrics collector instance
collector = MetricsCollector()


# ============================================================================
# Utility Functions
# ============================================================================

def get_metrics_summary() -> dict:
    """
    Get human-readable metrics summary.

    Returns:
        Dictionary with key metrics
    """
    return {
        "uptime_seconds": collector.get_uptime_seconds(),
        "total_requests": REQUEST_COUNT._metrics,
        "cache_hit_rate": collector.get_cache_hit_rate(),
        "total_tokens_generated": TOKENS_GENERATED._value.get(),
        "current_tokens_per_second": TOKEN_GENERATION_RATE._value.get(),
    }


def export_metrics() -> str:
    """
    Export metrics in Prometheus format.

    Returns:
        Prometheus-formatted metrics string
    """
    return generate_latest(REGISTRY).decode("utf-8")


# ============================================================================
# Example Monitoring Dashboard Config
# ============================================================================

GRAFANA_DASHBOARD_JSON = """
{
  "dashboard": {
    "title": "vLLM API Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(vllm_requests_total[5m])"
          }
        ]
      },
      {
        "title": "P95 Latency",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(vllm_request_duration_seconds_bucket[5m]))"
          }
        ]
      },
      {
        "title": "Token Throughput",
        "targets": [
          {
            "expr": "vllm_tokens_per_second"
          }
        ]
      },
      {
        "title": "Cache Hit Rate",
        "targets": [
          {
            "expr": "rate(vllm_cache_hits_total[5m]) / (rate(vllm_cache_hits_total[5m]) + rate(vllm_cache_misses_total[5m]))"
          }
        ]
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "rate(vllm_errors_total[5m])"
          }
        ]
      }
    ]
  }
}
"""


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Metrics Demo")
    print("=" * 60)

    # Simulate some requests
    print("\n1. Simulating requests...")
    for i in range(10):
        REQUEST_COUNT.labels(
            method="POST",
            endpoint="/generate",
            status=200,
        ).inc()

        REQUEST_LATENCY.labels(
            method="POST",
            endpoint="/generate",
        ).observe(0.05 + i * 0.01)

    # Simulate token generation
    print("\n2. Simulating token generation...")
    collector.record_generation(num_tokens=100, duration_seconds=0.5)
    collector.record_generation(num_tokens=150, duration_seconds=0.7)

    # Simulate cache operations
    print("\n3. Simulating cache operations...")
    for _ in range(7):
        collector.record_cache_hit()
    for _ in range(3):
        collector.record_cache_miss()

    # Get summary
    print("\n4. Metrics Summary:")
    summary = get_metrics_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")

    # Export metrics
    print("\n5. Prometheus Metrics (sample):")
    metrics = export_metrics()
    print(metrics[:500] + "...")

    print("\n✅ Metrics demo completed!")
