"""
Observability and Monitoring for RAG Systems
---------------------------------------------
Track performance, latency, and quality metrics for production RAG.
"""

import time
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
import threading
from collections import defaultdict


@dataclass
class QueryMetrics:
    """Metrics for a single query."""
    query_id: str
    query_text: str
    timestamp: float

    # Latency metrics (milliseconds)
    total_latency: float = 0.0
    embedding_latency: float = 0.0
    search_latency: float = 0.0
    rerank_latency: float = 0.0
    generation_latency: float = 0.0

    # Search metrics
    num_results: int = 0
    top_score: float = 0.0
    avg_score: float = 0.0

    # Quality metrics (if available)
    relevance_score: Optional[float] = None
    user_feedback: Optional[str] = None

    # Context
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceTracker:
    """
    Track performance metrics with minimal overhead.

    Designed to have <5ms overhead per query.
    """

    def __init__(self):
        self.metrics: List[QueryMetrics] = []
        self._lock = threading.Lock()

    def start_query(self, query_id: str, query_text: str) -> QueryMetrics:
        """Start tracking a new query."""
        return QueryMetrics(
            query_id=query_id,
            query_text=query_text,
            timestamp=time.time()
        )

    def record_metrics(self, metrics: QueryMetrics):
        """Record completed query metrics."""
        with self._lock:
            self.metrics.append(metrics)

    def get_stats(self, last_n: Optional[int] = None) -> Dict:
        """Get aggregate statistics."""
        with self._lock:
            metrics = self.metrics[-last_n:] if last_n else self.metrics

        if not metrics:
            return {}

        total_latencies = [m.total_latency for m in metrics]
        search_latencies = [m.search_latency for m in metrics]

        return {
            "total_queries": len(metrics),
            "avg_total_latency_ms": sum(total_latencies) / len(total_latencies),
            "p50_latency_ms": self._percentile(total_latencies, 50),
            "p95_latency_ms": self._percentile(total_latencies, 95),
            "p99_latency_ms": self._percentile(total_latencies, 99),
            "avg_search_latency_ms": sum(search_latencies) / len(search_latencies),
            "avg_results": sum(m.num_results for m in metrics) / len(metrics)
        }

    def _percentile(self, values: List[float], p: float) -> float:
        """Calculate percentile."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        idx = int(len(sorted_values) * p / 100)
        return sorted_values[min(idx, len(sorted_values) - 1)]

    def export_to_file(self, filepath: str):
        """Export metrics to JSON file."""
        with self._lock:
            data = [asdict(m) for m in self.metrics]

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


class LatencyMonitor:
    """
    Context manager for tracking latency of operations.

    Usage:
        with LatencyMonitor() as monitor:
            # Do work
            pass
        latency_ms = monitor.latency_ms
    """

    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.latency_ms: float = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.latency_ms = (self.end_time - self.start_time) * 1000


class QueryLogger:
    """
    Log queries and results for debugging and analysis.
    """

    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create daily log file
        date_str = datetime.now().strftime("%Y-%m-%d")
        self.log_file = self.log_dir / f"queries_{date_str}.jsonl"

    def log_query(
        self,
        query_id: str,
        query_text: str,
        results: List[Dict],
        latency_ms: float,
        metadata: Optional[Dict] = None
    ):
        """Log a query and its results."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query_id": query_id,
            "query": query_text,
            "latency_ms": latency_ms,
            "num_results": len(results),
            "top_score": results[0]["score"] if results else 0.0,
            "metadata": metadata or {}
        }

        # Append to JSONL file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')


class AlertManager:
    """
    Monitor metrics and trigger alerts when thresholds are exceeded.
    """

    def __init__(
        self,
        latency_threshold_ms: float = 300,
        error_rate_threshold: float = 0.05,
        window_size: int = 100
    ):
        self.latency_threshold = latency_threshold_ms
        self.error_rate_threshold = error_rate_threshold
        self.window_size = window_size

        self.recent_latencies: List[float] = []
        self.recent_errors: List[bool] = []
        self.alerts: List[Dict] = []

    def record_query(self, latency_ms: float, error: bool = False):
        """Record a query execution."""
        self.recent_latencies.append(latency_ms)
        self.recent_errors.append(error)

        # Keep only recent window
        if len(self.recent_latencies) > self.window_size:
            self.recent_latencies.pop(0)
            self.recent_errors.pop(0)

        # Check for alerts
        self._check_alerts()

    def _check_alerts(self):
        """Check if any alert conditions are met."""
        if len(self.recent_latencies) < 10:
            return

        # Check P95 latency
        p95 = self._percentile(self.recent_latencies, 95)
        if p95 > self.latency_threshold:
            self._trigger_alert(
                "high_latency",
                f"P95 latency {p95:.1f}ms exceeds threshold {self.latency_threshold}ms"
            )

        # Check error rate
        error_rate = sum(self.recent_errors) / len(self.recent_errors)
        if error_rate > self.error_rate_threshold:
            self._trigger_alert(
                "high_error_rate",
                f"Error rate {error_rate:.2%} exceeds threshold {self.error_rate_threshold:.2%}"
            )

    def _trigger_alert(self, alert_type: str, message: str):
        """Trigger an alert."""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "type": alert_type,
            "message": message
        }
        self.alerts.append(alert)
        print(f"[ALERT] {alert_type}: {message}")

    def _percentile(self, values: List[float], p: float) -> float:
        """Calculate percentile."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        idx = int(len(sorted_values) * p / 100)
        return sorted_values[min(idx, len(sorted_values) - 1)]

    def get_alerts(self) -> List[Dict]:
        """Get all triggered alerts."""
        return self.alerts.copy()


class MetricsDashboard:
    """
    Simple metrics dashboard for monitoring.
    """

    def __init__(self, tracker: PerformanceTracker):
        self.tracker = tracker

    def print_summary(self, last_n: Optional[int] = None):
        """Print a summary of metrics."""
        stats = self.tracker.get_stats(last_n=last_n)

        if not stats:
            print("No metrics available yet.")
            return

        print("\n" + "=" * 60)
        print("RAG SYSTEM PERFORMANCE METRICS")
        print("=" * 60)
        print(f"Total Queries:        {stats['total_queries']}")
        print(f"Avg Total Latency:    {stats['avg_total_latency_ms']:.2f} ms")
        print(f"P50 Latency:          {stats['p50_latency_ms']:.2f} ms")
        print(f"P95 Latency:          {stats['p95_latency_ms']:.2f} ms")
        print(f"P99 Latency:          {stats['p99_latency_ms']:.2f} ms")
        print(f"Avg Search Latency:   {stats['avg_search_latency_ms']:.2f} ms")
        print(f"Avg Results:          {stats['avg_results']:.1f}")
        print("=" * 60 + "\n")

    def check_health(self, latency_target_ms: float = 300) -> Dict[str, Any]:
        """Check system health against targets."""
        stats = self.tracker.get_stats(last_n=100)

        if not stats:
            return {"status": "no_data"}

        p95_latency = stats['p95_latency_ms']
        health_status = "healthy" if p95_latency < latency_target_ms else "degraded"

        return {
            "status": health_status,
            "p95_latency_ms": p95_latency,
            "target_latency_ms": latency_target_ms,
            "meeting_sla": p95_latency < latency_target_ms
        }
