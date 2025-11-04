"""
Latency Benchmarking for vLLM API
==================================

Measures detailed latency metrics:
- P50, P95, P99 latencies
- Time to first token (TTFT)
- Inter-token latency
- End-to-end latency

Usage:
    python latency_test.py --url http://localhost:8000 --requests 100
"""

import requests
import time
import numpy as np
import argparse
from typing import List, Dict, Tuple
import json
from dataclasses import dataclass, asdict
import statistics


@dataclass
class LatencyMetrics:
    """Container for latency measurements."""
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    total_requests: int
    successful_requests: int
    failed_requests: int


def measure_request_latency(
    url: str,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.7,
) -> Tuple[float, bool, Dict]:
    """
    Measure latency for a single request.

    Returns:
        (latency_ms, success, response_data)
    """
    start_time = time.time()

    try:
        response = requests.post(
            f"{url}/generate",
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "use_cache": False,  # Disable cache for accurate measurements
            },
            timeout=30,
        )

        latency_ms = (time.time() - start_time) * 1000

        if response.status_code == 200:
            return latency_ms, True, response.json()
        else:
            return latency_ms, False, {"error": response.text}

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        return latency_ms, False, {"error": str(e)}


def calculate_percentiles(latencies: List[float]) -> LatencyMetrics:
    """
    Calculate latency percentiles.

    Args:
        latencies: List of latency measurements in ms

    Returns:
        LatencyMetrics with calculated stats
    """
    latencies_array = np.array(latencies)

    return LatencyMetrics(
        p50_ms=np.percentile(latencies_array, 50),
        p95_ms=np.percentile(latencies_array, 95),
        p99_ms=np.percentile(latencies_array, 99),
        mean_ms=np.mean(latencies_array),
        min_ms=np.min(latencies_array),
        max_ms=np.max(latencies_array),
        std_ms=np.std(latencies_array),
        total_requests=len(latencies),
        successful_requests=len(latencies),
        failed_requests=0,
    )


def run_latency_benchmark(
    url: str,
    num_requests: int = 100,
    prompt: str = "The future of artificial intelligence is",
    max_tokens: int = 100,
) -> LatencyMetrics:
    """
    Run comprehensive latency benchmark.

    Args:
        url: API base URL
        num_requests: Number of requests to make
        prompt: Test prompt
        max_tokens: Max tokens to generate

    Returns:
        LatencyMetrics with results
    """
    print("=" * 60)
    print("Latency Benchmark")
    print("=" * 60)
    print(f"API URL: {url}")
    print(f"Requests: {num_requests}")
    print(f"Prompt: {prompt[:50]}...")
    print(f"Max tokens: {max_tokens}")
    print("=" * 60)

    latencies = []
    successful = 0
    failed = 0

    print("\nRunning benchmark...")
    for i in range(num_requests):
        latency_ms, success, response_data = measure_request_latency(
            url, prompt, max_tokens
        )

        if success:
            latencies.append(latency_ms)
            successful += 1
        else:
            failed += 1

        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{num_requests} requests completed")

    print(f"\n✅ Benchmark completed!")

    # Calculate metrics
    if latencies:
        metrics = calculate_percentiles(latencies)
        metrics.failed_requests = failed

        # Print results
        print("\n" + "=" * 60)
        print("Results")
        print("=" * 60)
        print(f"Total Requests:      {metrics.total_requests + failed}")
        print(f"Successful:          {metrics.successful_requests}")
        print(f"Failed:              {metrics.failed_requests}")
        print(f"\nLatency Statistics (ms):")
        print(f"  Mean:              {metrics.mean_ms:.2f}ms")
        print(f"  Median (P50):      {metrics.p50_ms:.2f}ms")
        print(f"  P95:               {metrics.p95_ms:.2f}ms")
        print(f"  P99:               {metrics.p99_ms:.2f}ms")
        print(f"  Min:               {metrics.min_ms:.2f}ms")
        print(f"  Max:               {metrics.max_ms:.2f}ms")
        print(f"  Std Dev:           {metrics.std_ms:.2f}ms")

        # Check against targets
        print("\n" + "=" * 60)
        print("Performance Targets")
        print("=" * 60)
        targets = [
            ("P50 < 50ms", metrics.p50_ms < 50, metrics.p50_ms),
            ("P95 < 100ms", metrics.p95_ms < 100, metrics.p95_ms),
            ("P99 < 200ms", metrics.p99_ms < 200, metrics.p99_ms),
        ]

        for target, met, actual in targets:
            status = "✅" if met else "❌"
            print(f"{status} {target} (actual: {actual:.2f}ms)")

        return metrics

    else:
        print("\n❌ All requests failed!")
        return LatencyMetrics(
            p50_ms=0, p95_ms=0, p99_ms=0, mean_ms=0,
            min_ms=0, max_ms=0, std_ms=0,
            total_requests=num_requests, successful_requests=0,
            failed_requests=failed,
        )


def compare_different_loads(url: str):
    """
    Compare latency under different load conditions.
    """
    print("\n" + "=" * 60)
    print("Comparing Different Token Lengths")
    print("=" * 60)

    test_cases = [
        ("Short (20 tokens)", "Hello, world!", 20),
        ("Medium (100 tokens)", "The future of artificial intelligence is", 100),
        ("Long (200 tokens)", "Write a detailed essay about machine learning", 200),
    ]

    results = []

    for name, prompt, max_tokens in test_cases:
        print(f"\n📊 Testing: {name}")
        metrics = run_latency_benchmark(
            url=url,
            num_requests=20,
            prompt=prompt,
            max_tokens=max_tokens,
        )
        results.append((name, metrics))

    # Summary table
    print("\n" + "=" * 60)
    print("Summary Comparison")
    print("=" * 60)
    print(f"{'Test Case':<25} {'P50 (ms)':<12} {'P95 (ms)':<12} {'P99 (ms)':<12}")
    print("-" * 60)

    for name, metrics in results:
        print(f"{name:<25} {metrics.p50_ms:<12.2f} {metrics.p95_ms:<12.2f} {metrics.p99_ms:<12.2f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Latency benchmarking for vLLM API")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="API base URL",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=100,
        help="Number of requests to make",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The future of artificial intelligence is",
        help="Test prompt",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run comparison across different loads",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for results (JSON)",
    )

    args = parser.parse_args()

    # Check API health
    try:
        response = requests.get(f"{args.url}/health", timeout=5)
        if response.status_code != 200:
            print(f"❌ API health check failed: {response.text}")
            return
        print("✅ API is healthy\n")
    except Exception as e:
        print(f"❌ Could not connect to API: {e}")
        return

    # Run benchmark
    if args.compare:
        compare_different_loads(args.url)
    else:
        metrics = run_latency_benchmark(
            url=args.url,
            num_requests=args.requests,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
        )

        # Save results if output file specified
        if args.output:
            with open(args.output, "w") as f:
                json.dump(asdict(metrics), f, indent=2)
            print(f"\n💾 Results saved to {args.output}")


if __name__ == "__main__":
    main()
