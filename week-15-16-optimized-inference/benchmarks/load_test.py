"""
Load Testing for vLLM API
==========================

Uses Locust to simulate concurrent users and measure:
- Requests per second
- Response times (P50, P95, P99)
- Error rates
- System stability under load

Installation:
    pip install locust requests

Usage:
    locust -f load_test.py --host http://localhost:8000

Then open http://localhost:8089 to start the test.
"""

from locust import HttpUser, task, between, events
import random
import time
import json
from typing import List


# ============================================================================
# Test Prompts
# ============================================================================

PROMPTS = [
    "The future of artificial intelligence is",
    "Once upon a time in a distant galaxy",
    "The three laws of robotics are",
    "Machine learning can be defined as",
    "The key to happiness is",
    "Climate change is caused by",
    "The best programming language is",
    "Quantum computing will revolutionize",
    "The meaning of life is",
    "Space exploration should focus on",
]

SHORT_PROMPTS = [
    "Hello",
    "What is AI?",
    "Tell me a joke",
    "Explain Python",
    "Define recursion",
]

LONG_PROMPTS = [
    "Write a detailed essay about the impact of artificial intelligence on society, covering economic, social, and ethical implications",
    "Explain the complete history of computer science from Charles Babbage to modern quantum computing",
    "Describe in detail how neural networks work, including backpropagation, gradient descent, and activation functions",
]


# ============================================================================
# Locust User Classes
# ============================================================================

class VLLMAPIUser(HttpUser):
    """
    Simulates a user making requests to the vLLM API.
    """
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests

    @task(3)  # Higher weight = more frequent
    def generate_standard(self):
        """
        Standard text generation request.
        """
        prompt = random.choice(PROMPTS)
        payload = {
            "prompt": prompt,
            "max_tokens": 100,
            "temperature": 0.7,
            "use_cache": True,
        }

        with self.client.post(
            "/generate",
            json=payload,
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                # Validate response
                if "generated_text" in data and "inference_time_ms" in data:
                    response.success()
                else:
                    response.failure("Invalid response format")
            else:
                response.failure(f"Got status code {response.status_code}")

    @task(1)
    def generate_short(self):
        """
        Short text generation (low latency test).
        """
        prompt = random.choice(SHORT_PROMPTS)
        payload = {
            "prompt": prompt,
            "max_tokens": 20,
            "temperature": 0.5,
            "use_cache": True,
        }

        with self.client.post(
            "/generate",
            json=payload,
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")

    @task(1)
    def generate_long(self):
        """
        Long text generation (throughput test).
        """
        prompt = random.choice(LONG_PROMPTS)
        payload = {
            "prompt": prompt,
            "max_tokens": 200,
            "temperature": 0.8,
            "use_cache": False,  # No cache for long generations
        }

        with self.client.post(
            "/generate",
            json=payload,
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")

    @task(1)
    def health_check(self):
        """
        Health check endpoint (should be very fast).
        """
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed")


class BatchUser(HttpUser):
    """
    User that makes batch requests.
    """
    wait_time = between(2, 5)

    @task
    def batch_generate(self):
        """
        Batch generation request.
        """
        batch_size = random.randint(2, 8)
        prompts = random.sample(PROMPTS, min(batch_size, len(PROMPTS)))

        payload = {
            "prompts": prompts,
            "max_tokens": 50,
            "temperature": 0.7,
        }

        with self.client.post(
            "/batch-generate",
            json=payload,
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "results" in data and len(data["results"]) == len(prompts):
                    response.success()
                else:
                    response.failure("Batch response incomplete")
            else:
                response.failure(f"Got status code {response.status_code}")


class CacheTestUser(HttpUser):
    """
    User that tests caching performance.
    """
    wait_time = between(0.5, 1.5)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Use same prompts repeatedly to test cache
        self.cached_prompts = random.sample(PROMPTS, 3)

    @task
    def cached_request(self):
        """
        Make cached requests (should hit cache frequently).
        """
        prompt = random.choice(self.cached_prompts)
        payload = {
            "prompt": prompt,
            "max_tokens": 50,
            "temperature": 0.7,
            "use_cache": True,
        }

        with self.client.post(
            "/generate",
            json=payload,
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                # Check if cached
                is_cached = data.get("cached", False)
                if is_cached:
                    # Cached requests should be very fast
                    if response.elapsed.total_seconds() < 0.1:
                        response.success()
                    else:
                        response.failure("Cached request too slow")
                else:
                    response.success()
            else:
                response.failure(f"Got status code {response.status_code}")


# ============================================================================
# Event Handlers
# ============================================================================

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when test starts."""
    print("=" * 60)
    print("🚀 Load Test Starting")
    print("=" * 60)
    print(f"Host: {environment.host}")
    print(f"Users: {environment.runner.target_user_count if hasattr(environment.runner, 'target_user_count') else 'N/A'}")
    print("=" * 60)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when test stops."""
    print("\n" + "=" * 60)
    print("🏁 Load Test Completed")
    print("=" * 60)

    stats = environment.stats
    print(f"\nTotal Requests: {stats.total.num_requests}")
    print(f"Total Failures: {stats.total.num_failures}")
    print(f"Average Response Time: {stats.total.avg_response_time:.2f}ms")
    print(f"Min Response Time: {stats.total.min_response_time:.2f}ms")
    print(f"Max Response Time: {stats.total.max_response_time:.2f}ms")
    print(f"Requests/sec: {stats.total.total_rps:.2f}")

    # Check if we met our performance targets
    print("\n" + "=" * 60)
    print("Performance Targets")
    print("=" * 60)

    p95 = stats.total.get_response_time_percentile(0.95)
    p99 = stats.total.get_response_time_percentile(0.99)

    targets = [
        ("P50 < 50ms", stats.total.median_response_time < 50, stats.total.median_response_time),
        ("P95 < 100ms", p95 < 100, p95),
        ("P99 < 200ms", p99 < 200, p99),
        ("Error rate < 1%", (stats.total.num_failures / max(stats.total.num_requests, 1)) < 0.01, (stats.total.num_failures / max(stats.total.num_requests, 1)) * 100),
    ]

    for target, met, actual in targets:
        status = "✅" if met else "❌"
        print(f"{status} {target} (actual: {actual:.2f})")

    print("=" * 60)


# ============================================================================
# Custom Test Scenarios
# ============================================================================

def create_test_plan():
    """
    Returns a test plan for different scenarios.
    """
    return """
    Load Testing Scenarios:

    1. Smoke Test (1 user, 1 min)
       locust -f load_test.py --host http://localhost:8000 --users 1 --spawn-rate 1 --run-time 1m --headless

    2. Load Test (10 users, 5 min)
       locust -f load_test.py --host http://localhost:8000 --users 10 --spawn-rate 2 --run-time 5m --headless

    3. Stress Test (50 users, 10 min)
       locust -f load_test.py --host http://localhost:8000 --users 50 --spawn-rate 5 --run-time 10m --headless

    4. Spike Test (100 users, rapid spawn)
       locust -f load_test.py --host http://localhost:8000 --users 100 --spawn-rate 10 --run-time 5m --headless

    5. Endurance Test (20 users, 1 hour)
       locust -f load_test.py --host http://localhost:8000 --users 20 --spawn-rate 2 --run-time 1h --headless
    """


if __name__ == "__main__":
    print(create_test_plan())
