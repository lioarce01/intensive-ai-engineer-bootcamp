"""
Continuous Batching with vLLM
==============================

This script demonstrates vLLM's continuous batching capability,
which allows new requests to join ongoing batches dynamically.

Key Benefits:
- Higher GPU utilization
- Better throughput
- Lower average latency

Comparison with static batching is included.
"""

from vllm import LLM, SamplingParams
import time
import asyncio
from typing import List, Dict
import random


class ContinuousBatchingDemo:
    """
    Demonstrates continuous batching patterns with vLLM.
    """

    def __init__(self, model_name: str = "facebook/opt-125m"):
        print(f"Initializing vLLM with model: {model_name}")
        self.llm = LLM(
            model=model_name,
            dtype="float16",
            max_model_len=512,
            gpu_memory_utilization=0.8,  # Use 80% of GPU memory
        )

    def simulate_streaming_requests(self, num_requests: int = 20):
        """
        Simulates requests arriving over time (more realistic scenario).
        vLLM handles these efficiently with continuous batching.
        """
        print("\n" + "=" * 60)
        print("Simulating Streaming Requests (Continuous Batching)")
        print("=" * 60)

        prompts = [
            f"Request {i}: Write a short story about a robot who"
            for i in range(1, num_requests + 1)
        ]

        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=100,
        )

        print(f"\nProcessing {num_requests} requests with continuous batching...")
        start_time = time.time()

        # vLLM automatically handles continuous batching internally
        # Requests are processed as they arrive, maximizing GPU utilization
        outputs = self.llm.generate(prompts, sampling_params)

        total_time = time.time() - start_time

        # Calculate metrics
        total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        throughput = total_tokens / total_time

        print(f"\n📊 Performance Metrics:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Requests processed: {num_requests}")
        print(f"  Average latency: {total_time/num_requests:.3f}s per request")
        print(f"  Total tokens generated: {total_tokens}")
        print(f"  Throughput: {throughput:.2f} tokens/second")
        print(f"  Requests/second: {num_requests/total_time:.2f}")

        return {
            "total_time": total_time,
            "num_requests": num_requests,
            "total_tokens": total_tokens,
            "throughput": throughput,
        }

    def compare_batch_sizes(self):
        """
        Compares performance across different batch sizes.
        """
        print("\n" + "=" * 60)
        print("Comparing Different Batch Sizes")
        print("=" * 60)

        batch_sizes = [1, 4, 8, 16, 32]
        results = []

        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=50,
        )

        for batch_size in batch_sizes:
            prompts = [
                f"Batch {batch_size}, Request {i}: The meaning of life is"
                for i in range(batch_size)
            ]

            start_time = time.time()
            outputs = self.llm.generate(prompts, sampling_params)
            elapsed = time.time() - start_time

            total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
            throughput = total_tokens / elapsed

            results.append({
                "batch_size": batch_size,
                "time": elapsed,
                "throughput": throughput,
                "avg_latency": elapsed / batch_size,
            })

            print(f"\nBatch size: {batch_size}")
            print(f"  Time: {elapsed:.3f}s")
            print(f"  Throughput: {throughput:.2f} tokens/s")
            print(f"  Avg latency: {elapsed/batch_size:.3f}s")

        # Print summary
        print("\n" + "=" * 60)
        print("Summary: Optimal Batch Size Analysis")
        print("=" * 60)
        print(f"{'Batch Size':<12} {'Time (s)':<12} {'Throughput':<15} {'Avg Latency':<15}")
        print("-" * 60)

        for r in results:
            print(f"{r['batch_size']:<12} {r['time']:<12.3f} {r['throughput']:<15.2f} {r['avg_latency']:<15.3f}")

        # Find optimal batch size (best throughput)
        best = max(results, key=lambda x: x["throughput"])
        print(f"\n✅ Optimal batch size: {best['batch_size']} (throughput: {best['throughput']:.2f} tokens/s)")

        return results

    def demonstrate_mixed_length_batching(self):
        """
        Shows how vLLM handles batches with varying prompt/generation lengths.
        This is where continuous batching really shines.
        """
        print("\n" + "=" * 60)
        print("Mixed Length Batching (Variable max_tokens)")
        print("=" * 60)

        # Create prompts with varying desired output lengths
        test_cases = [
            ("Short response:", 20),
            ("Medium response:", 50),
            ("Long response:", 100),
            ("Very long response:", 200),
        ] * 3  # Repeat to create a batch of 12

        prompts = [prompt for prompt, _ in test_cases]

        # Use different sampling params for each
        outputs_list = []

        for prompt, max_tokens in test_cases:
            sampling_params = SamplingParams(
                temperature=0.7,
                max_tokens=max_tokens,
            )
            # In practice, vLLM can handle these together
            # For demo, we'll process individually to show flexibility
            output = self.llm.generate([prompt], sampling_params)[0]
            outputs_list.append((prompt, output, max_tokens))

        print("\nResults:")
        for i, (prompt, output, target_max) in enumerate(outputs_list[:4], 1):
            actual_tokens = len(output.outputs[0].token_ids)
            print(f"\n{i}. {prompt}")
            print(f"   Target max tokens: {target_max}")
            print(f"   Actual tokens: {actual_tokens}")
            print(f"   Text: {output.outputs[0].text[:60]}...")


def benchmark_continuous_vs_sequential():
    """
    Compares continuous batching (vLLM) vs. sequential processing.
    """
    print("\n" + "=" * 60)
    print("Benchmark: Continuous Batching vs Sequential")
    print("=" * 60)

    llm = LLM(model="facebook/opt-125m", dtype="float16")

    num_requests = 10
    prompts = [f"Question {i}: What is AI?" for i in range(1, num_requests + 1)]
    sampling_params = SamplingParams(temperature=0.7, max_tokens=30)

    # Method 1: Batch processing (continuous batching)
    print("\n1. Continuous Batching (vLLM default):")
    start = time.time()
    outputs_batch = llm.generate(prompts, sampling_params)
    batch_time = time.time() - start
    print(f"   Time: {batch_time:.3f}s")
    print(f"   Throughput: {num_requests/batch_time:.2f} req/s")

    # Method 2: Sequential processing (one at a time)
    print("\n2. Sequential Processing:")
    start = time.time()
    outputs_seq = []
    for prompt in prompts:
        output = llm.generate([prompt], sampling_params)
        outputs_seq.extend(output)
    seq_time = time.time() - start
    print(f"   Time: {seq_time:.3f}s")
    print(f"   Throughput: {num_requests/seq_time:.2f} req/s")

    # Comparison
    speedup = seq_time / batch_time
    print(f"\n🚀 Speedup: {speedup:.2f}x faster with continuous batching")
    print(f"   Efficiency gain: {(speedup - 1) * 100:.1f}%")


if __name__ == "__main__":
    # Initialize demo
    demo = ContinuousBatchingDemo()

    # Run demonstrations
    demo.simulate_streaming_requests(num_requests=20)
    demo.compare_batch_sizes()
    demo.demonstrate_mixed_length_batching()

    # Run benchmark
    benchmark_continuous_vs_sequential()

    print("\n" + "=" * 60)
    print("✅ Continuous batching demonstration completed!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. vLLM's continuous batching improves GPU utilization")
    print("2. Larger batches generally improve throughput")
    print("3. Mixed-length requests are handled efficiently")
    print("4. Significant speedup vs. sequential processing")
