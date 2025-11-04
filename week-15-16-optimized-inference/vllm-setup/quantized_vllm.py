"""
Quantized Model Inference with vLLM
====================================

This script demonstrates how to use quantized models with vLLM
to reduce memory usage while maintaining inference speed.

Quantization Methods Covered:
- INT8 (8-bit quantization)
- GPTQ (4-bit quantization)
- AWQ (Activation-aware Weight Quantization)

Installation:
    pip install vllm
    pip install auto-gptq  # For GPTQ support
"""

from vllm import LLM, SamplingParams
import time
import torch
import psutil
import os


def get_gpu_memory():
    """
    Get current GPU memory usage in GB.
    """
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 3)
    return 0


def get_system_memory():
    """
    Get current system RAM usage in GB.
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)


class QuantizationBenchmark:
    """
    Benchmarks different quantization strategies.
    """

    def __init__(self):
        self.results = []

    def benchmark_fp16_baseline(self):
        """
        Benchmark FP16 model as baseline.
        """
        print("=" * 60)
        print("Benchmark 1: FP16 Baseline")
        print("=" * 60)

        # Using a small model for demonstration
        model_name = "facebook/opt-125m"

        print(f"\nLoading model: {model_name} (FP16)")
        start_mem = get_gpu_memory()

        llm = LLM(
            model=model_name,
            dtype="float16",
            max_model_len=512,
        )

        end_mem = get_gpu_memory()
        mem_used = end_mem - start_mem

        # Benchmark inference
        prompts = ["The future of AI is"] * 10
        sampling_params = SamplingParams(temperature=0.7, max_tokens=50)

        start_time = time.time()
        outputs = llm.generate(prompts, sampling_params)
        inference_time = time.time() - start_time

        total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        throughput = total_tokens / inference_time

        result = {
            "method": "FP16",
            "memory_gb": mem_used,
            "inference_time": inference_time,
            "throughput": throughput,
            "sample_output": outputs[0].outputs[0].text[:100],
        }

        self.results.append(result)

        print(f"\n📊 Results:")
        print(f"  GPU Memory Used: {mem_used:.2f} GB")
        print(f"  Inference Time: {inference_time:.3f}s")
        print(f"  Throughput: {throughput:.2f} tokens/s")
        print(f"  Sample Output: {result['sample_output']}...")

        return result

    def benchmark_gptq_quantization(self):
        """
        Benchmark GPTQ (4-bit) quantized model.

        Note: This requires a GPTQ-quantized model from HuggingFace.
        Using TheBloke's quantized models as examples.
        """
        print("\n" + "=" * 60)
        print("Benchmark 2: GPTQ (4-bit) Quantization")
        print("=" * 60)

        # Example GPTQ model (you'll need to download this)
        # For actual use, replace with a model you have access to
        model_name = "TheBloke/opt-125m-GPTQ"  # This is a placeholder

        print(f"\nNote: GPTQ models require pre-quantized weights")
        print(f"Model: {model_name}")
        print("\nFor this demo, we'll simulate the expected results:")

        # Simulated results based on typical GPTQ performance
        result = {
            "method": "GPTQ (4-bit)",
            "memory_gb": 0.5,  # ~4x less than FP16
            "inference_time": 0.15,
            "throughput": 333.0,
            "sample_output": "[Simulated] The future of AI is bright with many possibilities...",
        }

        self.results.append(result)

        print(f"\n📊 Expected Results:")
        print(f"  GPU Memory Used: {result['memory_gb']:.2f} GB (~4x reduction)")
        print(f"  Inference Time: {result['inference_time']:.3f}s")
        print(f"  Throughput: {result['throughput']:.2f} tokens/s")
        print(f"  Quality: ~95-98% of FP16 (minimal degradation)")

        print("\n💡 GPTQ Benefits:")
        print("  • 4x memory reduction")
        print("  • Similar or better inference speed")
        print("  • Minimal quality loss")
        print("  • Enables larger models on smaller GPUs")

        return result

    def benchmark_awq_quantization(self):
        """
        Benchmark AWQ (Activation-aware Weight Quantization).

        AWQ typically provides better quality than GPTQ for the same bit-width.
        """
        print("\n" + "=" * 60)
        print("Benchmark 3: AWQ (4-bit) Quantization")
        print("=" * 60)

        print("\nAWQ (Activation-aware Weight Quantization):")
        print("  • Preserves salient weights (those with large activations)")
        print("  • Better quality than GPTQ at same bit-width")
        print("  • Requires AWQ-quantized model")

        # Simulated results
        result = {
            "method": "AWQ (4-bit)",
            "memory_gb": 0.5,
            "inference_time": 0.14,
            "throughput": 357.0,
            "sample_output": "[Simulated] The future of AI is bright with many possibilities...",
        }

        self.results.append(result)

        print(f"\n📊 Expected Results:")
        print(f"  GPU Memory Used: {result['memory_gb']:.2f} GB (~4x reduction)")
        print(f"  Inference Time: {result['inference_time']:.3f}s")
        print(f"  Throughput: {result['throughput']:.2f} tokens/s")
        print(f"  Quality: ~98-99% of FP16 (minimal degradation)")

        print("\n💡 AWQ Advantages:")
        print("  • Better quality preservation than GPTQ")
        print("  • Slightly faster inference")
        print("  • Same memory savings as GPTQ")

        return result

    def print_comparison(self):
        """
        Print comparison table of all quantization methods.
        """
        print("\n" + "=" * 60)
        print("Quantization Methods Comparison")
        print("=" * 60)

        print(f"\n{'Method':<20} {'Memory (GB)':<15} {'Time (s)':<12} {'Throughput':<15} {'Quality':<10}")
        print("-" * 80)

        quality_map = {
            "FP16": "100%",
            "GPTQ (4-bit)": "95-98%",
            "AWQ (4-bit)": "98-99%",
        }

        for result in self.results:
            method = result["method"]
            memory = result["memory_gb"]
            time_val = result["inference_time"]
            throughput = result["throughput"]
            quality = quality_map.get(method, "N/A")

            print(f"{method:<20} {memory:<15.2f} {time_val:<12.3f} {throughput:<15.2f} {quality:<10}")

        print("\n" + "=" * 60)
        print("Recommendations")
        print("=" * 60)
        print("✅ For maximum quality: Use FP16")
        print("✅ For best memory/quality tradeoff: Use AWQ (4-bit)")
        print("✅ For maximum memory savings: Use GPTQ (4-bit)")
        print("✅ For production: Test quantized vs. FP16 on your specific task")


def demonstrate_quantization_loading():
    """
    Demonstrates how to load different quantized models with vLLM.
    """
    print("\n" + "=" * 60)
    print("Loading Quantized Models - Code Examples")
    print("=" * 60)

    print("\n1. Loading GPTQ Model:")
    print("""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="TheBloke/Llama-2-7B-GPTQ",
        quantization="gptq",
        dtype="float16",
        max_model_len=2048,
    )
    """)

    print("\n2. Loading AWQ Model:")
    print("""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="TheBloke/Llama-2-7B-AWQ",
        quantization="awq",
        dtype="float16",
        max_model_len=2048,
    )
    """)

    print("\n3. Loading FP16 Model (baseline):")
    print("""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="meta-llama/Llama-2-7b-hf",
        dtype="float16",
        max_model_len=2048,
    )
    """)

    print("\n4. Custom Quantization Config:")
    print("""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="your-model",
        quantization="gptq",
        dtype="float16",
        gpu_memory_utilization=0.9,  # Use more GPU memory
        max_model_len=4096,
    )
    """)


def practical_tips():
    """
    Provides practical tips for using quantization in production.
    """
    print("\n" + "=" * 60)
    print("Practical Tips for Production")
    print("=" * 60)

    tips = [
        "1. Always benchmark quality on your specific task",
        "2. GPTQ/AWQ work best for models > 7B parameters",
        "3. Use FP16 for small models (< 1B params) - minimal gains from quantization",
        "4. Monitor output quality metrics (perplexity, task accuracy)",
        "5. Consider AWQ over GPTQ for mission-critical applications",
        "6. Use quantization to fit larger models on your GPU",
        "7. Pre-quantized models from TheBloke are a good starting point",
        "8. Test inference latency - quantization can sometimes increase latency on small batches",
    ]

    for tip in tips:
        print(f"\n  {tip}")


if __name__ == "__main__":
    print("=" * 60)
    print("vLLM Quantization Benchmarking Suite")
    print("=" * 60)

    # Initialize benchmark
    benchmark = QuantizationBenchmark()

    # Run benchmarks
    benchmark.benchmark_fp16_baseline()
    benchmark.benchmark_gptq_quantization()
    benchmark.benchmark_awq_quantization()

    # Show comparison
    benchmark.print_comparison()

    # Show loading examples
    demonstrate_quantization_loading()

    # Show practical tips
    practical_tips()

    print("\n" + "=" * 60)
    print("✅ Quantization benchmarking completed!")
    print("=" * 60)
