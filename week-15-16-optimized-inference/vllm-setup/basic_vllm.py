"""
Basic vLLM Setup and Usage
===========================

This script demonstrates the fundamental usage of vLLM for efficient LLM inference.
vLLM uses PagedAttention to manage memory efficiently and achieve high throughput.

Installation:
    pip install vllm

Usage:
    python basic_vllm.py
"""

from vllm import LLM, SamplingParams
import time


def basic_inference_example():
    """
    Demonstrates basic inference with vLLM using a small model.
    """
    print("=" * 60)
    print("Basic vLLM Inference Example")
    print("=" * 60)

    # Initialize the LLM
    # Using a small model for quick testing (125M parameters)
    print("\n1. Loading model...")
    start_time = time.time()

    llm = LLM(
        model="facebook/opt-125m",
        dtype="float16",  # Use FP16 for faster inference
        max_model_len=512,  # Maximum sequence length
    )

    load_time = time.time() - start_time
    print(f"   Model loaded in {load_time:.2f}s")

    # Define sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,      # Controls randomness (0=deterministic, 1=creative)
        top_p=0.95,          # Nucleus sampling threshold
        max_tokens=100,      # Maximum tokens to generate
        repetition_penalty=1.1,  # Penalize repetition
    )

    # Example prompts
    prompts = [
        "The future of artificial intelligence is",
        "Once upon a time in a distant galaxy",
        "The three laws of robotics are",
    ]

    # Generate completions
    print("\n2. Generating completions...")
    start_time = time.time()

    outputs = llm.generate(prompts, sampling_params)

    generation_time = time.time() - start_time
    print(f"   Generated {len(prompts)} completions in {generation_time:.2f}s")
    print(f"   Average: {generation_time/len(prompts):.3f}s per prompt")

    # Display results
    print("\n3. Results:")
    print("-" * 60)
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        tokens_generated = len(output.outputs[0].token_ids)

        print(f"\n[Prompt {i+1}]")
        print(f"Input: {prompt}")
        print(f"Output: {generated_text}")
        print(f"Tokens: {tokens_generated}")


def batched_inference_example():
    """
    Demonstrates efficient batched inference with vLLM.
    vLLM automatically optimizes batch processing.
    """
    print("\n" + "=" * 60)
    print("Batched Inference Example")
    print("=" * 60)

    llm = LLM(model="facebook/opt-125m", dtype="float16")

    # Large batch of prompts
    prompts = [
        f"Question {i}: What is the meaning of life?"
        for i in range(1, 11)
    ]

    sampling_params = SamplingParams(
        temperature=0.8,
        max_tokens=50,
    )

    print(f"\nProcessing {len(prompts)} prompts in batch...")
    start_time = time.time()

    outputs = llm.generate(prompts, sampling_params)

    batch_time = time.time() - start_time

    print(f"Batch processed in {batch_time:.2f}s")
    print(f"Average latency: {batch_time/len(prompts):.3f}s per prompt")
    print(f"Throughput: {len(prompts)/batch_time:.2f} prompts/second")

    # Show first 3 results
    print("\nFirst 3 results:")
    for output in outputs[:3]:
        print(f"\nPrompt: {output.prompt}")
        print(f"Response: {output.outputs[0].text[:100]}...")


def compare_sampling_strategies():
    """
    Compares different sampling strategies (greedy, sampling, beam search).
    """
    print("\n" + "=" * 60)
    print("Comparing Sampling Strategies")
    print("=" * 60)

    llm = LLM(model="facebook/opt-125m", dtype="float16")

    prompt = "The key to happiness is"

    # Strategy 1: Greedy (temperature=0)
    greedy_params = SamplingParams(temperature=0, max_tokens=30)

    # Strategy 2: Sampling (temperature=0.7)
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=30)

    # Strategy 3: High temperature (more random)
    random_params = SamplingParams(temperature=1.5, max_tokens=30)

    print(f"\nPrompt: {prompt}\n")

    # Greedy decoding
    output = llm.generate([prompt], greedy_params)[0]
    print(f"Greedy (temp=0.0):\n  {output.outputs[0].text}\n")

    # Standard sampling
    output = llm.generate([prompt], sampling_params)[0]
    print(f"Sampling (temp=0.7):\n  {output.outputs[0].text}\n")

    # High temperature
    output = llm.generate([prompt], random_params)[0]
    print(f"Random (temp=1.5):\n  {output.outputs[0].text}\n")


if __name__ == "__main__":
    # Run all examples
    basic_inference_example()
    batched_inference_example()
    compare_sampling_strategies()

    print("\n" + "=" * 60)
    print("✅ All examples completed successfully!")
    print("=" * 60)
