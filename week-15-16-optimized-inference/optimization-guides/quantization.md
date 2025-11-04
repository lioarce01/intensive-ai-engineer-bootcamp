# Quantization Optimization Guide

## Overview

Quantization reduces model precision from FP32/FP16 to INT8/INT4, dramatically reducing memory usage while maintaining acceptable quality.

## Quick Reference

| Method | Bit-width | Memory Reduction | Quality | Speed |
|--------|-----------|------------------|---------|-------|
| FP16 | 16-bit | Baseline | 100% | Baseline |
| INT8 | 8-bit | 2x | 98-99% | Similar/Faster |
| GPTQ | 4-bit | 4x | 95-98% | Similar |
| AWQ | 4-bit | 4x | 98-99% | Faster |

## When to Use Quantization

✅ **Use quantization when:**
- Model doesn't fit in GPU memory
- Deploying to resource-constrained environments
- Need to serve larger batch sizes
- Cost optimization (smaller GPU requirements)

❌ **Don't use quantization when:**
- Model already fits comfortably in memory
- Quality is absolutely critical (e.g., medical, legal)
- Model is very small (< 1B parameters) - minimal gains

## Quantization Methods

### 1. INT8 Quantization

**Description:** Reduces precision to 8 bits while preserving most quality.

**Use case:** General-purpose optimization with minimal quality loss.

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    quantization="int8",
    dtype="float16",
)
```

**Pros:**
- Minimal quality degradation (< 2%)
- 2x memory reduction
- Often faster inference

**Cons:**
- Less memory savings than 4-bit methods
- Requires quantization-aware model or calibration

### 2. GPTQ (4-bit)

**Description:** Post-training quantization optimized for generative models.

**Use case:** Maximum memory savings with good quality.

```python
from vllm import LLM, SamplingParams

# Using pre-quantized model from TheBloke
llm = LLM(
    model="TheBloke/Llama-2-7B-GPTQ",
    quantization="gptq",
    dtype="float16",
    gpu_memory_utilization=0.9,
)
```

**Pros:**
- 4x memory reduction
- Works well for 7B+ parameter models
- Many pre-quantized models available

**Cons:**
- 2-5% quality degradation
- Requires pre-quantized weights
- Slightly slower on very small batches

**Where to find GPTQ models:**
- [TheBloke on HuggingFace](https://huggingface.co/TheBloke)
- Search for `-GPTQ` suffix

### 3. AWQ (Activation-aware Weight Quantization)

**Description:** Preserves important weights based on activation patterns.

**Use case:** Best quality-to-memory ratio for 4-bit quantization.

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="TheBloke/Llama-2-7B-AWQ",
    quantization="awq",
    dtype="float16",
)
```

**Pros:**
- Better quality than GPTQ (98-99% of FP16)
- 4x memory reduction
- Often faster inference than GPTQ

**Cons:**
- Fewer pre-quantized models available
- Requires AWQ-specific quantized weights

## Practical Implementation

### Step 1: Benchmark Your Baseline

Always measure your FP16 baseline first:

```python
from vllm import LLM, SamplingParams
import time

# Load FP16 model
llm_fp16 = LLM(model="facebook/opt-1.3b", dtype="float16")
sampling_params = SamplingParams(temperature=0.7, max_tokens=100)

# Benchmark
prompts = ["Test prompt"] * 10
start = time.time()
outputs = llm_fp16.generate(prompts, sampling_params)
baseline_time = time.time() - start

print(f"FP16 Baseline: {baseline_time:.2f}s")
```

### Step 2: Test Quantized Version

```python
# Load quantized model (INT8 example)
llm_int8 = LLM(
    model="facebook/opt-1.3b",
    quantization="int8",
    dtype="float16",
)

# Benchmark
start = time.time()
outputs = llm_int8.generate(prompts, sampling_params)
quantized_time = time.time() - start

print(f"INT8: {quantized_time:.2f}s")
print(f"Speedup: {baseline_time/quantized_time:.2f}x")
```

### Step 3: Validate Quality

Compare outputs for the same prompts:

```python
test_prompts = [
    "Explain machine learning in simple terms:",
    "Write a haiku about coding:",
    "What is the capital of France?",
]

# Generate with both models
outputs_fp16 = llm_fp16.generate(test_prompts, sampling_params)
outputs_int8 = llm_int8.generate(test_prompts, sampling_params)

# Manual comparison
for i, prompt in enumerate(test_prompts):
    print(f"\nPrompt: {prompt}")
    print(f"FP16: {outputs_fp16[i].outputs[0].text}")
    print(f"INT8: {outputs_int8[i].outputs[0].text}")
```

For quantitative evaluation, use perplexity or task-specific metrics.

### Step 4: Monitor GPU Memory

```python
import torch

# Check memory usage
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"Allocated: {allocated:.2f}GB")
    print(f"Reserved: {reserved:.2f}GB")
```

## Quantization Workflow

```
1. Baseline (FP16)
   ↓
2. Measure: latency, throughput, memory
   ↓
3. Test INT8
   ↓
4. Validate quality
   ↓ (If quality OK)
5. Deploy
   ↓ (If more memory savings needed)
6. Test GPTQ/AWQ (4-bit)
   ↓
7. Validate quality
   ↓
8. Deploy
```

## Common Pitfalls

### 1. Not Testing Quality

**Problem:** Assume quantization won't affect your task.

**Solution:** Always validate on your specific use case:
- Run evaluation metrics (accuracy, F1, BLEU, etc.)
- Manual inspection of outputs
- A/B testing in production

### 2. Using Quantization for Small Models

**Problem:** Quantizing models < 1B parameters.

**Solution:** Small models already fit in memory. Quantization overhead may slow them down.

### 3. Wrong Model Format

**Problem:** Loading FP16 model with `quantization="gptq"`.

**Solution:** Use models specifically quantized for that method:
- GPTQ models end with `-GPTQ`
- AWQ models end with `-AWQ`
- Check model card for supported quantization

### 4. Ignoring Batch Size

**Problem:** Quantized models may perform differently at various batch sizes.

**Solution:** Test with your production batch size.

## Production Checklist

- [ ] Benchmark FP16 baseline (latency, throughput, memory)
- [ ] Test quantized variant
- [ ] Compare quality on validation set
- [ ] Check memory usage reduction
- [ ] Test with production batch sizes
- [ ] Monitor P95/P99 latencies
- [ ] A/B test in staging environment
- [ ] Document quantization method in model card

## Resources

- [vLLM Quantization Docs](https://docs.vllm.ai/en/latest/quantization/overview.html)
- [GPTQ Paper](https://arxiv.org/abs/2210.17323)
- [AWQ Paper](https://arxiv.org/abs/2306.00978)
- [TheBloke's Quantized Models](https://huggingface.co/TheBloke)

## Example: Full Quantization Pipeline

See `week-15-16-optimized-inference/vllm-setup/quantized_vllm.py` for a complete benchmarking script.

```bash
cd week-15-16-optimized-inference/vllm-setup
python quantized_vllm.py
```
