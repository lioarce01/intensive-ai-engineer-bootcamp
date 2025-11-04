# Batching Optimization Guide

## Overview

Batching groups multiple inference requests together to maximize GPU utilization and throughput. vLLM's continuous batching takes this further by dynamically managing batches as requests arrive and complete.

## Types of Batching

### 1. Static Batching

**Description:** Fixed batch size, waits for batch to fill before processing.

**Pros:**
- Simple to implement
- Predictable latency per batch

**Cons:**
- Wastes GPU cycles waiting for batch to fill
- High latency for early requests in batch
- Poor GPU utilization with variable request rates

```python
# Static batching (NOT recommended)
def static_batch_inference(requests, batch_size=8):
    batches = []
    for i in range(0, len(requests), batch_size):
        batch = requests[i:i + batch_size]
        # Wait for full batch...
        batches.append(batch)

    # Process all batches
    for batch in batches:
        process_batch(batch)
```

### 2. Dynamic/Continuous Batching

**Description:** Requests join and leave batches as they arrive/complete.

**Pros:**
- Maximum GPU utilization
- Lower average latency
- Handles variable request rates gracefully
- No waiting for batch to fill

**Cons:**
- More complex to implement (but vLLM handles this!)

```python
# vLLM handles continuous batching automatically
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")

# These requests are processed with continuous batching
prompts = ["Prompt 1", "Prompt 2", "Prompt 3", ...]
outputs = llm.generate(prompts, sampling_params)
```

## How vLLM's Continuous Batching Works

vLLM implements PagedAttention-based continuous batching:

1. **Requests arrive** → Join active batch immediately
2. **Token generation** → Each request generates tokens independently
3. **Request completes** → Removed from batch, resources freed
4. **New requests** → Fill freed spots in batch
5. **Dynamic sizing** → Batch size adjusts based on load

## Optimizing Batch Performance

### 1. Right-size Your GPU Memory

```python
from vllm import LLM

llm = LLM(
    model="facebook/opt-1.3b",
    gpu_memory_utilization=0.9,  # Use 90% of GPU memory
    # Higher = larger batches possible
)
```

**Guidelines:**
- 0.8-0.9 for production (leaves headroom)
- 0.95 for maximum throughput (risky, may OOM)
- Monitor actual memory usage and adjust

### 2. Choose Optimal max_model_len

```python
llm = LLM(
    model="facebook/opt-1.3b",
    max_model_len=512,  # Shorter = more batch capacity
)
```

**Trade-off:**
- Shorter max_len → Larger batch sizes possible
- Longer max_len → Fewer concurrent requests

**Recommendation:**
- Set to your P95 input + output length
- Don't set to model maximum if you don't need it

### 3. Batch Size vs. Latency

There's a sweet spot between throughput and latency:

```
Small batches (1-4):
  ✅ Low latency
  ❌ Low GPU utilization
  ❌ Low throughput

Medium batches (8-16):
  ✅ Good latency
  ✅ Good GPU utilization
  ✅ Good throughput

Large batches (32+):
  ❌ Higher latency
  ✅ Maximum GPU utilization
  ✅ Maximum throughput
```

## Benchmarking Batch Performance

### Simple Throughput Test

```python
from vllm import LLM, SamplingParams
import time

llm = LLM(model="facebook/opt-125m")
sampling_params = SamplingParams(temperature=0.7, max_tokens=50)

batch_sizes = [1, 4, 8, 16, 32]
results = []

for batch_size in batch_sizes:
    prompts = [f"Request {i}" for i in range(batch_size)]

    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = total_tokens / elapsed

    results.append({
        "batch_size": batch_size,
        "time": elapsed,
        "throughput": throughput,
        "latency_per_request": elapsed / batch_size,
    })

    print(f"Batch {batch_size}: {throughput:.1f} tokens/s, "
          f"{elapsed/batch_size:.3f}s per request")

# Find optimal batch size
best = max(results, key=lambda x: x["throughput"])
print(f"\nOptimal batch size: {best['batch_size']}")
```

### Continuous Load Simulation

```python
import asyncio
import random

async def simulate_continuous_load(
    llm,
    num_requests: int = 100,
    arrival_rate: float = 10.0,  # requests per second
):
    """
    Simulates requests arriving continuously.
    """
    sampling_params = SamplingParams(temperature=0.7, max_tokens=50)

    async def process_request(i):
        prompt = f"Request {i}: Tell me about AI"
        start = time.time()
        # In real async setup, this would be non-blocking
        output = llm.generate([prompt], sampling_params)
        latency = time.time() - start
        return latency

    tasks = []
    for i in range(num_requests):
        tasks.append(process_request(i))
        # Simulate inter-arrival time
        await asyncio.sleep(1.0 / arrival_rate + random.uniform(-0.1, 0.1))

    latencies = await asyncio.gather(*tasks)

    print(f"Average latency: {sum(latencies)/len(latencies):.3f}s")
    print(f"P95 latency: {sorted(latencies)[int(len(latencies)*0.95)]:.3f}s")
```

## Production Patterns

### Pattern 1: Request Queuing

For asynchronous APIs, use a request queue:

```python
import queue
import threading

request_queue = queue.Queue()

def inference_worker(llm):
    while True:
        # Collect batch from queue
        batch = []
        for _ in range(MAX_BATCH_SIZE):
            try:
                req = request_queue.get(timeout=0.01)
                batch.append(req)
            except queue.Empty:
                break

        if batch:
            # Process batch
            prompts = [req["prompt"] for req in batch]
            outputs = llm.generate(prompts, sampling_params)

            # Return results
            for req, output in zip(batch, outputs):
                req["callback"](output)

# Start worker thread
threading.Thread(target=inference_worker, args=(llm,)).start()
```

### Pattern 2: Priority Batching

Some requests may need lower latency:

```python
import heapq

class PriorityBatchScheduler:
    def __init__(self, llm):
        self.llm = llm
        self.high_priority = []
        self.low_priority = []

    def add_request(self, prompt, priority="normal"):
        if priority == "high":
            heapq.heappush(self.high_priority, prompt)
        else:
            self.low_priority.append(prompt)

    def process_batch(self, max_batch_size=16):
        batch = []

        # Fill with high priority first
        while self.high_priority and len(batch) < max_batch_size:
            batch.append(heapq.heappop(self.high_priority))

        # Fill remaining with low priority
        while self.low_priority and len(batch) < max_batch_size:
            batch.append(self.low_priority.pop(0))

        if batch:
            return self.llm.generate(batch, sampling_params)
```

## Monitoring Batch Performance

### Key Metrics

1. **Batch Size Distribution**
   - Track actual batch sizes being processed
   - Identify if you're batching effectively

2. **Queue Depth**
   - Number of requests waiting
   - High queue depth → may need more GPU capacity

3. **GPU Utilization**
   - Should be > 80% for good batching
   - Low utilization → batch sizes too small

4. **Latency by Batch Size**
   - Understand latency impact of batching
   - Helps set SLAs

### Prometheus Metrics Example

```python
from prometheus_client import Histogram, Gauge

batch_size_histogram = Histogram(
    "vllm_batch_size",
    "Distribution of batch sizes",
    buckets=[1, 2, 4, 8, 16, 32, 64],
)

queue_depth_gauge = Gauge(
    "vllm_queue_depth",
    "Number of requests in queue",
)

# Record metrics
batch_size_histogram.observe(current_batch_size)
queue_depth_gauge.set(len(request_queue))
```

## Common Issues and Solutions

### Issue 1: High P99 Latency

**Symptom:** P50 is fine, but P99 is very high.

**Cause:** Requests getting stuck in large batches.

**Solution:**
- Implement max latency timeout
- Use priority queuing for latency-sensitive requests
- Scale horizontally (more GPU instances)

### Issue 2: Low GPU Utilization

**Symptom:** GPU utilization < 70%.

**Cause:** Not enough concurrent requests to batch.

**Solution:**
- Increase `max_model_len` (but watch memory)
- Lower `gpu_memory_utilization` slightly
- Check if you're CPU-bound on preprocessing

### Issue 3: OOM (Out of Memory)

**Symptom:** CUDA out of memory errors.

**Cause:** Batch size too large for GPU memory.

**Solution:**
- Reduce `gpu_memory_utilization` from 0.9 to 0.8
- Use quantization (INT8, GPTQ, AWQ)
- Reduce `max_model_len`

## Best Practices

1. **Monitor actual batch sizes** in production
2. **Set realistic max_model_len** based on your data
3. **Use continuous batching** (vLLM does this automatically)
4. **Test under load** before production
5. **Have fallback strategies** for high load (queuing, rate limiting)
6. **Profile GPU utilization** to ensure you're batching effectively

## Resources

- [vLLM PagedAttention Blog](https://blog.vllm.ai/2023/06/20/vllm.html)
- [Batching Examples](../vllm-setup/batch_inference.py)
- [Production API](../api-service/main.py)

## Next Steps

1. Run the batching benchmarks: `python vllm-setup/batch_inference.py`
2. Experiment with different `gpu_memory_utilization` values
3. Load test your API with various request rates
4. Monitor batch size distribution in production
