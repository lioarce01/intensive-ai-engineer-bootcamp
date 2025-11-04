# Week 15-16: Optimized Inference

> **Goal**: Build a scalable generation API with <100ms P95 latency using vLLM, quantization, batching, and caching strategies

## 🎯 Learning Objectives

By the end of this week, you will:

1. **Master vLLM** - Understand PagedAttention, continuous batching, and how to deploy models efficiently
2. **Optimize with Quantization** - Apply 4-bit, 8-bit quantization techniques to reduce memory footprint
3. **Implement Smart Batching** - Use dynamic batching to maximize throughput
4. **Build Caching Strategies** - Implement KV cache optimization and response caching
5. **Deploy Production API** - Create a FastAPI service with <100ms P95 latency

## 📚 Key Concepts

### 1. vLLM (PagedAttention)
- **What**: High-throughput and memory-efficient inference engine
- **Why**: 24x higher throughput than HuggingFace Transformers
- **How**: PagedAttention inspired by OS virtual memory management

### 2. Text Generation Inference (TGI)
- **What**: Production-ready inference server by Hugging Face
- **Why**: Built-in optimizations, tensor parallelism, quantization support
- **How**: Rust-based server with Python client

### 3. Quantization Techniques
- **INT8**: 8-bit quantization (memory ↓ 4x, minimal quality loss)
- **INT4**: 4-bit quantization (memory ↓ 8x, slight quality loss)
- **GPTQ**: Post-training quantization for generative models
- **AWQ**: Activation-aware Weight Quantization

### 4. Batching Strategies
- **Static Batching**: Fixed batch size, wastes GPU cycles
- **Dynamic/Continuous Batching**: Requests join/leave batch dynamically
- **Benefits**: Better GPU utilization, higher throughput

### 5. Caching Mechanisms
- **KV Cache**: Cache key-value pairs during attention computation
- **Response Cache**: Cache frequent query responses
- **Prefix Cache**: Share common prompt prefixes across requests

## 🛠️ Tech Stack

- **vLLM** - Primary inference engine
- **Text Generation Inference** - Alternative inference server
- **FastAPI** - API framework
- **Redis** - Response caching
- **Prometheus** - Metrics and monitoring
- **Docker** - Containerization

## 📂 Project Structure

```
week-15-16-optimized-inference/
├── README.md
├── vllm-setup/
│   ├── basic_vllm.py          # Basic vLLM setup
│   ├── batch_inference.py     # Continuous batching example
│   └── quantized_vllm.py      # Quantized model inference
├── tgi-setup/
│   ├── Dockerfile             # TGI container setup
│   ├── tgi_client.py          # TGI client example
│   └── config.json            # TGI configuration
├── api-service/
│   ├── main.py                # FastAPI application
│   ├── models.py              # Request/response models
│   ├── cache.py               # Caching layer
│   ├── metrics.py             # Prometheus metrics
│   └── requirements.txt       # Dependencies
├── benchmarks/
│   ├── load_test.py           # Locust load testing
│   ├── latency_test.py        # P50/P95/P99 measurements
│   └── throughput_test.py     # Requests per second
└── optimization-guides/
    ├── quantization.md        # Quantization strategies
    ├── batching.md            # Batching best practices
    └── caching.md             # Caching patterns
```

## 🚀 Quick Start

### 1. Install vLLM

```bash
pip install vllm
```

### 2. Basic vLLM Inference

```python
from vllm import LLM, SamplingParams

# Initialize model
llm = LLM(model="facebook/opt-125m")

# Define sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=100
)

# Generate
prompts = ["Hello, my name is", "The future of AI is"]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}")
```

### 3. Quantized Model with vLLM

```python
from vllm import LLM, SamplingParams

# Load with quantization
llm = LLM(
    model="TheBloke/Llama-2-7B-GPTQ",
    quantization="gptq",
    dtype="float16"
)
```

### 4. FastAPI Service Example

```python
from fastapi import FastAPI
from vllm import LLM, SamplingParams
from pydantic import BaseModel

app = FastAPI()
llm = LLM(model="facebook/opt-125m")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7

@app.post("/generate")
async def generate(request: GenerateRequest):
    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )
    outputs = llm.generate([request.prompt], sampling_params)
    return {"generated_text": outputs[0].outputs[0].text}
```

## 📊 Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| **P50 Latency** | <50ms | 50th percentile response time |
| **P95 Latency** | <100ms | 95th percentile response time |
| **P99 Latency** | <200ms | 99th percentile response time |
| **Throughput** | >100 req/s | Requests per second |
| **GPU Memory** | <8GB | Memory utilization |

## 🎯 Week Projects

### Project 1: vLLM Basics (Days 1-2)
- Set up vLLM with a small model
- Implement continuous batching
- Compare throughput vs. standard HuggingFace

### Project 2: Quantization Optimization (Days 3-4)
- Load GPTQ/AWQ quantized models
- Benchmark memory usage and latency
- Compare quality vs. full precision

### Project 3: Production API (Days 5-7)
- Build FastAPI service with vLLM
- Implement Redis caching
- Add Prometheus metrics
- Load test and optimize for <100ms P95

### Project 4: Advanced Optimizations (Days 8-10)
- Implement prefix caching
- Test tensor parallelism (if multi-GPU available)
- Compare vLLM vs. TGI performance

## 📖 Resources

### Official Documentation
- [vLLM Documentation](https://docs.vllm.ai/)
- [Text Generation Inference](https://github.com/huggingface/text-generation-inference)
- [GPTQ Paper](https://arxiv.org/abs/2210.17323)
- [AWQ Paper](https://arxiv.org/abs/2306.00978)

### Tutorials
- [vLLM QuickStart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [PagedAttention Blog](https://blog.vllm.ai/2023/06/20/vllm.html)
- [Quantization Guide](https://huggingface.co/docs/transformers/main/en/quantization)

### Videos
- [vLLM Overview](https://www.youtube.com/watch?v=80bIUggRJf4)
- [LLM Inference at Scale](https://www.youtube.com/watch?v=H3g9qBP0lNc)

## 🔍 Key Takeaways

1. **vLLM's PagedAttention** reduces memory waste and enables continuous batching
2. **Quantization** can reduce memory 4-8x with minimal quality loss
3. **Dynamic batching** maximizes GPU utilization vs. static batching
4. **Caching strategies** can dramatically reduce latency for common queries
5. **Production APIs** require monitoring, metrics, and load testing

## 💡 Pro Tips

1. **Start small**: Use OPT-125M or similar small models for initial testing
2. **Profile everything**: Use Prometheus + Grafana to identify bottlenecks
3. **Test quantization quality**: Always benchmark output quality vs. speed
4. **Monitor GPU memory**: Use `nvidia-smi` to track memory usage
5. **Load test early**: Don't wait until the end to test performance

## ✅ Success Criteria

- [ ] Successfully run vLLM with continuous batching
- [ ] Deploy quantized model with <50% memory vs. full precision
- [ ] Build FastAPI service with <100ms P95 latency
- [ ] Implement caching that improves hit rate >30%
- [ ] Load test showing >100 requests/second throughput
- [ ] Document performance metrics and optimization decisions

## 🔜 Next Week

**Week 17-18**: LLM Training - Dataset curation, training pipelines, and efficient training techniques

---

**Ready to optimize!** 🚀 Start with the vLLM basics and work your way up to the production API.
