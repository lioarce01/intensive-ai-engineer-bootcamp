# Caching Optimization Guide

## Overview

Caching stores computation results to avoid redundant work. For LLM inference, we cache:
1. **Response caching** - Cache complete generated responses
2. **KV caching** - Cache attention key-value pairs
3. **Prefix caching** - Cache common prompt prefixes

## Types of Caching

### 1. Response Caching

**What:** Store complete responses for identical queries.

**When to use:**
- FAQs and common questions
- Identical API calls
- Low temperature (deterministic outputs)

**Implementation:**

```python
import redis
import hashlib
import json

class ResponseCache:
    def __init__(self):
        self.redis = redis.Redis(host='localhost', port=6379)

    def get_cache_key(self, prompt, params):
        """Generate cache key from prompt and parameters."""
        data = {
            "prompt": prompt,
            "temperature": params.temperature,
            "max_tokens": params.max_tokens,
        }
        key_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, key):
        """Get cached response."""
        cached = self.redis.get(key)
        if cached:
            return json.loads(cached)
        return None

    def set(self, key, response, ttl=3600):
        """Cache response with TTL."""
        self.redis.setex(key, ttl, json.dumps(response))

# Usage
cache = ResponseCache()

# Before inference
cache_key = cache.get_cache_key(prompt, sampling_params)
cached_response = cache.get(cache_key)

if cached_response:
    return cached_response  # Cache hit!

# After inference
response = generate(prompt, sampling_params)
cache.set(cache_key, response, ttl=3600)
```

**Pros:**
- Instant response for cached queries
- Reduces GPU load
- Lower costs

**Cons:**
- Stale responses if data changes
- Storage costs (Redis)
- Only works for identical queries

### 2. KV Caching

**What:** Cache attention key-value pairs during generation to avoid recomputation.

**How it works:**
- During attention, each token computes K (key) and V (value) tensors
- These are reused for subsequent tokens
- vLLM implements this automatically via PagedAttention

**vLLM's KV Caching:**

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="facebook/opt-125m",
    # KV caching is enabled by default
    # More GPU memory = more KV cache capacity
    gpu_memory_utilization=0.9,
)
```

**Benefits:**
- ~10-100x faster than recomputing
- Automatic in vLLM
- No code changes needed

**Memory usage:**

```
KV cache memory = 2 * num_layers * d_model * sequence_length * batch_size
```

For Llama-2-7B (32 layers, d_model=4096):
- 512 tokens: ~2GB per request
- 2048 tokens: ~8GB per request

### 3. Prefix Caching

**What:** Cache KV states for common prompt prefixes (system instructions, few-shot examples).

**Use case:**
- System prompts that are identical across requests
- Few-shot examples
- Conversation history

**Example scenario:**

```
Request 1: [System prompt: "You are a helpful AI"] + "What is Python?"
Request 2: [System prompt: "You are a helpful AI"] + "Explain ML"
Request 3: [System prompt: "You are a helpful AI"] + "What is NLP?"

Common prefix: "You are a helpful AI" ← Cache KV states for this!
```

**Implementation:**

```python
class PrefixCache:
    """
    Cache KV states for common prefixes.
    """

    def __init__(self):
        self.cache = {}

    def get_prefix_key(self, prefix: str) -> str:
        """Hash prefix for cache key."""
        return hashlib.md5(prefix.encode()).hexdigest()

    def cache_prefix(self, prefix: str, kv_states):
        """Store KV states for prefix."""
        key = self.get_prefix_key(prefix)
        self.cache[key] = kv_states

    def get_prefix_cache(self, prefix: str):
        """Retrieve cached KV states."""
        key = self.get_prefix_key(prefix)
        return self.cache.get(key)

# Usage with vLLM (conceptual)
system_prompt = "You are a helpful AI assistant."

# First request: compute and cache
user_prompt_1 = "What is Python?"
full_prompt_1 = system_prompt + "\n" + user_prompt_1

# Subsequent requests: reuse cached KV for system_prompt
user_prompt_2 = "What is ML?"
full_prompt_2 = system_prompt + "\n" + user_prompt_2
# vLLM can detect the common prefix and reuse KV cache
```

**vLLM Automatic Prefix Caching:**

vLLM 0.2+ has built-in prefix caching:

```python
llm = LLM(
    model="facebook/opt-125m",
    enable_prefix_caching=True,  # Enable automatic prefix caching
)

# vLLM will automatically detect and cache common prefixes
prompts = [
    "System: You are helpful\nUser: Question 1",
    "System: You are helpful\nUser: Question 2",
    "System: You are helpful\nUser: Question 3",
]
outputs = llm.generate(prompts, sampling_params)
# "System: You are helpful" KV states are cached and reused!
```

## Caching Strategies

### Strategy 1: Semantic Caching

Cache responses for semantically similar queries, not just identical ones.

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class SemanticCache:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = faiss.IndexFlatL2(384)  # Embedding dimension
        self.cache = {}
        self.prompts = []

    def add(self, prompt: str, response: str):
        """Add prompt-response pair to cache."""
        embedding = self.model.encode([prompt])[0]
        idx = len(self.prompts)

        self.index.add(np.array([embedding]))
        self.prompts.append(prompt)
        self.cache[idx] = response

    def get(self, prompt: str, threshold: float = 0.9) -> str:
        """Find semantically similar cached response."""
        if len(self.prompts) == 0:
            return None

        embedding = self.model.encode([prompt])[0]
        distances, indices = self.index.search(np.array([embedding]), k=1)

        # Check similarity (lower distance = more similar)
        if distances[0][0] < (1 - threshold):
            return self.cache[indices[0][0]]

        return None

# Usage
semantic_cache = SemanticCache()

# Add to cache
semantic_cache.add("What is AI?", "AI is artificial intelligence...")

# Query with similar prompt
response = semantic_cache.get("Can you explain AI?")  # May return cached response!
```

### Strategy 2: Layered Caching

Use multiple cache layers for different needs:

```
L1: In-memory cache (fastest, small capacity)
 ↓ (miss)
L2: Redis cache (fast, medium capacity)
 ↓ (miss)
L3: Generate with LLM (slow, unlimited)
```

```python
class LayeredCache:
    def __init__(self):
        self.l1_cache = {}  # In-memory
        self.l2_cache = redis.Redis()  # Redis
        self.max_l1_size = 1000

    def get(self, key: str):
        # Try L1
        if key in self.l1_cache:
            return self.l1_cache[key]

        # Try L2
        cached = self.l2_cache.get(key)
        if cached:
            response = json.loads(cached)
            # Promote to L1
            self._set_l1(key, response)
            return response

        return None

    def set(self, key: str, value: dict, ttl: int = 3600):
        # Set in both layers
        self._set_l1(key, value)
        self.l2_cache.setex(key, ttl, json.dumps(value))

    def _set_l1(self, key: str, value: dict):
        # Evict if full
        if len(self.l1_cache) >= self.max_l1_size:
            # Simple FIFO eviction
            oldest_key = next(iter(self.l1_cache))
            del self.l1_cache[oldest_key]

        self.l1_cache[key] = value
```

### Strategy 3: TTL-based Caching

Set appropriate Time-To-Live based on data freshness requirements:

```python
# Dynamic TTL based on query type
def get_ttl(prompt: str) -> int:
    if "latest news" in prompt.lower():
        return 300  # 5 minutes (frequently changing)
    elif "historical fact" in prompt.lower():
        return 86400  # 24 hours (rarely changes)
    else:
        return 3600  # 1 hour (default)

# Usage
ttl = get_ttl(prompt)
cache.set(cache_key, response, ttl=ttl)
```

## Measuring Cache Performance

### Key Metrics

1. **Hit Rate:**
   ```
   Hit Rate = Cache Hits / (Cache Hits + Cache Misses)
   ```

2. **Hit Latency:**
   - Time to retrieve from cache
   - Target: < 10ms

3. **Miss Latency:**
   - Time to generate + cache
   - Target: < 100ms (P95)

4. **Memory Usage:**
   - Cache size in bytes
   - Monitor and set limits

### Implementation:

```python
class CacheMetrics:
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.hit_latencies = []
        self.miss_latencies = []

    def record_hit(self, latency_ms: float):
        self.hits += 1
        self.hit_latencies.append(latency_ms)

    def record_miss(self, latency_ms: float):
        self.misses += 1
        self.miss_latencies.append(latency_ms)

    def get_stats(self):
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0

        return {
            "hit_rate": hit_rate,
            "hits": self.hits,
            "misses": self.misses,
            "avg_hit_latency": np.mean(self.hit_latencies) if self.hit_latencies else 0,
            "avg_miss_latency": np.mean(self.miss_latencies) if self.miss_latencies else 0,
        }

# Usage
metrics = CacheMetrics()

start = time.time()
cached = cache.get(key)
latency = (time.time() - start) * 1000

if cached:
    metrics.record_hit(latency)
else:
    # Generate and cache
    response = generate(prompt)
    cache.set(key, response)
    metrics.record_miss((time.time() - start) * 1000)

# Print stats
stats = metrics.get_stats()
print(f"Hit rate: {stats['hit_rate']:.1f}%")
```

## Cache Invalidation

> "There are only two hard things in Computer Science: cache invalidation and naming things." - Phil Karlton

### Strategies:

1. **TTL-based:** Set expiration time
   ```python
   cache.setex(key, ttl=3600, value=response)  # Expires after 1 hour
   ```

2. **Event-based:** Invalidate when data changes
   ```python
   def on_data_update(event):
       cache.delete_pattern("affected_*")
   ```

3. **Version-based:** Include version in cache key
   ```python
   key = f"{prompt}:{model_version}:{data_version}"
   ```

4. **LRU eviction:** Automatically evict least recently used
   ```python
   from functools import lru_cache

   @lru_cache(maxsize=1000)
   def generate_cached(prompt):
       return generate(prompt)
   ```

## Production Best Practices

### 1. Set Reasonable TTLs

```python
# Good
cache.set(key, response, ttl=3600)  # 1 hour

# Bad - too long for dynamic content
cache.set(key, response, ttl=86400 * 30)  # 30 days
```

### 2. Monitor Cache Size

```python
# Check Redis memory
info = redis_client.info()
memory_used = info['used_memory_human']

# Set max memory and eviction policy
redis_client.config_set('maxmemory', '2gb')
redis_client.config_set('maxmemory-policy', 'allkeys-lru')
```

### 3. Handle Cache Failures Gracefully

```python
def get_with_fallback(prompt):
    try:
        cached = cache.get(cache_key)
        if cached:
            return cached
    except Exception as e:
        logger.warning(f"Cache get failed: {e}")
        # Continue to generation

    # Generate if cache miss or error
    response = generate(prompt)

    try:
        cache.set(cache_key, response)
    except Exception as e:
        logger.warning(f"Cache set failed: {e}")
        # Still return the response

    return response
```

### 4. Warm Up Cache

Pre-populate cache with common queries:

```python
common_queries = [
    "What is AI?",
    "How does ML work?",
    "Explain deep learning",
]

for query in common_queries:
    response = generate(query)
    cache.set(get_cache_key(query), response)

print(f"Cache warmed with {len(common_queries)} queries")
```

## Caching Anti-patterns

❌ **Don't cache everything**
- Cache only frequently accessed data
- Monitor hit rates

❌ **Don't ignore cache failures**
- Always have fallback to generation
- Log cache errors

❌ **Don't set infinite TTLs**
- Data may become stale
- Set reasonable expiration

❌ **Don't cache errors**
- Only cache successful responses
- Retry on generation failure

## Production Checklist

- [ ] Implement response caching for common queries
- [ ] Monitor cache hit rate (target > 30% for good ROI)
- [ ] Set appropriate TTLs based on data freshness
- [ ] Handle cache failures gracefully
- [ ] Use layered caching for performance
- [ ] Monitor cache memory usage
- [ ] Implement cache invalidation strategy
- [ ] Warm up cache with common queries
- [ ] Add cache metrics to monitoring dashboard

## Resources

- [Redis Caching Guide](https://redis.io/docs/manual/patterns/cache/)
- [vLLM Caching](https://docs.vllm.ai/en/latest/features/caching.html)
- [Cache Implementation](../api-service/cache.py)

## Example: Full Caching Setup

See `week-15-16-optimized-inference/api-service/cache.py` for a complete implementation.

```bash
# Start Redis
docker run -d -p 6379:6379 redis:alpine

# Test cache
cd week-15-16-optimized-inference/api-service
python cache.py
```
