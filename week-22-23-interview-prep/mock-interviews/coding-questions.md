# Coding Interview Questions for AI Engineers

## 🎯 Overview

AI/ML engineer coding interviews focus on:
- **ML fundamentals** (implementing components from scratch)
- **Data processing** (efficient text/data manipulation)
- **API design** (building production-ready ML services)
- **Optimization** (latency, memory, cost)

Time limit: **30-45 minutes per question**

---

## 🟢 Easy Questions (15-20 min)

### 1. Implement Semantic Search with Cosine Similarity

**Problem**: Given a list of document embeddings and a query embedding, return the top-k most similar documents.

```python
import numpy as np
from typing import List, Tuple

def semantic_search(
    query_embedding: np.ndarray,  # Shape: (embedding_dim,)
    doc_embeddings: np.ndarray,   # Shape: (num_docs, embedding_dim)
    doc_ids: List[str],
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """
    Return top-k documents ranked by cosine similarity.

    Args:
        query_embedding: Query vector
        doc_embeddings: Document vectors (normalized)
        doc_ids: Document identifiers
        top_k: Number of results to return

    Returns:
        List of (doc_id, similarity_score) tuples, sorted by score (desc)

    Example:
        >>> query = np.array([0.1, 0.2, 0.3])
        >>> docs = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        >>> ids = ["doc1", "doc2"]
        >>> semantic_search(query, docs, ids, top_k=2)
        [("doc1", 1.0), ("doc2", 0.975)]
    """
    # TODO: Implement
    pass
```

**Expected solution**:

```python
def semantic_search(
    query_embedding: np.ndarray,
    doc_embeddings: np.ndarray,
    doc_ids: List[str],
    top_k: int = 5
) -> List[Tuple[str, float]]:
    # Normalize query (in case not already normalized)
    query_norm = query_embedding / np.linalg.norm(query_embedding)

    # Compute cosine similarity (dot product since vectors are normalized)
    similarities = doc_embeddings @ query_norm

    # Get top-k indices
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]

    # Return results
    return [
        (doc_ids[i], float(similarities[i]))
        for i in top_k_indices
    ]

# Time complexity: O(n * d) where n = num_docs, d = embedding_dim
# Space complexity: O(n)
```

**Follow-up questions**:
- How would you optimize for very large document collections (millions)?
- What if documents aren't pre-normalized?
- How would you add filtering (e.g., only search documents from category X)?

---

### 2. Implement LRU Cache for LLM Responses

**Problem**: Implement an LRU (Least Recently Used) cache for storing LLM responses.

```python
from typing import Optional

class LLMCache:
    """
    LRU cache for LLM responses.

    Example:
        >>> cache = LLMCache(capacity=2)
        >>> cache.get("hello")  # Miss
        None
        >>> cache.put("hello", "Hi there!")
        >>> cache.get("hello")  # Hit
        "Hi there!"
        >>> cache.put("goodbye", "See you!")
        >>> cache.put("thanks", "You're welcome!")  # Evicts "hello"
        >>> cache.get("hello")  # Miss (was evicted)
        None
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        # TODO: Initialize data structures

    def get(self, query: str) -> Optional[str]:
        """Get cached response for query (None if not found)"""
        # TODO: Implement
        pass

    def put(self, query: str, response: str) -> None:
        """Store response for query"""
        # TODO: Implement
        pass
```

**Expected solution**:

```python
from collections import OrderedDict

class LLMCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, query: str) -> Optional[str]:
        if query not in self.cache:
            return None

        # Move to end (most recently used)
        self.cache.move_to_end(query)
        return self.cache[query]

    def put(self, query: str, response: str) -> None:
        if query in self.cache:
            # Update and move to end
            self.cache.move_to_end(query)
        else:
            # Add new item
            if len(self.cache) >= self.capacity:
                # Evict least recently used (first item)
                self.cache.popitem(last=False)

        self.cache[query] = response

# Time complexity: O(1) for both get and put
# Space complexity: O(capacity)
```

**Follow-up questions**:
- How would you add semantic similarity matching (not just exact match)?
- How would you add TTL (time-to-live)?
- How would you make this thread-safe?

---

### 3. Implement Token Counter

**Problem**: Estimate token count for a string (for LLM APIs that charge per token).

```python
def count_tokens(text: str) -> int:
    """
    Estimate token count using simple heuristic:
    - 1 token ≈ 4 characters for English text
    - Punctuation and spaces count as separate tokens

    Example:
        >>> count_tokens("Hello, world!")
        4
        >>> count_tokens("The quick brown fox jumps over the lazy dog.")
        12
    """
    # TODO: Implement
    pass
```

**Expected solution**:

```python
import re

def count_tokens(text: str) -> int:
    # Split on whitespace and punctuation
    tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
    return len(tokens)

# More accurate version using tiktoken (OpenAI's tokenizer):
def count_tokens_accurate(text: str, model: str = "gpt-4") -> int:
    import tiktoken
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))
```

**Follow-up questions**:
- How would you handle non-English text?
- What if you need exact token count (not estimate)?
- How would you optimize for very long texts?

---

## 🟡 Medium Questions (25-35 min)

### 4. Implement Sliding Window Chunker

**Problem**: Split long text into chunks with sliding window (for RAG systems).

```python
from typing import List

def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
    separators: List[str] = ["\n\n", "\n", ". ", " "]
) -> List[str]:
    """
    Chunk text with sliding window, respecting sentence boundaries.

    Args:
        text: Input text
        chunk_size: Target chunk size (in characters)
        overlap: Overlap between chunks (in characters)
        separators: Boundaries to split on (in priority order)

    Returns:
        List of text chunks

    Example:
        >>> text = "First sentence. Second sentence. Third sentence."
        >>> chunks = chunk_text(text, chunk_size=30, overlap=10)
        >>> len(chunks)
        2
    """
    # TODO: Implement
    pass
```

**Expected solution**:

```python
def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
    separators: List[str] = ["\n\n", "\n", ". ", " "]
) -> List[str]:
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        # Define chunk end
        end = min(start + chunk_size, len(text))

        # If not at text end, try to break at separator
        if end < len(text):
            # Try each separator in priority order
            for sep in separators:
                # Find last occurrence of separator before end
                sep_idx = text.rfind(sep, start, end)
                if sep_idx != -1:
                    end = sep_idx + len(sep)
                    break

        # Extract chunk
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start forward (with overlap)
        start = end - overlap

        # Avoid infinite loop
        if start <= chunks[-1] if chunks else 0:
            start = end

    return chunks

# Time complexity: O(n * s) where n = text length, s = num separators
# Space complexity: O(n)
```

**Follow-up questions**:
- How would you ensure no chunk exceeds a token limit (not character limit)?
- How would you preserve context across chunks (e.g., section headers)?
- How would you optimize for very large documents (GB size)?

---

### 5. Implement Prompt Template Engine

**Problem**: Create a simple template engine for prompts with variable substitution.

```python
from typing import Dict, Any

class PromptTemplate:
    """
    Template engine for LLM prompts.

    Example:
        >>> template = PromptTemplate(
        ...     "Hello {name}! Your score is {score}."
        ... )
        >>> template.format(name="Alice", score=95)
        "Hello Alice! Your score is 95."

        >>> template = PromptTemplate(
        ...     "Answer the question: {question}\n\nContext: {context}"
        ... )
        >>> template.format(
        ...     question="What is AI?",
        ...     context="AI stands for Artificial Intelligence..."
        ... )
        "Answer the question: What is AI?\n\nContext: AI stands for..."
    """
    def __init__(self, template: str):
        self.template = template
        # TODO: Parse template and identify variables

    def format(self, **kwargs: Any) -> str:
        """Fill in template with provided variables"""
        # TODO: Implement
        pass

    def get_variables(self) -> List[str]:
        """Return list of variable names in template"""
        # TODO: Implement
        pass
```

**Expected solution**:

```python
import re
from typing import Dict, Any, List

class PromptTemplate:
    def __init__(self, template: str):
        self.template = template
        # Find all variables in {var_name} format
        self.variables = re.findall(r'\{(\w+)\}', template)

    def format(self, **kwargs: Any) -> str:
        # Check all required variables are provided
        missing = set(self.variables) - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing variables: {missing}")

        # Substitute variables
        result = self.template
        for var, value in kwargs.items():
            result = result.replace(f"{{{var}}}", str(value))

        return result

    def get_variables(self) -> List[str]:
        return list(set(self.variables))

# Alternative: Use f-string style (more Pythonic)
class PromptTemplateAdvanced:
    def __init__(self, template: str):
        self.template = template

    def format(self, **kwargs: Any) -> str:
        return self.template.format(**kwargs)

    def get_variables(self) -> List[str]:
        from string import Formatter
        return [
            field_name
            for _, field_name, _, _ in Formatter().parse(self.template)
            if field_name is not None
        ]
```

**Follow-up questions**:
- How would you add default values for variables?
- How would you add conditional sections (if/else)?
- How would you handle nested templates?

---

### 6. Implement Rate Limiter for LLM API

**Problem**: Implement token bucket rate limiter for API requests.

```python
import time
from typing import Optional

class RateLimiter:
    """
    Token bucket rate limiter.

    Example:
        >>> limiter = RateLimiter(max_requests=10, window_seconds=60)
        >>> limiter.allow_request()  # First request
        True
        >>> # ... 9 more requests ...
        >>> limiter.allow_request()  # 11th request
        False
    """
    def __init__(self, max_requests: int, window_seconds: int):
        """
        Args:
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        # TODO: Initialize state

    def allow_request(self, tokens: int = 1) -> bool:
        """
        Check if request is allowed.

        Args:
            tokens: Number of tokens to consume (default 1)

        Returns:
            True if request allowed, False otherwise
        """
        # TODO: Implement
        pass

    def wait_time(self) -> float:
        """Return seconds to wait before next request is allowed"""
        # TODO: Implement
        pass
```

**Expected solution**:

```python
import time

class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.tokens = max_requests
        self.last_update = time.time()
        self.refill_rate = max_requests / window_seconds

    def allow_request(self, tokens: int = 1) -> bool:
        # Refill tokens based on time elapsed
        now = time.time()
        elapsed = now - self.last_update

        # Add tokens (up to max)
        self.tokens = min(
            self.max_requests,
            self.tokens + elapsed * self.refill_rate
        )
        self.last_update = now

        # Check if enough tokens
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def wait_time(self) -> float:
        if self.tokens >= 1:
            return 0.0
        # Time to accumulate 1 token
        return (1 - self.tokens) / self.refill_rate

# Time complexity: O(1)
# Space complexity: O(1)
```

**Follow-up questions**:
- How would you make this distributed (multiple servers)?
- How would you add different rate limits per user tier?
- How would you handle burst traffic?

---

### 7. Implement Reciprocal Rank Fusion

**Problem**: Combine multiple ranked lists into a single ranked list (for hybrid search).

```python
from typing import List, Dict

def reciprocal_rank_fusion(
    ranked_lists: List[List[str]],
    k: int = 60
) -> List[str]:
    """
    Combine multiple ranked lists using RRF algorithm.

    Formula: score(doc) = Σ(1 / (k + rank))

    Args:
        ranked_lists: List of ranked document IDs (best first)
        k: Constant (default 60, from research)

    Returns:
        Combined ranked list

    Example:
        >>> list1 = ["doc1", "doc2", "doc3"]
        >>> list2 = ["doc2", "doc1", "doc4"]
        >>> reciprocal_rank_fusion([list1, list2])
        ["doc2", "doc1", "doc3", "doc4"]  # doc2 ranks high in both
    """
    # TODO: Implement
    pass
```

**Expected solution**:

```python
def reciprocal_rank_fusion(
    ranked_lists: List[List[str]],
    k: int = 60
) -> List[str]:
    scores = {}

    # Compute RRF score for each doc
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list):
            score = 1.0 / (k + rank + 1)  # +1 because rank is 0-indexed

            if doc_id in scores:
                scores[doc_id] += score
            else:
                scores[doc_id] = score

    # Sort by score (descending)
    sorted_docs = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [doc_id for doc_id, score in sorted_docs]

# Time complexity: O(n * m) where n = num_lists, m = avg list length
# Space complexity: O(unique_docs)
```

**Follow-up questions**:
- How would you add weights to different lists (e.g., semantic more important than keyword)?
- How would you handle ties (same score)?
- How would you optimize for very long lists?

---

## 🔴 Hard Questions (35-45 min)

### 8. Implement Streaming Text Generator with Backpressure

**Problem**: Implement a streaming text generator that respects backpressure (client can't keep up).

```python
import asyncio
from typing import AsyncIterator

class StreamingGenerator:
    """
    Stream text tokens with backpressure handling.

    Example:
        >>> async def generate():
        ...     generator = StreamingGenerator(model="gpt-4")
        ...     async for token in generator.stream("Hello"):
        ...         print(token, end="", flush=True)
        ...         await asyncio.sleep(0.1)  # Simulate slow client
    """
    def __init__(self, model: str, buffer_size: int = 10):
        self.model = model
        self.buffer_size = buffer_size

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        """Stream tokens with backpressure"""
        # TODO: Implement
        pass
```

**Expected solution**:

```python
import asyncio
from typing import AsyncIterator

class StreamingGenerator:
    def __init__(self, model: str, buffer_size: int = 10):
        self.model = model
        self.buffer_size = buffer_size

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        # Create bounded queue for backpressure
        queue = asyncio.Queue(maxsize=self.buffer_size)

        # Producer task (generates tokens)
        async def producer():
            try:
                # Simulate token generation
                tokens = self._generate_tokens(prompt)
                for token in tokens:
                    # This will block if queue is full (backpressure!)
                    await queue.put(token)
                    await asyncio.sleep(0.01)  # Simulate generation time

                # Signal completion
                await queue.put(None)
            except Exception as e:
                await queue.put(e)

        # Start producer in background
        producer_task = asyncio.create_task(producer())

        # Consumer (yield tokens)
        try:
            while True:
                token = await queue.get()

                if token is None:  # End of stream
                    break

                if isinstance(token, Exception):
                    raise token

                yield token
        finally:
            # Cleanup
            producer_task.cancel()
            try:
                await producer_task
            except asyncio.CancelledError:
                pass

    def _generate_tokens(self, prompt: str) -> List[str]:
        # Simulate tokenization
        return prompt.split()
```

**Follow-up questions**:
- How would you handle errors in the producer?
- How would you add timeout (max time per token)?
- How would you monitor queue depth (for metrics)?

---

### 9. Implement Query Rewriter with Cache Optimization

**Problem**: Rewrite user queries to improve search results, with caching for common patterns.

```python
from typing import List, Optional

class QueryRewriter:
    """
    Rewrite queries to improve search results.

    Examples:
        - "how to use x?" → "using x tutorial guide"
        - "best practices for y" → "y best practices guide"
        - Fix typos: "machien learning" → "machine learning"
    """
    def __init__(self):
        self.cache = {}  # Query -> rewritten
        self.patterns = self._load_patterns()

    def rewrite(self, query: str) -> str:
        """Rewrite query to improve results"""
        # TODO: Implement with caching
        pass

    def _load_patterns(self) -> List[Dict]:
        """Load rewrite patterns (regex + replacement)"""
        return [
            {"pattern": r"how to (use|do) (\w+)", "replace": r"\2 tutorial guide"},
            {"pattern": r"best practices for (\w+)", "replace": r"\1 best practices guide"},
            # ... more patterns
        ]
```

**Expected solution**:

```python
import re
from typing import List, Dict, Optional
import hashlib

class QueryRewriter:
    def __init__(self, use_llm: bool = False):
        self.cache = {}
        self.patterns = self._load_patterns()
        self.use_llm = use_llm

    def rewrite(self, query: str) -> str:
        # Normalize
        query = query.lower().strip()

        # Check cache
        cache_key = self._hash(query)
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Apply pattern-based rewrites
        rewritten = self._apply_patterns(query)

        # Optionally use LLM for complex rewrites
        if self.use_llm and rewritten == query:
            rewritten = self._llm_rewrite(query)

        # Cache result
        self.cache[cache_key] = rewritten

        return rewritten

    def _apply_patterns(self, query: str) -> str:
        for pattern_dict in self.patterns:
            pattern = pattern_dict["pattern"]
            replacement = pattern_dict["replace"]

            match = re.search(pattern, query)
            if match:
                return re.sub(pattern, replacement, query)

        return query

    def _llm_rewrite(self, query: str) -> str:
        # Call LLM to rewrite query
        prompt = f"""Rewrite this search query to be more effective:
        Original: {query}
        Rewritten:"""

        # response = llm.generate(prompt)
        # return response.strip()

        return query  # Placeholder

    def _hash(self, query: str) -> str:
        return hashlib.md5(query.encode()).hexdigest()

    def _load_patterns(self) -> List[Dict]:
        return [
            {"pattern": r"how (?:to|do i) (.+)", "replace": r"\1 tutorial"},
            {"pattern": r"what is (.+)", "replace": r"\1 definition explanation"},
            {"pattern": r"best (.+) for (.+)", "replace": r"\2 best \1"},
            {"pattern": r"(.+) vs (.+)", "replace": r"\1 \2 comparison"},
        ]
```

**Follow-up questions**:
- How would you measure if rewrites actually improve results?
- How would you handle multi-language queries?
- How would you update patterns based on feedback?

---

### 10. Implement Document Deduplicator

**Problem**: Find and remove near-duplicate documents efficiently.

```python
from typing import List, Set
import numpy as np

class DocumentDeduplicator:
    """
    Find and remove near-duplicate documents.

    Uses MinHash + LSH for efficiency.
    """
    def __init__(self, threshold: float = 0.9):
        """
        Args:
            threshold: Similarity threshold (0-1) for considering duplicates
        """
        self.threshold = threshold

    def find_duplicates(
        self,
        documents: List[str]
    ) -> List[Set[int]]:
        """
        Find groups of near-duplicate documents.

        Args:
            documents: List of document texts

        Returns:
            List of sets, each set contains indices of duplicate docs

        Example:
            >>> docs = ["hello world", "hello world!", "goodbye"]
            >>> dedup = DocumentDeduplicator(threshold=0.9)
            >>> dedup.find_duplicates(docs)
            [{0, 1}]  # docs 0 and 1 are duplicates
        """
        # TODO: Implement
        pass
```

**Expected solution** (MinHash + LSH):

```python
from typing import List, Set, Dict
import hashlib
import re

class DocumentDeduplicator:
    def __init__(self, threshold: float = 0.9, num_hashes: int = 100):
        self.threshold = threshold
        self.num_hashes = num_hashes

    def find_duplicates(self, documents: List[str]) -> List[Set[int]]:
        # Compute MinHash signatures
        signatures = [self._minhash(doc) for doc in documents]

        # Use LSH to find candidates
        candidates = self._lsh(signatures)

        # Verify candidates with exact Jaccard similarity
        duplicates = []
        seen = set()

        for i, j in candidates:
            if i in seen or j in seen:
                continue

            # Compute exact similarity
            sim = self._jaccard_similarity(documents[i], documents[j])

            if sim >= self.threshold:
                # Found duplicate pair
                group = {i, j}

                # Merge with existing groups
                merged = False
                for existing_group in duplicates:
                    if group & existing_group:
                        existing_group.update(group)
                        merged = True
                        break

                if not merged:
                    duplicates.append(group)

                seen.update(group)

        return duplicates

    def _minhash(self, text: str) -> np.ndarray:
        """Compute MinHash signature"""
        shingles = self._get_shingles(text)
        signature = []

        for i in range(self.num_hashes):
            min_hash = float('inf')
            for shingle in shingles:
                # Hash with seed i
                h = int(hashlib.md5(f"{i}{shingle}".encode()).hexdigest(), 16)
                min_hash = min(min_hash, h)
            signature.append(min_hash)

        return np.array(signature)

    def _get_shingles(self, text: str, k: int = 3) -> Set[str]:
        """Get k-shingles (character n-grams)"""
        text = re.sub(r'\s+', ' ', text.lower())
        return {text[i:i+k] for i in range(len(text) - k + 1)}

    def _lsh(self, signatures: List[np.ndarray]) -> Set[tuple]:
        """Find candidate pairs using LSH"""
        bands = 20
        rows_per_band = self.num_hashes // bands

        buckets = {}
        candidates = set()

        for doc_idx, sig in enumerate(signatures):
            for band_idx in range(bands):
                # Get band
                start = band_idx * rows_per_band
                end = start + rows_per_band
                band = tuple(sig[start:end])

                # Hash band
                band_hash = hash(band)

                # Add to bucket
                if band_hash not in buckets:
                    buckets[band_hash] = []

                # Check for candidates in same bucket
                for other_idx in buckets[band_hash]:
                    candidates.add(
                        (min(doc_idx, other_idx), max(doc_idx, other_idx))
                    )

                buckets[band_hash].append(doc_idx)

        return candidates

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Compute Jaccard similarity"""
        shingles1 = self._get_shingles(text1)
        shingles2 = self._get_shingles(text2)

        intersection = len(shingles1 & shingles2)
        union = len(shingles1 | shingles2)

        return intersection / union if union > 0 else 0.0

# Time complexity: O(n * m) where n = num_docs, m = doc_length
# Space complexity: O(n * num_hashes)
```

**Follow-up questions**:
- How would you scale to millions of documents?
- How would you handle different similarity thresholds for different document types?
- How would you incorporate semantic similarity (embeddings) instead of just text?

---

## 🎯 Interview Tips

### 1. Communication
- **Think out loud**: Explain your approach before coding
- **Ask clarifying questions**: Input format, edge cases, constraints
- **Discuss trade-offs**: Time vs space, accuracy vs speed

### 2. Code Quality
- **Clean code**: Clear variable names, good structure
- **Handle edge cases**: Empty inputs, None values, large inputs
- **Add comments**: Explain complex logic
- **Type hints**: Use type annotations (Python 3.6+)

### 3. Testing
- **Test as you go**: Run examples mentally or on paper
- **Cover edge cases**: Empty, single element, large inputs
- **Discuss test strategy**: Unit tests, integration tests

### 4. Optimization
- **Start simple**: Get a working solution first
- **Optimize if asked**: Discuss time/space complexity
- **Know trade-offs**: When to optimize and when not to

### 5. ML-Specific
- **Understand the ML context**: Why would you use this in an ML system?
- **Discuss productionization**: How would this scale? What could go wrong?
- **Consider data**: What if data distribution changes?

---

## 📚 Practice Resources

- **LeetCode**: ML/AI tagged problems
- **HackerRank**: AI challenges
- **Kaggle**: Competitions with coding challenges
- **GitHub**: Study production ML code (Hugging Face, LangChain)

---

**Next**: [Behavioral Questions](./behavioral-questions.md)
**Back**: [System Design Questions](./system-design-questions.md)
