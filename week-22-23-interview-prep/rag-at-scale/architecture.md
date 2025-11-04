# RAG at Scale: Production Architecture

## 🎯 Overview

Scaling RAG (Retrieval-Augmented Generation) from prototype to production at millions of documents and thousands of QPS requires addressing:
- **Indexing**: Efficient ingestion and updates
- **Search**: Low-latency retrieval with high quality
- **Generation**: Context-aware responses
- **Cost**: Token usage optimization
- **Quality**: Continuous evaluation and improvement

## 📊 Scale Benchmarks

| Scale Tier | Documents | Users | QPS | P95 Latency | Monthly Cost |
|------------|-----------|-------|-----|-------------|--------------|
| **Small** | 10K | 100 | 10 | <500ms | $500 |
| **Medium** | 100K | 1K | 100 | <500ms | $3K |
| **Large** | 1M | 10K | 1K | <500ms | $20K |
| **Enterprise** | 10M+ | 100K+ | 10K+ | <300ms | $150K+ |

## 🏗️ Reference Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Data Ingestion Pipeline                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Documents → Chunking → Embedding → Vector DB → Monitoring      │
│     (S3)      Service    Service    (Pinecone)    (Datadog)     │
│                  ↓          ↓           ↓                        │
│              Metadata   BatchAPI    Indexing                     │
│              Extract    Parallel    Strategy                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      Query Pipeline (Real-time)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  User Query                                                       │
│      ↓                                                            │
│  Query Preprocessing                                              │
│      ↓                                                            │
│  ┌────────────────────────────────┐                              │
│  │  Multi-stage Retrieval         │                              │
│  │  1. Semantic Search (top 100)  │                              │
│  │  2. Keyword Search (top 100)   │                              │
│  │  3. Fusion (top 50)            │                              │
│  │  4. Reranking (top 10)         │                              │
│  └────────────────────────────────┘                              │
│      ↓                                                            │
│  Context Assembly                                                 │
│      ↓                                                            │
│  LLM Generation                                                   │
│      ↓                                                            │
│  Response + Citations                                             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    Observability & Feedback                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Logs → Metrics → Traces → Analytics → Model Improvement         │
│  (Loki)  (Prom)  (Jaeger) (BigQuery)   (Fine-tuning)           │
└─────────────────────────────────────────────────────────────────┘
```

## 🔧 Component Deep Dive

### 1. Data Ingestion Pipeline

**Challenge**: Index millions of documents efficiently with minimal downtime

#### Chunking Service

```python
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib

class OptimizedChunker:
    """
    Production-grade chunking with semantic awareness
    """
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,  # Optimal for most embedding models
            chunk_overlap=50,  # 10% overlap for context preservation
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=self._token_length
        )

    def chunk_document(self, doc: Dict) -> List[Dict]:
        """
        Chunk document with metadata preservation
        """
        text = doc["content"]
        metadata = doc["metadata"]

        chunks = self.splitter.split_text(text)

        # Add chunk-level metadata
        result = []
        for i, chunk in enumerate(chunks):
            chunk_id = self._generate_chunk_id(doc["id"], i)

            result.append({
                "chunk_id": chunk_id,
                "text": chunk,
                "metadata": {
                    **metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "doc_id": doc["id"],
                    "char_count": len(chunk)
                }
            })

        return result

    def _token_length(self, text: str) -> int:
        """Approximate token count (1 token ≈ 4 chars)"""
        return len(text) // 4

    def _generate_chunk_id(self, doc_id: str, chunk_index: int) -> str:
        """Generate deterministic chunk ID"""
        return hashlib.md5(f"{doc_id}:{chunk_index}".encode()).hexdigest()
```

**Key Strategies**:
- **Semantic chunking**: Use separators that preserve meaning
- **Chunk size optimization**: 512 tokens works well for most embedding models
- **Overlap**: 10% overlap prevents context loss at boundaries
- **Metadata preservation**: Keep all metadata at chunk level

#### Embedding Service

```python
import asyncio
from typing import List
from openai import AsyncOpenAI
import numpy as np

class EmbeddingService:
    """
    Batch embedding generation with rate limiting
    """
    def __init__(self, model="text-embedding-3-small"):
        self.client = AsyncOpenAI()
        self.model = model
        self.batch_size = 100  # OpenAI allows up to 100
        self.rate_limit = 3000  # RPM

    async def embed_batch(
        self,
        texts: List[str],
        metadata: List[Dict] = None
    ) -> List[Dict]:
        """
        Embed texts in parallel with rate limiting
        """
        # Split into batches
        batches = [
            texts[i:i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]

        # Rate limiting: max concurrent requests
        semaphore = asyncio.Semaphore(50)

        async def embed_with_limit(batch):
            async with semaphore:
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                return [item.embedding for item in response.data]

        # Process all batches
        results = await asyncio.gather(
            *[embed_with_limit(batch) for batch in batches]
        )

        # Flatten results
        embeddings = [emb for batch in results for emb in batch]

        # Combine with metadata
        return [
            {
                "text": text,
                "embedding": emb,
                "metadata": meta
            }
            for text, emb, meta in zip(texts, embeddings, metadata or [{}] * len(texts))
        ]

    async def embed_query(self, query: str) -> np.ndarray:
        """Single query embedding (cached)"""
        response = await self.client.embeddings.create(
            model=self.model,
            input=query
        )
        return np.array(response.data[0].embedding)
```

**Optimizations**:
- **Batching**: Process 100 texts per API call (OpenAI limit)
- **Parallel processing**: Use asyncio for concurrent requests
- **Rate limiting**: Semaphore to avoid hitting API limits
- **Caching**: Cache embeddings for common queries

#### Vector DB Indexing

```python
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict
import asyncio

class VectorDBManager:
    """
    Manage vector DB indexing at scale
    """
    def __init__(self, index_name: str, dimension: int = 1536):
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = index_name
        self.dimension = dimension

        # Create index if not exists
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )

        self.index = self.pc.Index(index_name)

    async def upsert_vectors(
        self,
        vectors: List[Dict],
        namespace: str = "default",
        batch_size: int = 100
    ):
        """
        Upsert vectors in batches with retry logic
        """
        # Format for Pinecone
        records = [
            (
                vec["chunk_id"],
                vec["embedding"],
                vec["metadata"]
            )
            for vec in vectors
        ]

        # Batch upsert
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]

            try:
                self.index.upsert(
                    vectors=batch,
                    namespace=namespace
                )
            except Exception as e:
                # Retry logic
                await asyncio.sleep(1)
                self.index.upsert(vectors=batch, namespace=namespace)

    async def update_metadata(
        self,
        chunk_id: str,
        metadata: Dict,
        namespace: str = "default"
    ):
        """
        Update metadata without re-embedding
        """
        self.index.update(
            id=chunk_id,
            set_metadata=metadata,
            namespace=namespace
        )

    async def delete_by_filter(
        self,
        filter: Dict,
        namespace: str = "default"
    ):
        """
        Delete vectors by metadata filter
        """
        self.index.delete(
            filter=filter,
            namespace=namespace
        )
```

**Strategies**:
- **Namespaces**: Multi-tenancy (one namespace per customer)
- **Batch upserts**: 100 vectors per request (Pinecone limit)
- **Retry logic**: Handle transient failures
- **Metadata-only updates**: Update metadata without re-embedding

### 2. Query Pipeline (Real-time)

#### Multi-stage Retrieval

```python
class HybridRetriever:
    """
    Hybrid search combining semantic + keyword + reranking
    """
    def __init__(self):
        self.vector_db = VectorDBManager("production")
        self.bm25 = BM25Index()  # Keyword search
        self.reranker = CrossEncoderReranker()
        self.cache = Redis()

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Dict = None
    ) -> List[Dict]:
        """
        Multi-stage retrieval pipeline
        """
        # Stage 1: Check cache
        cache_key = f"query:{hashlib.md5(query.encode()).hexdigest()}"
        cached = await self.cache.get(cache_key)
        if cached:
            return cached

        # Stage 2: Parallel semantic + keyword search
        semantic_results, keyword_results = await asyncio.gather(
            self._semantic_search(query, top_k=100, filters=filters),
            self._keyword_search(query, top_k=100, filters=filters)
        )

        # Stage 3: Fusion (combine + deduplicate)
        fused_results = self._reciprocal_rank_fusion(
            semantic_results,
            keyword_results,
            top_k=50
        )

        # Stage 4: Reranking
        reranked_results = await self._rerank(query, fused_results, top_k=top_k)

        # Cache results (1 hour TTL)
        await self.cache.setex(cache_key, 3600, reranked_results)

        return reranked_results

    async def _semantic_search(
        self,
        query: str,
        top_k: int,
        filters: Dict = None
    ) -> List[Dict]:
        """Vector similarity search"""
        query_embedding = await embedding_service.embed_query(query)

        results = self.vector_db.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            filter=filters,
            include_metadata=True
        )

        return [
            {
                "chunk_id": match.id,
                "score": match.score,
                "text": match.metadata["text"],
                "metadata": match.metadata
            }
            for match in results.matches
        ]

    async def _keyword_search(
        self,
        query: str,
        top_k: int,
        filters: Dict = None
    ) -> List[Dict]:
        """BM25 keyword search"""
        return self.bm25.search(query, top_k=top_k, filters=filters)

    def _reciprocal_rank_fusion(
        self,
        results_list: List[List[Dict]],
        k: int = 60,
        top_k: int = 50
    ) -> List[Dict]:
        """
        Combine multiple ranked lists using RRF
        Formula: score = Σ(1 / (k + rank))
        """
        scores = {}

        for results in results_list:
            for rank, result in enumerate(results):
                chunk_id = result["chunk_id"]
                score = 1.0 / (k + rank + 1)

                if chunk_id in scores:
                    scores[chunk_id]["score"] += score
                else:
                    scores[chunk_id] = {**result, "score": score}

        # Sort by combined score
        sorted_results = sorted(
            scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        return sorted_results[:top_k]

    async def _rerank(
        self,
        query: str,
        results: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """
        Cross-encoder reranking for final precision
        """
        pairs = [(query, result["text"]) for result in results]

        rerank_scores = await self.reranker.predict(pairs)

        # Update scores
        for result, score in zip(results, rerank_scores):
            result["rerank_score"] = score

        # Sort by rerank score
        reranked = sorted(
            results,
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return reranked[:top_k]
```

**Why Multi-stage?**
- **Stage 1 (Semantic)**: Catches semantic similarity (different words, same meaning)
- **Stage 2 (Keyword)**: Catches exact matches (entity names, technical terms)
- **Stage 3 (Fusion)**: Combines best of both worlds
- **Stage 4 (Reranking)**: Final precision boost (+10-15% relevance)

#### Context Assembly

```python
class ContextAssembler:
    """
    Assemble context for LLM generation
    """
    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens

    def assemble_context(
        self,
        query: str,
        retrieved_chunks: List[Dict]
    ) -> str:
        """
        Assemble context with smart truncation
        """
        context_parts = []
        total_tokens = 0

        # Add chunks until hitting token limit
        for i, chunk in enumerate(retrieved_chunks):
            chunk_text = chunk["text"]
            chunk_tokens = self._estimate_tokens(chunk_text)

            if total_tokens + chunk_tokens > self.max_tokens:
                break

            # Add with citation
            context_parts.append(
                f"[{i+1}] {chunk_text}\n"
                f"Source: {chunk['metadata'].get('source', 'Unknown')}"
            )
            total_tokens += chunk_tokens

        return "\n\n".join(context_parts)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate tokens (1 token ≈ 4 chars)"""
        return len(text) // 4
```

### 3. Generation with Citations

```python
class RAGGenerator:
    """
    Generate responses with citations
    """
    def __init__(self):
        self.llm = AsyncOpenAI()
        self.retriever = HybridRetriever()

    async def generate(
        self,
        query: str,
        model: str = "gpt-4-turbo"
    ) -> Dict:
        """
        Full RAG pipeline with citations
        """
        # Retrieve relevant chunks
        chunks = await self.retriever.retrieve(query, top_k=10)

        # Assemble context
        context = ContextAssembler().assemble_context(query, chunks)

        # Generate with prompt
        prompt = self._build_prompt(query, context)

        response = await self.llm.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Use the provided context to answer questions accurately. Always cite sources using [1], [2], etc."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2  # Lower temp for factual responses
        )

        answer = response.choices[0].message.content

        return {
            "answer": answer,
            "sources": [
                {
                    "citation": i+1,
                    "text": chunk["text"],
                    "metadata": chunk["metadata"]
                }
                for i, chunk in enumerate(chunks)
            ]
        }

    def _build_prompt(self, query: str, context: str) -> str:
        return f"""Answer the following question based on the provided context.

Context:
{context}

Question: {query}

Answer (with citations):"""
```

## 📈 Performance Optimization

### Caching Strategy

```python
class MultiLevelCache:
    """
    3-level caching for RAG pipeline
    """
    def __init__(self):
        self.l1 = {}  # In-memory (hot queries)
        self.l2 = Redis()  # Redis (warm queries)
        self.l3 = None  # CDN (static content)

    async def get(self, key: str) -> Optional[Dict]:
        # L1: In-memory
        if key in self.l1:
            return self.l1[key]

        # L2: Redis
        cached = await self.l2.get(key)
        if cached:
            self.l1[key] = cached  # Promote to L1
            return cached

        return None

    async def set(self, key: str, value: Dict, ttl: int = 3600):
        self.l1[key] = value
        await self.l2.setex(key, ttl, value)
```

### Latency Targets

| Component | Target | Optimization |
|-----------|--------|--------------|
| Embedding | <50ms | Batch + cache |
| Vector search | <100ms | Index tuning |
| Reranking | <100ms | GPU inference |
| LLM generation | <2s | Streaming |
| **Total** | **<300ms** | Parallel + cache |

## 💰 Cost Optimization

### Token Usage

```python
class TokenOptimizer:
    """
    Optimize token usage for cost reduction
    """
    def optimize_context(self, chunks: List[str]) -> List[str]:
        """
        Reduce context tokens while preserving quality
        """
        # 1. Remove redundancy
        chunks = self._deduplicate(chunks)

        # 2. Extract key sentences
        chunks = self._extract_key_sentences(chunks)

        # 3. Compress with summarization (for very long contexts)
        if self._estimate_tokens(chunks) > 4000:
            chunks = self._summarize(chunks)

        return chunks

    def _deduplicate(self, chunks: List[str]) -> List[str]:
        """Remove near-duplicate chunks"""
        from sklearn.metrics.pairwise import cosine_similarity

        embeddings = [self._embed(chunk) for chunk in chunks]
        unique_chunks = []

        for i, chunk in enumerate(chunks):
            is_duplicate = False
            for j in range(i):
                sim = cosine_similarity(
                    [embeddings[i]],
                    [embeddings[j]]
                )[0][0]
                if sim > 0.95:  # 95% similar
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_chunks.append(chunk)

        return unique_chunks
```

## 🔍 Quality Monitoring

### Evaluation Pipeline

```python
class RAGEvaluator:
    """
    Evaluate RAG system quality
    """
    def __init__(self):
        self.metrics = {
            "retrieval_metrics": ["precision@k", "recall@k", "mrr"],
            "generation_metrics": ["faithfulness", "relevance", "coherence"]
        }

    async def evaluate_retrieval(
        self,
        queries: List[str],
        ground_truth: List[List[str]]
    ) -> Dict:
        """
        Evaluate retrieval quality
        """
        metrics = {
            "precision@5": [],
            "recall@10": [],
            "mrr": []
        }

        for query, gt_chunks in zip(queries, ground_truth):
            retrieved = await retriever.retrieve(query, top_k=10)

            # Precision@5
            p5 = len(set(retrieved[:5]) & set(gt_chunks)) / 5
            metrics["precision@5"].append(p5)

            # Recall@10
            r10 = len(set(retrieved[:10]) & set(gt_chunks)) / len(gt_chunks)
            metrics["recall@10"].append(r10)

            # MRR
            for i, chunk in enumerate(retrieved):
                if chunk in gt_chunks:
                    metrics["mrr"].append(1.0 / (i + 1))
                    break

        return {k: np.mean(v) for k, v in metrics.items()}
```

---

**Next**: [Indexing Strategies](./indexing-strategies.md)
**Back**: [System Design](../system-design/framework.md)
