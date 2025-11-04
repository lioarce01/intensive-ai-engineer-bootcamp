# Week 11-12: Production RAG System

A production-ready Retrieval-Augmented Generation (RAG) system with **hybrid search**, **reranking**, and **comprehensive monitoring** optimized for **<300ms P95 latency**.

## 🎯 Features

### Core Capabilities
- ✅ **Multiple Chunking Strategies**: Fixed-size, sentence-based, semantic, and recursive chunking
- ✅ **Hybrid Search**: Combines vector similarity (FAISS) with BM25 keyword search
- ✅ **Advanced Reranking**: Cross-encoder and MMR reranking for improved relevance
- ✅ **Comprehensive Monitoring**: Track latency, throughput, and quality metrics
- ✅ **Production-Ready**: Optimized for <300ms P95 latency with minimal overhead

### Performance Optimizations
- Fast vector search with FAISS (flat, IVF, and HNSW indexes)
- Reciprocal Rank Fusion (RRF) for efficient hybrid search
- Configurable reranking with top-k optimization
- Minimal monitoring overhead (<5ms per query)

## 📁 Project Structure

```
week-11-12-rag-production/
├── src/
│   ├── __init__.py           # Package exports
│   ├── chunking.py           # Document chunking strategies
│   ├── vector_store.py       # FAISS vector store
│   ├── hybrid_search.py      # Hybrid search (vector + BM25)
│   ├── reranker.py          # Reranking algorithms
│   ├── monitoring.py         # Performance monitoring
│   └── rag_pipeline.py       # Complete RAG pipeline
├── examples/
│   ├── basic_usage.py        # Simple usage example
│   └── advanced_usage.py     # Advanced features demo
├── tests/
│   └── test_components.py    # Unit tests
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.rag_pipeline import ProductionRAG, RAGConfig

# Create RAG system
config = RAGConfig(
    chunking_strategy="recursive",
    use_hybrid_search=True,
    use_reranking=True,
    latency_target_ms=300
)

rag = ProductionRAG(config=config)

# Ingest documents
documents = [
    "Machine learning is a subset of AI...",
    "Deep learning uses neural networks...",
    # ... more documents
]

rag.ingest_documents(documents)

# Query
docs, scores, metrics = rag.query("What is deep learning?", top_k=5)

print(f"Found {len(docs)} results in {metrics.total_latency:.2f}ms")
for doc, score in zip(docs, scores):
    print(f"Score: {score:.4f} - {doc.text[:100]}...")
```

### Run Examples

```bash
# Basic example
python examples/basic_usage.py

# Advanced example with custom embeddings and monitoring
python examples/advanced_usage.py
```

## 🧩 Components

### 1. Chunking Strategies

Four chunking strategies for different use cases:

```python
from src.chunking import get_chunker

# Fixed-size chunking (simple, fast)
chunker = get_chunker("fixed", chunk_size=512, overlap=50)

# Sentence-based chunking (respects boundaries)
chunker = get_chunker("sentence", target_size=512)

# Semantic chunking (groups related content)
chunker = get_chunker("semantic", min_chunk_size=200, max_chunk_size=800)

# Recursive chunking (best for structured docs)
chunker = get_chunker("recursive", chunk_size=512, overlap=50)

chunks = chunker.chunk(text)
```

**When to use each:**
- **Fixed**: Simple documents, fast processing
- **Sentence**: Maintain sentence coherence
- **Semantic**: Documents with clear topics
- **Recursive**: Structured documents (markdown, code)

### 2. Vector Store (FAISS)

Efficient vector similarity search:

```python
from src.vector_store import FAISSVectorStore, Document

# Create store
store = FAISSVectorStore(
    dimension=384,
    index_type="flat",  # "flat", "ivf", "hnsw"
    metric="cosine"     # "cosine", "l2"
)

# Add documents
documents = [
    Document(id="1", text="...", embedding=embedding1),
    Document(id="2", text="...", embedding=embedding2)
]
store.add_documents(documents)

# Search
results = store.search(query_embedding, k=10)

# Save/load
store.save("./index")
store = FAISSVectorStore.load("./index")
```

**Index types:**
- **Flat**: Exact search, best accuracy (small datasets)
- **IVF**: Fast approximate search (medium datasets)
- **HNSW**: Very fast approximate search (large datasets)

### 3. Hybrid Search

Combines vector similarity and keyword search:

```python
from src.hybrid_search import HybridSearch, BM25

# Setup
vector_store = FAISSVectorStore(dimension=384)
bm25 = BM25()

hybrid = HybridSearch(
    vector_store=vector_store,
    bm25=bm25,
    use_rrf=True,  # Reciprocal Rank Fusion
    vector_weight=0.7,
    bm25_weight=0.3
)

# Add documents
hybrid.add_documents(documents)

# Search
results = hybrid.search(
    query="What is machine learning?",
    query_embedding=query_emb,
    k=10
)
```

**Why hybrid search?**
- Vector search: Great for semantic similarity
- BM25: Great for exact keyword matches
- Combined: Best of both worlds!

### 4. Reranking

Improve search quality with reranking:

```python
from src.reranker import CrossEncoderReranker, MMRReranker

# Cross-encoder reranking (accuracy)
reranker = CrossEncoderReranker(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
)
reranked = reranker.rerank(query, initial_results, top_k=10)

# MMR reranking (diversity)
mmr = MMRReranker(lambda_param=0.7)  # 0=diversity, 1=relevance
diverse_results = mmr.rerank(query_emb, initial_results, top_k=10)
```

**Reranking strategy:**
1. Retrieve top 20-50 candidates (fast)
2. Rerank to top 10 (accurate)
3. Best latency/quality tradeoff

### 5. Monitoring & Observability

Track performance and quality:

```python
from src.monitoring import PerformanceTracker, MetricsDashboard

# Track metrics
tracker = PerformanceTracker()

# Query with tracking
metrics = tracker.start_query(query_id, query_text)
# ... perform query ...
metrics.total_latency = 150.0
tracker.record_metrics(metrics)

# View dashboard
dashboard = MetricsDashboard(tracker)
dashboard.print_summary()

# Get stats
stats = tracker.get_stats()
print(f"P95 latency: {stats['p95_latency_ms']:.2f}ms")

# Export for analysis
tracker.export_to_file("metrics.json")
```

**Key metrics:**
- **Latency**: P50, P95, P99 response times
- **Throughput**: Queries per second
- **Quality**: Top scores, result counts
- **Alerts**: Automatic threshold monitoring

## 📊 Performance Benchmarks

Typical performance on modern hardware (M1/M2 Mac, modern Linux):

| Operation | Latency | Notes |
|-----------|---------|-------|
| Embedding | 5-20ms | Depends on model size |
| Vector Search | 10-50ms | FAISS flat index |
| BM25 Search | 5-15ms | Python implementation |
| Hybrid Search | 20-60ms | RRF combination |
| Reranking (10 docs) | 50-100ms | Cross-encoder |
| **Total (P95)** | **<300ms** | ✅ Production target |

**Optimization tips:**
- Use HNSW index for >100K documents
- Batch queries when possible
- Cache embeddings for repeated queries
- Consider GPU acceleration for embeddings

## 🔧 Configuration

### RAGConfig Options

```python
from src.rag_pipeline import RAGConfig

config = RAGConfig(
    # Chunking
    chunking_strategy="recursive",  # "fixed", "sentence", "semantic", "recursive"
    chunk_size=512,
    chunk_overlap=50,

    # Vector Store
    embedding_dimension=384,
    index_type="flat",             # "flat", "ivf", "hnsw"
    metric="cosine",               # "cosine", "l2"

    # Search
    use_hybrid_search=True,
    vector_weight=0.7,
    bm25_weight=0.3,
    use_rrf=True,                  # Reciprocal Rank Fusion

    # Retrieval
    top_k=10,
    search_k=20,                   # Retrieve more, rerank to top_k

    # Reranking
    use_reranking=True,
    rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",

    # Monitoring
    enable_monitoring=True,
    enable_logging=True,
    latency_target_ms=300
)
```

## 🧪 Testing

Run unit tests:

```bash
pytest tests/
```

Run examples:

```bash
python examples/basic_usage.py
python examples/advanced_usage.py
```

## 📈 Production Deployment

### Best Practices

1. **Use Real Embeddings**: Replace the placeholder embedding function
   ```python
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer('all-MiniLM-L6-v2')
   rag.set_embedding_function(lambda text: model.encode(text))
   ```

2. **Choose Right Index**:
   - <10K docs: Flat index
   - 10K-1M docs: IVF index
   - >1M docs: HNSW index

3. **Monitor Continuously**:
   - Track P95 latency
   - Set up alerts for degradation
   - Log queries for analysis

4. **Optimize Chunking**:
   - Test different strategies
   - Measure retrieval quality
   - A/B test configurations

5. **Cache Aggressively**:
   - Cache embeddings
   - Cache search results
   - Use Redis for distributed cache

### Deployment Checklist

- [ ] Use production embedding model
- [ ] Enable comprehensive logging
- [ ] Set up monitoring dashboard
- [ ] Configure alerts
- [ ] Implement caching layer
- [ ] Load test for peak traffic
- [ ] Set up A/B testing
- [ ] Document all configurations

## 🎓 Key Learnings

### Why Hybrid Search?

Vector search excels at **semantic similarity** but may miss exact keywords. BM25 catches **exact matches** but lacks semantic understanding. Combining both gives:

- Better recall (find more relevant docs)
- Better precision (rank them correctly)
- Robustness to different query styles

### Why Reranking?

Initial retrieval (vector + BM25) is fast but approximate. Cross-encoder reranking:

- Jointly processes query-document pairs
- Much more accurate than bi-encoders
- Only applied to top candidates (efficiency)
- Typical boost: 10-20% better relevance

### Production Latency

Target: **<300ms P95** (300ms or faster for 95% of queries)

Breakdown:
- Embedding: ~20ms (30ms budget)
- Search: ~50ms (100ms budget)
- Reranking: ~80ms (150ms budget)
- Buffer: ~20ms (safety margin)

Monitor each component separately to identify bottlenecks.

## 📚 Resources

### Papers
- [BEIR: Benchmark for Information Retrieval](https://arxiv.org/abs/2104.08663)
- [ColBERT: Efficient and Effective Passage Search](https://arxiv.org/abs/2004.12832)
- [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906)

### Tools & Libraries
- [FAISS](https://github.com/facebookresearch/faiss): Vector similarity search
- [Sentence Transformers](https://www.sbert.net/): Embedding models
- [LlamaIndex](https://www.llamaindex.ai/): RAG framework
- [LangChain](https://www.langchain.com/): LLM application framework
- [Pinecone](https://www.pinecone.io/): Managed vector database
- [Weaviate](https://weaviate.io/): Open-source vector database

### Courses & Tutorials
- [Hugging Face: Sentence Transformers](https://huggingface.co/sentence-transformers)
- [LlamaIndex Guides](https://docs.llamaindex.ai/)
- [Building RAG Systems (DeepLearning.AI)](https://www.deeplearning.ai/short-courses/)

## 🤝 Contributing

This is a learning project for the AI Engineer Bootcamp. Feel free to:
- Experiment with different configurations
- Add new chunking strategies
- Implement additional reranking methods
- Improve monitoring capabilities

## 📝 License

MIT License - Feel free to use for learning and projects.

---

**Week 11-12 Project** | AI Engineer Bootcamp 2025 | Focus: Production RAG Systems
