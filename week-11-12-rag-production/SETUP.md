# Setup Guide - Week 11-12: Production RAG System

## Prerequisites

- Python 3.8+
- pip package manager
- (Optional) GPU for faster embeddings

## Installation Steps

### 1. Install Dependencies

```bash
# Navigate to project directory
cd week-11-12-rag-production

# Install required packages
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
# Run tests to verify everything works
python tests/test_components.py
```

You should see output like:
```
Running RAG Component Tests...

✅ Fixed Size Chunker
✅ Sentence Chunker
✅ Semantic Chunker
...

Results: 13 passed, 0 failed
```

### 3. Run Basic Example

```bash
# Test the basic RAG pipeline
python examples/basic_usage.py
```

Expected output:
```
Initializing RAG system...
Ingesting documents...
Ingested 5 documents (5 chunks)

=============================================================
QUERY 1: What is deep learning?
=============================================================

Found 3 results in 125.45ms
  - Embedding: 15.23ms
  - Search: 45.12ms
  - Rerank: 65.10ms

[Result 1] Score: 0.8234
...
```

### 4. Run Advanced Example

```bash
# Test advanced features
python examples/advanced_usage.py
```

This demonstrates:
- Custom embedding functions
- Multiple query types
- Performance monitoring
- Metrics dashboard

## Optional: Production Setup

### Use Real Embeddings

For production, install sentence-transformers:

```bash
pip install sentence-transformers torch
```

Then use in your code:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
rag.set_embedding_function(lambda text: model.encode(text))
```

### Use GPU Acceleration

Install FAISS with GPU support:

```bash
# Remove CPU version
pip uninstall faiss-cpu

# Install GPU version
pip install faiss-gpu
```

### Install Production Monitoring

```bash
pip install mlflow wandb
```

## Troubleshooting

### Issue: ImportError for FAISS

**Solution**: Make sure faiss-cpu is installed:
```bash
pip install faiss-cpu==1.7.4
```

### Issue: Slow Performance

**Solutions**:
1. Use HNSW index for large datasets
2. Enable GPU if available
3. Reduce chunk_size or search_k
4. Cache embeddings

### Issue: Memory Errors

**Solutions**:
1. Reduce batch size
2. Use IVF index instead of flat
3. Process documents in batches
4. Increase system memory

## Next Steps

1. ✅ Run all examples successfully
2. 📖 Read the README.md for detailed docs
3. 🧪 Experiment with different configurations
4. 🚀 Build your own RAG application
5. 📊 Monitor performance with the dashboard

## Project Structure

```
week-11-12-rag-production/
├── src/                    # Core implementation
│   ├── chunking.py        # 4 chunking strategies
│   ├── vector_store.py    # FAISS integration
│   ├── hybrid_search.py   # Vector + BM25 search
│   ├── reranker.py       # Reranking algorithms
│   ├── monitoring.py      # Performance tracking
│   └── rag_pipeline.py    # Complete pipeline
├── examples/              # Usage examples
├── tests/                 # Unit tests
├── README.md             # Full documentation
└── requirements.txt       # Dependencies
```

## Learning Goals

After completing this week, you should understand:

- ✅ How to implement production-grade RAG systems
- ✅ Different chunking strategies and when to use them
- ✅ Hybrid search (vector + keyword) advantages
- ✅ Why and how to use reranking
- ✅ Performance monitoring and optimization
- ✅ Achieving <300ms P95 latency targets

## Resources

- [FAISS Documentation](https://github.com/facebookresearch/faiss/wiki)
- [Sentence Transformers](https://www.sbert.net/)
- [LlamaIndex RAG Guide](https://docs.llamaindex.ai/)
- [Pinecone: RAG Best Practices](https://www.pinecone.io/learn/rag/)

## Support

For questions or issues:
1. Check the README.md documentation
2. Review example code in examples/
3. Run tests to verify setup
4. Check Troubleshooting section above

---

**Ready to build production RAG systems!** 🚀
