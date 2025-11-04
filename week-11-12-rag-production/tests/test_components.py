"""
Unit Tests for RAG Components
------------------------------
Test each component independently.
"""

import sys
from pathlib import Path
import numpy as np
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chunking import (
    FixedSizeChunker,
    SentenceChunker,
    SemanticChunker,
    RecursiveChunker,
    get_chunker
)
from vector_store import FAISSVectorStore, Document
from hybrid_search import BM25, HybridSearch
from reranker import CrossEncoderReranker, MMRReranker
from monitoring import PerformanceTracker, LatencyMonitor
from rag_pipeline import ProductionRAG, RAGConfig


# Test Data
SAMPLE_TEXT = """
Machine learning is a subset of artificial intelligence.
It focuses on enabling computers to learn from data.

Deep learning uses neural networks with multiple layers.
It has revolutionized computer vision and NLP.

Natural language processing deals with text and speech.
It includes tasks like translation and sentiment analysis.
"""


def test_fixed_size_chunker():
    """Test fixed-size chunking."""
    chunker = FixedSizeChunker(chunk_size=50, overlap=10)
    chunks = chunker.chunk(SAMPLE_TEXT)

    assert len(chunks) > 0
    assert all(len(c.text) <= 50 for c in chunks)
    assert all(c.end_idx > c.start_idx for c in chunks)


def test_sentence_chunker():
    """Test sentence-based chunking."""
    chunker = SentenceChunker(target_size=100, tolerance=50)
    chunks = chunker.chunk(SAMPLE_TEXT)

    assert len(chunks) > 0
    # Chunks should respect sentence boundaries
    for chunk in chunks:
        assert '.' in chunk.text or chunk == chunks[-1]


def test_semantic_chunker():
    """Test semantic chunking."""
    chunker = SemanticChunker(min_chunk_size=50, max_chunk_size=200)
    chunks = chunker.chunk(SAMPLE_TEXT)

    assert len(chunks) > 0
    assert all(len(c.text) >= 50 or c == chunks[-1] for c in chunks)
    assert all(len(c.text) <= 200 for c in chunks)


def test_recursive_chunker():
    """Test recursive chunking."""
    chunker = RecursiveChunker(chunk_size=100, overlap=20)
    chunks = chunker.chunk(SAMPLE_TEXT)

    assert len(chunks) > 0
    assert all(c.end_idx > c.start_idx for c in chunks)


def test_chunker_factory():
    """Test get_chunker factory function."""
    strategies = ["fixed", "sentence", "semantic", "recursive"]

    for strategy in strategies:
        chunker = get_chunker(strategy)
        chunks = chunker.chunk(SAMPLE_TEXT)
        assert len(chunks) > 0


def test_vector_store():
    """Test FAISS vector store."""
    store = FAISSVectorStore(dimension=128, index_type="flat", metric="cosine")

    # Create test documents
    documents = []
    for i in range(10):
        embedding = np.random.randn(128).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        doc = Document(
            id=f"doc_{i}",
            text=f"Document {i} content",
            embedding=embedding,
            metadata={"index": i}
        )
        documents.append(doc)

    # Add documents
    store.add_documents(documents)
    assert len(store) == 10

    # Search
    query = np.random.randn(128).astype(np.float32)
    query = query / np.linalg.norm(query)

    results = store.search(query, k=5)
    assert len(results) == 5
    assert all(isinstance(doc, Document) for doc, _ in results)
    assert all(isinstance(score, float) for _, score in results)

    # Get document
    doc = store.get_document("doc_0")
    assert doc is not None
    assert doc.id == "doc_0"

    # Test save/load
    with tempfile.TemporaryDirectory() as tmpdir:
        store.save(tmpdir)
        loaded_store = FAISSVectorStore.load(tmpdir)

        assert len(loaded_store) == 10
        assert loaded_store.dimension == 128


def test_bm25():
    """Test BM25 keyword search."""
    bm25 = BM25()

    # Create test documents
    documents = [
        Document(id="1", text="machine learning and artificial intelligence"),
        Document(id="2", text="deep learning neural networks"),
        Document(id="3", text="natural language processing and text analysis")
    ]

    bm25.add_documents(documents)

    # Search
    results = bm25.search("machine learning", k=2)
    assert len(results) <= 2
    assert results[0][0].id == "1"  # Should rank doc 1 highest


def test_hybrid_search():
    """Test hybrid search."""
    # Setup
    vector_store = FAISSVectorStore(dimension=128, index_type="flat")
    bm25 = BM25()

    hybrid = HybridSearch(
        vector_store=vector_store,
        bm25=bm25,
        use_rrf=True
    )

    # Create documents with embeddings
    documents = []
    for i in range(5):
        embedding = np.random.randn(128).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        doc = Document(
            id=f"doc_{i}",
            text=f"machine learning document {i}",
            embedding=embedding
        )
        documents.append(doc)

    hybrid.add_documents(documents)

    # Search
    query_embedding = np.random.randn(128).astype(np.float32)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    results = hybrid.search(
        query="machine learning",
        query_embedding=query_embedding,
        k=3
    )

    assert len(results) <= 3


def test_cross_encoder_reranker():
    """Test cross-encoder reranking."""
    reranker = CrossEncoderReranker()

    # Create test results
    documents = [
        Document(id="1", text="machine learning is a subset of AI"),
        Document(id="2", text="deep learning uses neural networks"),
        Document(id="3", text="NLP processes natural language")
    ]

    results = [(doc, 0.5) for doc in documents]

    # Rerank
    reranked = reranker.rerank("what is machine learning", results, top_k=2)

    assert len(reranked) == 2
    assert all(hasattr(r, 'score') for r in reranked)
    assert all(hasattr(r, 'original_rank') for r in reranked)


def test_mmr_reranker():
    """Test MMR reranking for diversity."""
    reranker = MMRReranker(lambda_param=0.7)

    # Create documents with embeddings
    documents = []
    for i in range(5):
        embedding = np.random.randn(128).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        doc = Document(
            id=f"doc_{i}",
            text=f"document {i}",
            embedding=embedding
        )
        documents.append(doc)

    results = [(doc, 0.8 - i * 0.1) for i, doc in enumerate(documents)]

    # Rerank
    query_embedding = np.random.randn(128).astype(np.float32)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    reranked = reranker.rerank(query_embedding, results, top_k=3)

    assert len(reranked) == 3


def test_performance_tracker():
    """Test performance tracking."""
    tracker = PerformanceTracker()

    # Record some metrics
    for i in range(10):
        metrics = tracker.start_query(f"query_{i}", f"test query {i}")
        metrics.total_latency = 100 + i * 10
        metrics.search_latency = 50 + i * 5
        metrics.num_results = 5
        tracker.record_metrics(metrics)

    # Get stats
    stats = tracker.get_stats()
    assert stats['total_queries'] == 10
    assert 'avg_total_latency_ms' in stats
    assert 'p95_latency_ms' in stats


def test_latency_monitor():
    """Test latency monitoring context manager."""
    import time

    with LatencyMonitor() as monitor:
        time.sleep(0.01)  # 10ms

    assert monitor.latency_ms >= 10
    assert monitor.latency_ms < 20  # Should be close to 10ms


def test_production_rag():
    """Test complete RAG pipeline."""
    config = RAGConfig(
        chunking_strategy="fixed",
        chunk_size=100,
        embedding_dimension=128,
        use_hybrid_search=True,
        use_reranking=False,  # Skip for speed
        enable_monitoring=True
    )

    rag = ProductionRAG(config=config)

    # Ingest documents
    documents = [
        "Machine learning is a subset of artificial intelligence that focuses on data.",
        "Deep learning uses neural networks with multiple layers for complex tasks.",
        "Natural language processing enables computers to understand human language."
    ]

    rag.ingest_documents(documents)

    # Query
    docs, scores, metrics = rag.query("What is machine learning?", top_k=2)

    assert len(docs) <= 2
    assert len(scores) == len(docs)
    assert metrics is not None
    assert metrics.total_latency > 0

    # Get stats
    stats = rag.get_stats()
    assert stats['total_queries'] == 1

    # Test save/load
    with tempfile.TemporaryDirectory() as tmpdir:
        rag.save(tmpdir)
        loaded_rag = ProductionRAG.load(tmpdir, config=config)

        # Query loaded system
        docs2, scores2, _ = loaded_rag.query("What is deep learning?", top_k=2)
        assert len(docs2) <= 2


def run_all_tests():
    """Run all tests manually."""
    print("Running RAG Component Tests...\n")

    tests = [
        ("Fixed Size Chunker", test_fixed_size_chunker),
        ("Sentence Chunker", test_sentence_chunker),
        ("Semantic Chunker", test_semantic_chunker),
        ("Recursive Chunker", test_recursive_chunker),
        ("Chunker Factory", test_chunker_factory),
        ("Vector Store", test_vector_store),
        ("BM25", test_bm25),
        ("Hybrid Search", test_hybrid_search),
        ("Cross-Encoder Reranker", test_cross_encoder_reranker),
        ("MMR Reranker", test_mmr_reranker),
        ("Performance Tracker", test_performance_tracker),
        ("Latency Monitor", test_latency_monitor),
        ("Production RAG", test_production_rag),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            print(f"✅ {name}")
            passed += 1
        except Exception as e:
            print(f"❌ {name}: {str(e)}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_all_tests()
