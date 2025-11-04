"""
Basic RAG Usage Example
------------------------
Simple example showing core RAG functionality.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_pipeline import ProductionRAG, RAGConfig


def main():
    """Basic usage example."""

    # Create RAG system with default config
    print("Initializing RAG system...")
    config = RAGConfig(
        chunking_strategy="recursive",
        chunk_size=512,
        embedding_dimension=384,
        use_hybrid_search=True,
        use_reranking=True,
        enable_monitoring=True
    )

    rag = ProductionRAG(config=config)

    # Sample documents
    documents = [
        """
        Machine learning is a subset of artificial intelligence that focuses on
        enabling computers to learn from data without being explicitly programmed.
        It uses algorithms to identify patterns and make decisions.
        """,
        """
        Deep learning is a type of machine learning based on artificial neural networks.
        It has multiple layers that progressively extract higher-level features from raw input.
        Deep learning has revolutionized computer vision and natural language processing.
        """,
        """
        Natural Language Processing (NLP) is a field of AI that focuses on the interaction
        between computers and human language. It includes tasks like translation,
        sentiment analysis, and text generation.
        """,
        """
        Retrieval-Augmented Generation (RAG) combines information retrieval with
        language generation. It retrieves relevant documents and uses them to generate
        more accurate and grounded responses.
        """,
        """
        Vector databases store high-dimensional vectors and enable efficient similarity search.
        They are essential for modern AI applications like semantic search and recommendation systems.
        """
    ]

    metadatas = [
        {"source": "ml_basics", "topic": "machine_learning"},
        {"source": "ml_basics", "topic": "deep_learning"},
        {"source": "nlp_guide", "topic": "nlp"},
        {"source": "rag_guide", "topic": "rag"},
        {"source": "vector_db_guide", "topic": "vector_databases"}
    ]

    # Ingest documents
    print("\nIngesting documents...")
    rag.ingest_documents(documents, metadatas)

    # Query
    print("\n" + "="*60)
    print("QUERY 1: What is deep learning?")
    print("="*60)

    docs, scores, metrics = rag.query("What is deep learning?", top_k=3)

    print(f"\nFound {len(docs)} results in {metrics.total_latency:.2f}ms")
    print(f"  - Embedding: {metrics.embedding_latency:.2f}ms")
    print(f"  - Search: {metrics.search_latency:.2f}ms")
    print(f"  - Rerank: {metrics.rerank_latency:.2f}ms")

    for i, (doc, score) in enumerate(zip(docs, scores)):
        print(f"\n[Result {i+1}] Score: {score:.4f}")
        print(f"Text: {doc.text[:200]}...")
        print(f"Metadata: {doc.metadata}")

    # Another query
    print("\n" + "="*60)
    print("QUERY 2: How does RAG work?")
    print("="*60)

    docs, scores, metrics = rag.query("How does RAG work?", top_k=3)

    print(f"\nFound {len(docs)} results in {metrics.total_latency:.2f}ms")

    for i, (doc, score) in enumerate(zip(docs, scores)):
        print(f"\n[Result {i+1}] Score: {score:.4f}")
        print(f"Text: {doc.text[:200]}...")

    # Get statistics
    print("\n" + "="*60)
    print("PERFORMANCE STATISTICS")
    print("="*60)

    stats = rag.get_stats()
    print(f"Total Queries: {stats['total_queries']}")
    print(f"Avg Latency: {stats['avg_total_latency_ms']:.2f}ms")
    print(f"P95 Latency: {stats['p95_latency_ms']:.2f}ms")
    print(f"Avg Results: {stats['avg_results']:.1f}")

    # Save system
    print("\nSaving RAG system...")
    rag.save("./rag_index")

    print("\nDone!")


if __name__ == "__main__":
    main()
