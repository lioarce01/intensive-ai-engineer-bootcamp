"""
Advanced RAG Usage Example
---------------------------
Shows advanced features like custom embeddings, reranking strategies, and monitoring.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_pipeline import ProductionRAG, RAGConfig
from monitoring import MetricsDashboard


def custom_embedding_function(text: str) -> np.ndarray:
    """
    Custom embedding function.

    In production, this would use sentence-transformers:

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding = model.encode(text)
    """
    # Simulate better embeddings based on text content
    hash_val = hash(text) % (2**32)
    np.random.seed(hash_val)

    # Create embedding that varies based on text characteristics
    length_factor = len(text) / 1000
    embedding = np.random.randn(384)

    # Add some structure based on keywords
    keywords = ["machine", "learning", "deep", "neural", "rag", "vector"]
    for i, keyword in enumerate(keywords):
        if keyword in text.lower():
            embedding[i*10:(i+1)*10] *= 2.0

    # Normalize
    embedding = embedding / np.linalg.norm(embedding)
    return embedding.astype(np.float32)


def main():
    """Advanced usage example."""

    # Create RAG with custom configuration
    print("Initializing advanced RAG system...")
    config = RAGConfig(
        # Chunking
        chunking_strategy="semantic",
        chunk_size=512,
        chunk_overlap=50,

        # Vector store
        embedding_dimension=384,
        index_type="flat",  # Try "hnsw" for larger datasets
        metric="cosine",

        # Search
        use_hybrid_search=True,
        vector_weight=0.7,
        bm25_weight=0.3,
        use_rrf=True,  # Reciprocal Rank Fusion

        # Retrieval
        top_k=5,
        search_k=20,  # Retrieve more, rerank to top_k

        # Reranking
        use_reranking=True,

        # Monitoring
        enable_monitoring=True,
        enable_logging=True,
        latency_target_ms=300
    )

    rag = ProductionRAG(config=config)

    # Set custom embedding function
    rag.set_embedding_function(custom_embedding_function)

    # Load more extensive documents
    documents = [
        """
        Transformer Architecture: The transformer is a neural network architecture
        introduced in the "Attention is All You Need" paper. It uses self-attention
        mechanisms to process sequences in parallel, unlike RNNs which process sequentially.
        Key components include multi-head attention, position encodings, and feed-forward layers.
        Transformers have become the foundation for models like BERT, GPT, and T5.
        """,
        """
        Fine-tuning LLMs: Fine-tuning adapts pre-trained language models to specific tasks.
        Modern techniques like LoRA (Low-Rank Adaptation) and QLoRA enable efficient fine-tuning
        by only updating a small subset of parameters. This reduces memory requirements and
        training time while maintaining model quality. PEFT (Parameter-Efficient Fine-Tuning)
        is a library that implements these techniques.
        """,
        """
        Vector Embeddings: Embeddings map text into high-dimensional vector spaces where
        semantic similarity corresponds to geometric proximity. Modern embedding models like
        sentence-transformers produce 384 or 768-dimensional vectors. These vectors enable
        efficient similarity search using techniques like cosine similarity or approximate
        nearest neighbor search with FAISS or Annoy.
        """,
        """
        RAG Systems: Retrieval-Augmented Generation combines the strengths of retrieval and
        generation. First, relevant documents are retrieved from a knowledge base using
        semantic search. Then, these documents are provided as context to a language model
        to generate accurate, grounded responses. This approach reduces hallucination and
        enables LLMs to access up-to-date information.
        """,
        """
        Chunking Strategies: Effective chunking is crucial for RAG performance. Fixed-size
        chunking is simple but may split sentences. Semantic chunking groups related content
        together. Recursive chunking respects document structure by splitting on paragraphs,
        then sentences, then words. The right strategy depends on your documents and use case.
        """,
        """
        Hybrid Search: Combining dense vector search with sparse keyword search (BM25)
        improves retrieval. Vector search excels at semantic similarity, while BM25 handles
        exact keyword matches. Methods like Reciprocal Rank Fusion (RRF) elegantly combine
        rankings from both approaches without needing score normalization.
        """,
        """
        Reranking: Initial retrieval casts a wide net to find candidate documents.
        Cross-encoder rerankers then precisely score query-document pairs. Unlike bi-encoders
        that embed queries and documents separately, cross-encoders jointly encode them,
        enabling better accuracy at the cost of higher latency. Use reranking for top results only.
        """,
        """
        MLOps for LLMs: Production ML systems require monitoring, versioning, and deployment
        infrastructure. Track latency, throughput, and quality metrics. Use A/B testing to
        evaluate changes. Implement caching and batching for efficiency. Monitor for model
        drift and degradation. Tools like MLflow, LangSmith, and Weights & Biases help manage
        the full lifecycle.
        """
    ]

    metadatas = [
        {"topic": "transformers", "difficulty": "intermediate"},
        {"topic": "fine-tuning", "difficulty": "advanced"},
        {"topic": "embeddings", "difficulty": "beginner"},
        {"topic": "rag", "difficulty": "intermediate"},
        {"topic": "chunking", "difficulty": "intermediate"},
        {"topic": "search", "difficulty": "advanced"},
        {"topic": "reranking", "difficulty": "advanced"},
        {"topic": "mlops", "difficulty": "advanced"}
    ]

    print(f"\nIngesting {len(documents)} documents...")
    rag.ingest_documents(documents, metadatas)

    # Run multiple queries to generate metrics
    queries = [
        "How do transformer models work?",
        "What is the best way to fine-tune LLMs?",
        "Explain vector embeddings for semantic search",
        "How does RAG reduce hallucination?",
        "What are different chunking strategies?",
        "Why combine vector search with keyword search?",
        "How does reranking improve search results?",
        "What metrics should I monitor for LLM systems?"
    ]

    print("\n" + "="*60)
    print("RUNNING QUERIES WITH DETAILED METRICS")
    print("="*60)

    for i, query in enumerate(queries, 1):
        print(f"\n[Query {i}] {query}")

        docs, scores, metrics = rag.query(query, top_k=3)

        print(f"  Latency: {metrics.total_latency:.2f}ms "
              f"(embed: {metrics.embedding_latency:.2f}ms, "
              f"search: {metrics.search_latency:.2f}ms, "
              f"rerank: {metrics.rerank_latency:.2f}ms)")

        print(f"  Top result: {docs[0].text[:100]}...")
        print(f"  Score: {scores[0]:.4f}")
        print(f"  Metadata: {docs[0].metadata}")

    # Display comprehensive metrics
    print("\n" + "="*60)
    print("COMPREHENSIVE PERFORMANCE METRICS")
    print("="*60)

    dashboard = MetricsDashboard(rag.tracker)
    dashboard.print_summary()

    # Check health
    health = dashboard.check_health(latency_target_ms=300)
    print(f"System Health: {health['status'].upper()}")
    print(f"P95 Latency: {health['p95_latency_ms']:.2f}ms")
    print(f"Meeting SLA (<300ms): {health['meeting_sla']}")

    # Check alerts
    alerts = rag.get_alerts()
    if alerts:
        print(f"\n⚠️  {len(alerts)} alerts triggered:")
        for alert in alerts:
            print(f"  - {alert['type']}: {alert['message']}")
    else:
        print("\n✅ No alerts triggered")

    # Export metrics
    print("\nExporting metrics...")
    rag.tracker.export_to_file("./metrics.json")

    # Save system
    print("Saving RAG system...")
    rag.save("./advanced_rag_index")

    print("\n✅ Advanced RAG example completed!")


if __name__ == "__main__":
    main()
