"""
Production RAG System
---------------------
High-performance RAG implementation with hybrid search, reranking, and monitoring.
"""

from .chunking import (
    Chunk,
    BaseChunker,
    FixedSizeChunker,
    SentenceChunker,
    SemanticChunker,
    RecursiveChunker,
    get_chunker
)

from .vector_store import (
    Document,
    FAISSVectorStore
)

from .hybrid_search import (
    BM25,
    HybridSearch
)

from .reranker import (
    RerankResult,
    CrossEncoderReranker,
    MMRReranker,
    EnsembleReranker
)

from .monitoring import (
    QueryMetrics,
    PerformanceTracker,
    LatencyMonitor,
    QueryLogger,
    AlertManager,
    MetricsDashboard
)

from .rag_pipeline import (
    RAGConfig,
    ProductionRAG
)

__version__ = "1.0.0"

__all__ = [
    # Chunking
    "Chunk",
    "BaseChunker",
    "FixedSizeChunker",
    "SentenceChunker",
    "SemanticChunker",
    "RecursiveChunker",
    "get_chunker",

    # Vector Store
    "Document",
    "FAISSVectorStore",

    # Search
    "BM25",
    "HybridSearch",

    # Reranking
    "RerankResult",
    "CrossEncoderReranker",
    "MMRReranker",
    "EnsembleReranker",

    # Monitoring
    "QueryMetrics",
    "PerformanceTracker",
    "LatencyMonitor",
    "QueryLogger",
    "AlertManager",
    "MetricsDashboard",

    # Pipeline
    "RAGConfig",
    "ProductionRAG",
]
