"""
Production RAG Pipeline
------------------------
Complete RAG system integrating all components for <300ms latency.
"""

import uuid
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .chunking import get_chunker, Chunk
from .vector_store import FAISSVectorStore, Document
from .hybrid_search import HybridSearch, BM25
from .reranker import CrossEncoderReranker, MMRReranker
from .monitoring import (
    PerformanceTracker,
    LatencyMonitor,
    QueryLogger,
    AlertManager,
    QueryMetrics
)


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline."""

    # Chunking
    chunking_strategy: str = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Vector store
    embedding_dimension: int = 384
    index_type: str = "flat"  # "flat", "ivf", "hnsw"
    metric: str = "cosine"

    # Search
    use_hybrid_search: bool = True
    vector_weight: float = 0.7
    bm25_weight: float = 0.3
    use_rrf: bool = True

    # Retrieval
    top_k: int = 10
    search_k: int = 20  # Retrieve more, then rerank

    # Reranking
    use_reranking: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Monitoring
    enable_monitoring: bool = True
    enable_logging: bool = True
    latency_target_ms: float = 300


class ProductionRAG:
    """
    Production-ready RAG system with:
    - Hybrid search (vector + BM25)
    - Reranking for improved relevance
    - Comprehensive monitoring
    - Optimized for <300ms P95 latency
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        """Initialize RAG pipeline."""
        self.config = config or RAGConfig()

        # Initialize components
        self.chunker = get_chunker(
            self.config.chunking_strategy,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap
        )

        self.vector_store = FAISSVectorStore(
            dimension=self.config.embedding_dimension,
            index_type=self.config.index_type,
            metric=self.config.metric
        )

        self.bm25 = BM25()

        if self.config.use_hybrid_search:
            self.search_engine = HybridSearch(
                vector_store=self.vector_store,
                bm25=self.bm25,
                vector_weight=self.config.vector_weight,
                bm25_weight=self.config.bm25_weight,
                use_rrf=self.config.use_rrf
            )
        else:
            self.search_engine = self.vector_store

        if self.config.use_reranking:
            self.reranker = CrossEncoderReranker(
                model_name=self.config.rerank_model
            )
        else:
            self.reranker = None

        # Monitoring
        if self.config.enable_monitoring:
            self.tracker = PerformanceTracker()
            self.alert_manager = AlertManager(
                latency_threshold_ms=self.config.latency_target_ms
            )
        else:
            self.tracker = None
            self.alert_manager = None

        if self.config.enable_logging:
            self.logger = QueryLogger()
        else:
            self.logger = None

        # Embedding function (placeholder)
        self.embed_fn = self._default_embed

    def _default_embed(self, text: str) -> np.ndarray:
        """
        Default embedding function (placeholder).

        In production, use:
        - sentence-transformers
        - OpenAI embeddings
        - Custom models
        """
        # Simulate embedding with random vector
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.randn(self.config.embedding_dimension)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.astype(np.float32)

    def set_embedding_function(self, embed_fn):
        """Set custom embedding function."""
        self.embed_fn = embed_fn

    def ingest_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None):
        """
        Ingest documents into the RAG system.

        Args:
            texts: List of document texts
            metadatas: Optional metadata for each document
        """
        all_documents = []

        for i, text in enumerate(texts):
            doc_id = str(uuid.uuid4())
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}

            # Chunk document
            chunks = self.chunker.chunk(text)

            # Create document for each chunk
            for j, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{j}"

                # Embed chunk
                embedding = self.embed_fn(chunk.text)

                document = Document(
                    id=chunk_id,
                    text=chunk.text,
                    embedding=embedding,
                    metadata={
                        **metadata,
                        "doc_id": doc_id,
                        "chunk_idx": j,
                        "start_idx": chunk.start_idx,
                        "end_idx": chunk.end_idx
                    }
                )
                all_documents.append(document)

        # Add to search engine
        if self.config.use_hybrid_search:
            self.search_engine.add_documents(all_documents)
        else:
            self.vector_store.add_documents(all_documents)

        print(f"Ingested {len(texts)} documents ({len(all_documents)} chunks)")

    def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict] = None
    ) -> Tuple[List[Document], List[float], Optional[QueryMetrics]]:
        """
        Query the RAG system.

        Args:
            query_text: Query text
            top_k: Number of results to return
            filters: Optional metadata filters

        Returns:
            Tuple of (documents, scores, metrics)
        """
        top_k = top_k or self.config.top_k
        query_id = str(uuid.uuid4())

        # Initialize metrics
        if self.tracker:
            metrics = self.tracker.start_query(query_id, query_text)
        else:
            metrics = None

        try:
            # Embed query
            with LatencyMonitor() as embed_monitor:
                query_embedding = self.embed_fn(query_text)

            if metrics:
                metrics.embedding_latency = embed_monitor.latency_ms

            # Search
            with LatencyMonitor() as search_monitor:
                if self.config.use_hybrid_search:
                    results = self.search_engine.search(
                        query=query_text,
                        query_embedding=query_embedding,
                        k=self.config.search_k
                    )
                else:
                    results = self.vector_store.search(
                        query_embedding=query_embedding,
                        k=self.config.search_k
                    )

            if metrics:
                metrics.search_latency = search_monitor.latency_ms

            # Rerank
            if self.reranker and results:
                with LatencyMonitor() as rerank_monitor:
                    reranked = self.reranker.rerank(
                        query=query_text,
                        results=results,
                        top_k=top_k
                    )

                if metrics:
                    metrics.rerank_latency = rerank_monitor.latency_ms

                documents = [r.document for r in reranked]
                scores = [r.score for r in reranked]
            else:
                documents = [doc for doc, _ in results[:top_k]]
                scores = [score for _, score in results[:top_k]]

            # Update metrics
            if metrics:
                metrics.total_latency = (
                    metrics.embedding_latency +
                    metrics.search_latency +
                    metrics.rerank_latency
                )
                metrics.num_results = len(documents)
                metrics.top_score = scores[0] if scores else 0.0
                metrics.avg_score = sum(scores) / len(scores) if scores else 0.0

                self.tracker.record_metrics(metrics)

                if self.alert_manager:
                    self.alert_manager.record_query(
                        latency_ms=metrics.total_latency,
                        error=False
                    )

            # Log query
            if self.logger:
                self.logger.log_query(
                    query_id=query_id,
                    query_text=query_text,
                    results=[
                        {"text": doc.text[:100], "score": score}
                        for doc, score in zip(documents, scores)
                    ],
                    latency_ms=metrics.total_latency if metrics else 0.0
                )

            return documents, scores, metrics

        except Exception as e:
            if self.alert_manager:
                self.alert_manager.record_query(latency_ms=0.0, error=True)
            raise e

    def get_stats(self) -> Dict:
        """Get performance statistics."""
        if self.tracker:
            return self.tracker.get_stats()
        return {}

    def get_alerts(self) -> List[Dict]:
        """Get triggered alerts."""
        if self.alert_manager:
            return self.alert_manager.get_alerts()
        return []

    def save(self, path: str):
        """Save RAG system to disk."""
        self.vector_store.save(path)
        print(f"Saved RAG system to {path}")

    @classmethod
    def load(cls, path: str, config: Optional[RAGConfig] = None) -> "ProductionRAG":
        """Load RAG system from disk."""
        rag = cls(config=config)
        rag.vector_store = FAISSVectorStore.load(path)

        # Rebuild BM25 index
        documents = rag.vector_store.documents
        rag.bm25.add_documents(documents)

        # Rebuild hybrid search
        if rag.config.use_hybrid_search:
            rag.search_engine = HybridSearch(
                vector_store=rag.vector_store,
                bm25=rag.bm25,
                vector_weight=rag.config.vector_weight,
                bm25_weight=rag.config.bm25_weight,
                use_rrf=rag.config.use_rrf
            )

        print(f"Loaded RAG system from {path}")
        return rag
