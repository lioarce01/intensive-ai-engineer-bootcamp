"""
Hybrid Search Implementation
-----------------------------
Combines vector similarity search with BM25 keyword search.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict, Counter
import math
from dataclasses import dataclass

from .vector_store import Document, FAISSVectorStore


class BM25:
    """
    BM25 (Best Matching 25) implementation for keyword search.

    BM25 is a ranking function used in information retrieval that
    considers term frequency, document length, and corpus statistics.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25.

        Args:
            k1: Controls term frequency saturation (typically 1.2-2.0)
            b: Controls document length normalization (typically 0.75)
        """
        self.k1 = k1
        self.b = b

        self.documents: List[Document] = []
        self.doc_freqs: Dict[str, int] = {}  # Term document frequency
        self.doc_lengths: List[int] = []
        self.avgdl: float = 0.0  # Average document length
        self.N: int = 0  # Number of documents

    def add_documents(self, documents: List[Document]):
        """Add documents to BM25 index."""
        for doc in documents:
            self.documents.append(doc)

            # Tokenize (simple whitespace tokenization)
            tokens = self._tokenize(doc.text)
            self.doc_lengths.append(len(tokens))

            # Update document frequencies
            for token in set(tokens):
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        self.N = len(self.documents)
        self.avgdl = sum(self.doc_lengths) / self.N if self.N > 0 else 0

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization (can be improved with proper NLP)."""
        return text.lower().split()

    def _idf(self, term: str) -> float:
        """Calculate IDF (Inverse Document Frequency)."""
        df = self.doc_freqs.get(term, 0)
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

    def _bm25_score(self, query_tokens: List[str], doc_idx: int) -> float:
        """Calculate BM25 score for a document."""
        doc = self.documents[doc_idx]
        doc_tokens = self._tokenize(doc.text)
        doc_len = self.doc_lengths[doc_idx]

        # Count term frequencies in document
        doc_term_freqs = Counter(doc_tokens)

        score = 0.0
        for term in query_tokens:
            if term not in doc_term_freqs:
                continue

            tf = doc_term_freqs[term]
            idf = self._idf(term)

            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)

            score += idf * (numerator / denominator)

        return score

    def search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """
        Search documents using BM25.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of (document, score) tuples
        """
        query_tokens = self._tokenize(query)

        # Score all documents
        scores = [
            (i, self._bm25_score(query_tokens, i))
            for i in range(len(self.documents))
        ]

        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)

        # Return top k
        results = [
            (self.documents[idx], score)
            for idx, score in scores[:k]
        ]

        return results


class HybridSearch:
    """
    Hybrid search combining vector similarity and BM25 keyword search.

    Uses Reciprocal Rank Fusion (RRF) to combine rankings from both methods.
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        bm25: BM25,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        use_rrf: bool = True,
        rrf_k: int = 60
    ):
        """
        Initialize hybrid search.

        Args:
            vector_store: FAISS vector store
            bm25: BM25 index
            vector_weight: Weight for vector search (if not using RRF)
            bm25_weight: Weight for BM25 search (if not using RRF)
            use_rrf: Use Reciprocal Rank Fusion instead of weighted scores
            rrf_k: RRF parameter (typically 60)
        """
        self.vector_store = vector_store
        self.bm25 = bm25
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.use_rrf = use_rrf
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        query_embedding: np.ndarray,
        k: int = 10,
        vector_k: int = 20,
        bm25_k: int = 20
    ) -> List[Tuple[Document, float]]:
        """
        Perform hybrid search.

        Args:
            query: Query text (for BM25)
            query_embedding: Query vector (for vector search)
            k: Number of final results
            vector_k: Number of results from vector search
            bm25_k: Number of results from BM25

        Returns:
            List of (document, score) tuples
        """
        # Get results from both methods
        vector_results = self.vector_store.search(query_embedding, k=vector_k)
        bm25_results = self.bm25.search(query, k=bm25_k)

        if self.use_rrf:
            # Reciprocal Rank Fusion
            combined = self._reciprocal_rank_fusion(vector_results, bm25_results)
        else:
            # Weighted combination
            combined = self._weighted_combination(vector_results, bm25_results)

        # Sort by final score
        combined_sorted = sorted(combined.items(), key=lambda x: x[1], reverse=True)

        # Return top k
        results = [
            (doc, score)
            for doc, score in combined_sorted[:k]
        ]

        return results

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Tuple[Document, float]],
        bm25_results: List[Tuple[Document, float]]
    ) -> Dict[Document, float]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).

        RRF score = sum(1 / (k + rank)) for each ranking

        RRF is robust and doesn't require score normalization.
        """
        scores = defaultdict(float)

        # Add vector search rankings
        for rank, (doc, _) in enumerate(vector_results, start=1):
            scores[doc] += 1.0 / (self.rrf_k + rank)

        # Add BM25 rankings
        for rank, (doc, _) in enumerate(bm25_results, start=1):
            scores[doc] += 1.0 / (self.rrf_k + rank)

        return dict(scores)

    def _weighted_combination(
        self,
        vector_results: List[Tuple[Document, float]],
        bm25_results: List[Tuple[Document, float]]
    ) -> Dict[Document, float]:
        """
        Combine results using weighted score combination.

        Requires score normalization since vector and BM25 scores
        are on different scales.
        """
        scores = defaultdict(float)

        # Normalize and combine vector scores
        if vector_results:
            max_vector = max(score for _, score in vector_results)
            min_vector = min(score for _, score in vector_results)
            range_vector = max_vector - min_vector or 1.0

            for doc, score in vector_results:
                normalized = (score - min_vector) / range_vector
                scores[doc] += self.vector_weight * normalized

        # Normalize and combine BM25 scores
        if bm25_results:
            max_bm25 = max(score for _, score in bm25_results)
            min_bm25 = min(score for _, score in bm25_results)
            range_bm25 = max_bm25 - min_bm25 or 1.0

            for doc, score in bm25_results:
                normalized = (score - min_bm25) / range_bm25
                scores[doc] += self.bm25_weight * normalized

        return dict(scores)

    def add_documents(self, documents: List[Document]):
        """Add documents to both indexes."""
        self.vector_store.add_documents(documents)
        self.bm25.add_documents(documents)
