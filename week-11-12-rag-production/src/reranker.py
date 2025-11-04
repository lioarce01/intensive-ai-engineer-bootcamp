"""
Reranking Module
----------------
Rerank search results using cross-encoder models for improved relevance.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .vector_store import Document


@dataclass
class RerankResult:
    """Result from reranking."""
    document: Document
    score: float
    original_score: float
    original_rank: int


class CrossEncoderReranker:
    """
    Cross-encoder based reranking.

    Cross-encoders process query-document pairs jointly,
    providing more accurate relevance scores than bi-encoders.

    In production, you would use models like:
    - cross-encoder/ms-marco-MiniLM-L-6-v2
    - cross-encoder/ms-marco-electra-base
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32,
        max_length: int = 512
    ):
        """
        Initialize cross-encoder reranker.

        Args:
            model_name: Hugging Face model name
            batch_size: Batch size for inference
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.model = None

    def load_model(self):
        """
        Load cross-encoder model.

        This is a placeholder - in production you would:
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(self.model_name, max_length=self.max_length)
        """
        # Placeholder for demonstration
        self.model = "cross_encoder_model"
        print(f"[Reranker] Loaded model: {self.model_name}")

    def rerank(
        self,
        query: str,
        results: List[Tuple[Document, float]],
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """
        Rerank search results.

        Args:
            query: Query text
            results: Initial search results (document, score) tuples
            top_k: Number of top results to return (None = all)

        Returns:
            Reranked results
        """
        if not self.model:
            self.load_model()

        if not results:
            return []

        # In production, you would do:
        # pairs = [(query, doc.text) for doc, _ in results]
        # scores = self.model.predict(pairs, batch_size=self.batch_size)

        # For demonstration, simulate cross-encoder scores
        reranked = []
        for i, (doc, original_score) in enumerate(results):
            # Simulate cross-encoder score based on query-document relevance
            simulated_score = self._simulate_cross_encoder_score(query, doc.text)

            reranked.append(RerankResult(
                document=doc,
                score=simulated_score,
                original_score=original_score,
                original_rank=i + 1
            ))

        # Sort by new scores
        reranked.sort(key=lambda x: x.score, reverse=True)

        if top_k:
            reranked = reranked[:top_k]

        return reranked

    def _simulate_cross_encoder_score(self, query: str, doc_text: str) -> float:
        """
        Simulate cross-encoder scoring.

        In production, the actual model would score the query-document pair.
        Here we use a simple heuristic for demonstration.
        """
        query_lower = query.lower()
        doc_lower = doc_text.lower()

        # Simple scoring based on term overlap and position
        query_terms = set(query_lower.split())
        doc_terms = doc_lower.split()

        # Term overlap score
        overlap = len(query_terms & set(doc_terms))
        overlap_score = overlap / len(query_terms) if query_terms else 0

        # Early position bonus (terms appearing early are more important)
        position_score = 0
        for term in query_terms:
            if term in doc_lower:
                pos = doc_lower.index(term)
                # Earlier positions get higher scores
                position_score += 1.0 / (1.0 + pos / 100)

        position_score /= len(query_terms) if query_terms else 1

        # Combine scores
        final_score = 0.6 * overlap_score + 0.4 * position_score

        # Add some noise for variety
        noise = np.random.normal(0, 0.05)
        final_score = max(0, min(1, final_score + noise))

        return final_score


class MMRReranker:
    """
    Maximal Marginal Relevance (MMR) reranking.

    MMR balances relevance and diversity to avoid redundant results.

    MMR = λ * relevance - (1-λ) * max_similarity_to_selected
    """

    def __init__(self, lambda_param: float = 0.7):
        """
        Initialize MMR reranker.

        Args:
            lambda_param: Balance between relevance (1.0) and diversity (0.0)
        """
        self.lambda_param = lambda_param

    def rerank(
        self,
        query_embedding: np.ndarray,
        results: List[Tuple[Document, float]],
        top_k: int = 10
    ) -> List[RerankResult]:
        """
        Rerank using MMR for diversity.

        Args:
            query_embedding: Query vector
            results: Initial search results
            top_k: Number of results to return

        Returns:
            Diverse set of results
        """
        if not results:
            return []

        selected = []
        remaining = list(results)

        while len(selected) < top_k and remaining:
            if not selected:
                # First selection: most relevant
                best_idx = 0
            else:
                # Find document with best MMR score
                best_idx = self._find_best_mmr(
                    query_embedding,
                    remaining,
                    selected
                )

            doc, score = remaining.pop(best_idx)
            selected.append(RerankResult(
                document=doc,
                score=score,
                original_score=score,
                original_rank=len(selected)
            ))

        return selected

    def _find_best_mmr(
        self,
        query_embedding: np.ndarray,
        remaining: List[Tuple[Document, float]],
        selected: List[RerankResult]
    ) -> int:
        """Find document with best MMR score."""
        best_score = float('-inf')
        best_idx = 0

        for i, (doc, relevance) in enumerate(remaining):
            # Calculate max similarity to already selected documents
            max_sim = 0.0
            if selected and doc.embedding is not None:
                for result in selected:
                    if result.document.embedding is not None:
                        sim = self._cosine_similarity(
                            doc.embedding,
                            result.document.embedding
                        )
                        max_sim = max(max_sim, sim)

            # MMR score
            mmr_score = (self.lambda_param * relevance -
                        (1 - self.lambda_param) * max_sim)

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        return best_idx

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class EnsembleReranker:
    """
    Ensemble reranker combining multiple reranking strategies.
    """

    def __init__(self, rerankers: List[Tuple[str, object, float]]):
        """
        Initialize ensemble reranker.

        Args:
            rerankers: List of (name, reranker, weight) tuples
        """
        self.rerankers = rerankers

    def rerank(
        self,
        query: str,
        query_embedding: np.ndarray,
        results: List[Tuple[Document, float]],
        top_k: int = 10
    ) -> List[RerankResult]:
        """
        Rerank using ensemble of methods.

        Args:
            query: Query text
            query_embedding: Query vector
            results: Initial results
            top_k: Number of results

        Returns:
            Ensemble-reranked results
        """
        # Collect scores from each reranker
        all_scores = {}

        for name, reranker, weight in self.rerankers:
            # Get reranked results from this reranker
            if isinstance(reranker, CrossEncoderReranker):
                reranked = reranker.rerank(query, results)
            elif isinstance(reranker, MMRReranker):
                reranked = reranker.rerank(query_embedding, results, top_k=len(results))
            else:
                continue

            # Accumulate weighted scores
            for result in reranked:
                doc_id = result.document.id
                if doc_id not in all_scores:
                    all_scores[doc_id] = {
                        'document': result.document,
                        'score': 0.0,
                        'original_score': result.original_score,
                        'original_rank': result.original_rank
                    }
                all_scores[doc_id]['score'] += weight * result.score

        # Create final results
        final_results = [
            RerankResult(
                document=info['document'],
                score=info['score'],
                original_score=info['original_score'],
                original_rank=info['original_rank']
            )
            for info in all_scores.values()
        ]

        # Sort by ensemble score
        final_results.sort(key=lambda x: x.score, reverse=True)

        return final_results[:top_k]
