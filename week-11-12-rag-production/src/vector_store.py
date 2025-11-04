"""
Vector Store Implementation with FAISS
---------------------------------------
Efficient vector similarity search using FAISS.
"""

import numpy as np
import faiss
import pickle
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    """Represents a document in the vector store."""
    id: str
    text: str
    embedding: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None


class FAISSVectorStore:
    """
    FAISS-based vector store for efficient similarity search.

    Supports multiple index types and persistent storage.
    """

    def __init__(
        self,
        dimension: int,
        index_type: str = "flat",
        metric: str = "cosine"
    ):
        """
        Initialize vector store.

        Args:
            dimension: Embedding dimension
            index_type: Type of FAISS index ("flat", "ivf", "hnsw")
            metric: Distance metric ("cosine", "l2")
        """
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric

        self.index = self._create_index()
        self.documents: List[Document] = []
        self.id_to_idx: Dict[str, int] = {}

    def _create_index(self) -> faiss.Index:
        """Create FAISS index based on type."""

        if self.index_type == "flat":
            # Flat index: exact search, slower but accurate
            if self.metric == "cosine":
                index = faiss.IndexFlatIP(self.dimension)  # Inner Product for cosine
            else:
                index = faiss.IndexFlatL2(self.dimension)

        elif self.index_type == "ivf":
            # IVF index: approximate search with clustering
            nlist = 100  # Number of clusters
            quantizer = faiss.IndexFlatL2(self.dimension)

            if self.metric == "cosine":
                index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            else:
                index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_L2)

        elif self.index_type == "hnsw":
            # HNSW: Hierarchical Navigable Small World graph
            # Fast approximate search
            M = 32  # Number of connections
            index = faiss.IndexHNSWFlat(self.dimension, M)

            if self.metric == "cosine":
                index.metric_type = faiss.METRIC_INNER_PRODUCT

        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        return index

    def add_documents(self, documents: List[Document]):
        """
        Add documents to the vector store.

        Args:
            documents: List of documents with embeddings
        """
        if not documents:
            return

        # Extract embeddings
        embeddings = np.array([doc.embedding for doc in documents], dtype=np.float32)

        # Normalize for cosine similarity
        if self.metric == "cosine":
            faiss.normalize_L2(embeddings)

        # Train index if needed (for IVF)
        if self.index_type == "ivf" and not self.index.is_trained:
            self.index.train(embeddings)

        # Add to index
        start_idx = len(self.documents)
        self.index.add(embeddings)

        # Update documents and mapping
        for i, doc in enumerate(documents):
            idx = start_idx + i
            self.documents.append(doc)
            self.id_to_idx[doc.id] = idx

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        filter_fn: Optional[callable] = None
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query vector
            k: Number of results to return
            filter_fn: Optional filter function for metadata filtering

        Returns:
            List of (document, score) tuples
        """
        # Normalize query for cosine similarity
        query = query_embedding.reshape(1, -1).astype(np.float32)
        if self.metric == "cosine":
            faiss.normalize_L2(query)

        # Search
        if self.index_type == "ivf":
            # Set number of clusters to search
            self.index.nprobe = 10

        # Get more results if filtering
        search_k = k * 3 if filter_fn else k
        distances, indices = self.index.search(query, search_k)

        # Build results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # No more results
                break

            doc = self.documents[idx]

            # Apply filter if provided
            if filter_fn and not filter_fn(doc):
                continue

            # Convert distance to score (higher is better)
            if self.metric == "cosine":
                score = float(dist)  # Already similarity score
            else:
                score = 1.0 / (1.0 + float(dist))  # Convert L2 distance

            results.append((doc, score))

            if len(results) >= k:
                break

        return results

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get document by ID."""
        idx = self.id_to_idx.get(doc_id)
        if idx is not None:
            return self.documents[idx]
        return None

    def delete_document(self, doc_id: str) -> bool:
        """
        Mark document as deleted (soft delete).

        Note: FAISS doesn't support true deletion,
        so we just mark it in our mapping.
        """
        if doc_id in self.id_to_idx:
            idx = self.id_to_idx[doc_id]
            del self.id_to_idx[doc_id]
            # Keep document in list for index consistency
            # but mark as deleted
            self.documents[idx].metadata = self.documents[idx].metadata or {}
            self.documents[idx].metadata['deleted'] = True
            return True
        return False

    def save(self, path: str):
        """Save index and documents to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(path / "index.faiss"))

        # Save documents and metadata
        with open(path / "documents.pkl", "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "id_to_idx": self.id_to_idx,
                "dimension": self.dimension,
                "index_type": self.index_type,
                "metric": self.metric
            }, f)

    @classmethod
    def load(cls, path: str) -> "FAISSVectorStore":
        """Load index and documents from disk."""
        path = Path(path)

        # Load metadata
        with open(path / "documents.pkl", "rb") as f:
            data = pickle.load(f)

        # Create instance
        store = cls(
            dimension=data["dimension"],
            index_type=data["index_type"],
            metric=data["metric"]
        )

        # Load FAISS index
        store.index = faiss.read_index(str(path / "index.faiss"))

        # Restore documents and mapping
        store.documents = data["documents"]
        store.id_to_idx = data["id_to_idx"]

        return store

    def __len__(self) -> int:
        """Return number of documents."""
        return len(self.documents)

    def get_stats(self) -> Dict:
        """Get vector store statistics."""
        return {
            "total_documents": len(self.documents),
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric": self.metric,
            "index_size_mb": self.index.ntotal * self.dimension * 4 / (1024 * 1024)
        }
