#!/usr/bin/env python3
"""
FAISS Vector Store for Semantic Search
======================================

This module provides FAISS-based vector storage and retrieval functionality
for the semantic search engine. It supports multiple embedding types,
efficient similarity search, and persistent storage.

Features:
- FAISS IndexFlatIP for exact cosine similarity search
- Support for multiple embedding models (Word2Vec, Sentence Transformers)
- Batch indexing of embeddings
- Incremental document additions
- Persistent storage and loading
- Efficient similarity search with metadata

Author: AI Bootcamp Week 3-4
Date: 2025
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
import time
from datetime import datetime

# Core libraries
import numpy as np

# Vector search
try:
    import faiss
    FAISS_AVAILABLE = True
    print("FAISS library loaded successfully")
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available. Install with: pip install faiss-cpu")

# Import components
try:
    from document_store import DocumentStore, StoredChunk
    from embedding_engine import EnhancedEmbeddingEngine, EmbeddingResult
    COMPONENTS_AVAILABLE = True
except ImportError:
    COMPONENTS_AVAILABLE = False
    print("Warning: Could not import local components. Some features will be limited.")


@dataclass
class SearchResult:
    """Result from vector search."""
    chunk_id: str
    document_id: str
    content: str
    score: float
    model_name: str
    rank: int
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class IndexMetadata:
    """Metadata about a vector index."""
    model_name: str
    embedding_dim: int
    num_vectors: int
    index_type: str
    created_at: str
    updated_at: str
    file_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IndexMetadata':
        """Create from dictionary."""
        return cls(**data)


class FAISSVectorStore:
    """
    FAISS-based vector store for efficient similarity search.
    
    This class manages FAISS indices for different embedding models,
    supports batch operations, and provides persistent storage.
    """
    
    def __init__(self, storage_dir: str = "vector_indices"):
        """
        Initialize FAISS vector store.
        
        Args:
            storage_dir: Directory to store FAISS indices and metadata
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Store indices and metadata
        self.indices: Dict[str, faiss.Index] = {}
        self.metadata: Dict[str, IndexMetadata] = {}
        self.chunk_mappings: Dict[str, List[str]] = {}  # model_name -> list of chunk_ids
        
        # Load existing indices
        self._load_existing_indices()
        
        if not FAISS_AVAILABLE:
            print("Warning: FAISS not available. Vector store will have limited functionality.")
    
    def _load_existing_indices(self):
        """Load existing FAISS indices from disk."""
        if not FAISS_AVAILABLE:
            return
        
        metadata_file = self.storage_dir / "metadata.json"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata_dict = json.load(f)
                
                for model_name, meta_data in metadata_dict.items():
                    # Load metadata
                    metadata = IndexMetadata.from_dict(meta_data)
                    self.metadata[model_name] = metadata
                    
                    # Load FAISS index
                    index_file = self.storage_dir / f"{model_name}_index.faiss"
                    if index_file.exists():
                        index = faiss.read_index(str(index_file))
                        self.indices[model_name] = index
                        
                        # Load chunk mappings
                        mapping_file = self.storage_dir / f"{model_name}_mappings.json"
                        if mapping_file.exists():
                            with open(mapping_file, 'r') as f:
                                self.chunk_mappings[model_name] = json.load(f)
                        
                        print(f"Loaded FAISS index for {model_name}: {metadata.num_vectors} vectors")
                
            except Exception as e:
                print(f"Error loading existing indices: {e}")
    
    def create_index(self, model_name: str, embedding_dim: int, 
                    index_type: str = "IndexFlatIP") -> bool:
        """
        Create a new FAISS index for a specific embedding model.
        
        Args:
            model_name: Name of the embedding model
            embedding_dim: Dimensionality of embeddings
            index_type: Type of FAISS index (default: IndexFlatIP for cosine similarity)
        
        Returns:
            True if index created successfully, False otherwise
        """
        if not FAISS_AVAILABLE:
            print("FAISS not available. Cannot create index.")
            return False
        
        try:
            if index_type == "IndexFlatIP":
                # IndexFlatIP is good for cosine similarity (after normalization)
                index = faiss.IndexFlatIP(embedding_dim)
            elif index_type == "IndexFlatL2":
                # IndexFlatL2 for L2 distance
                index = faiss.IndexFlatL2(embedding_dim)
            else:
                raise ValueError(f"Unsupported index type: {index_type}")
            
            self.indices[model_name] = index
            
            # Create metadata
            metadata = IndexMetadata(
                model_name=model_name,
                embedding_dim=embedding_dim,
                num_vectors=0,
                index_type=index_type,
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat()
            )
            self.metadata[model_name] = metadata
            
            # Initialize chunk mapping
            self.chunk_mappings[model_name] = []
            
            print(f"Created FAISS index for {model_name} ({embedding_dim}D, {index_type})")
            return True
            
        except Exception as e:
            print(f"Error creating FAISS index for {model_name}: {e}")
            return False
    
    def add_embeddings(self, model_name: str, embeddings: List[np.ndarray], 
                      chunk_ids: List[str]) -> bool:
        """
        Add embeddings to the FAISS index.
        
        Args:
            model_name: Name of the embedding model
            embeddings: List of embedding vectors
            chunk_ids: List of corresponding chunk IDs
        
        Returns:
            True if embeddings added successfully, False otherwise
        """
        if not FAISS_AVAILABLE:
            print("FAISS not available. Cannot add embeddings.")
            return False
        
        if model_name not in self.indices:
            print(f"No index found for model {model_name}. Create index first.")
            return False
        
        if len(embeddings) != len(chunk_ids):
            print(f"Mismatch between embeddings ({len(embeddings)}) and chunk_ids ({len(chunk_ids)})")
            return False
        
        try:
            index = self.indices[model_name]
            
            # Convert embeddings to numpy array
            embedding_matrix = np.array(embeddings, dtype=np.float32)
            
            # Normalize embeddings for cosine similarity (if using IndexFlatIP)
            if self.metadata[model_name].index_type == "IndexFlatIP":
                # Normalize each embedding vector
                norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
                # Avoid division by zero
                norms = np.where(norms == 0, 1, norms)
                embedding_matrix = embedding_matrix / norms
            
            # Add to FAISS index
            index.add(embedding_matrix)
            
            # Update chunk mappings
            self.chunk_mappings[model_name].extend(chunk_ids)
            
            # Update metadata
            self.metadata[model_name].num_vectors = len(self.chunk_mappings[model_name])
            self.metadata[model_name].updated_at = datetime.now().isoformat()
            
            print(f"Added {len(embeddings)} embeddings to {model_name} index (total: {self.metadata[model_name].num_vectors})")
            return True
            
        except Exception as e:
            print(f"Error adding embeddings to {model_name}: {e}")
            return False
    
    def search(self, model_name: str, query_embedding: np.ndarray, 
               k: int = 10, threshold: float = 0.0) -> List[int]:
        """
        Search for similar vectors in the FAISS index.
        
        Args:
            model_name: Name of the embedding model
            query_embedding: Query embedding vector
            k: Number of top results to return
            threshold: Minimum similarity threshold (for filtering)
        
        Returns:
            List of indices of similar vectors, sorted by similarity
        """
        if not FAISS_AVAILABLE:
            print("FAISS not available. Cannot perform search.")
            return []
        
        if model_name not in self.indices:
            print(f"No index found for model {model_name}")
            return []
        
        try:
            index = self.indices[model_name]
            
            # Normalize query embedding for cosine similarity (if using IndexFlatIP)
            query = query_embedding.astype(np.float32).reshape(1, -1)
            if self.metadata[model_name].index_type == "IndexFlatIP":
                norm = np.linalg.norm(query)
                if norm != 0:
                    query = query / norm
            
            # Search FAISS index
            scores, indices = index.search(query, k)
            
            # Filter by threshold if specified
            valid_results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx != -1 and score >= threshold:  # -1 indicates no result found
                    valid_results.append((int(idx), float(score)))
            
            # Return sorted indices (FAISS already returns them sorted by score)
            return [(idx, score) for idx, score in valid_results]
            
        except Exception as e:
            print(f"Error searching {model_name} index: {e}")
            return []
    
    def search_with_metadata(self, model_name: str, query_embedding: np.ndarray,
                           document_store: 'DocumentStore', k: int = 10,
                           threshold: float = 0.0) -> List[SearchResult]:
        """
        Search with full metadata from document store.
        
        Args:
            model_name: Name of the embedding model
            query_embedding: Query embedding vector
            document_store: Document store to get chunk metadata
            k: Number of top results to return
            threshold: Minimum similarity threshold
        
        Returns:
            List of SearchResult objects with full metadata
        """
        search_results = self.search(model_name, query_embedding, k, threshold)
        
        results = []
        for rank, (idx, score) in enumerate(search_results):
            try:
                # Get chunk ID from mapping
                chunk_id = self.chunk_mappings[model_name][idx]
                
                # Get chunk details from document store
                chunk = document_store.get_chunk(chunk_id)
                
                if chunk:
                    # Convert StoredChunk object to dict if needed
                    if hasattr(chunk, 'to_dict'):
                        chunk_dict = chunk.to_dict()
                    else:
                        chunk_dict = chunk
                    
                    result = SearchResult(
                        chunk_id=chunk_id,
                        document_id=chunk_dict['document_id'],
                        content=chunk_dict['content'],
                        score=score,
                        model_name=model_name,
                        rank=rank + 1,
                        metadata=chunk_dict.get('metadata', {})
                    )
                    results.append(result)
                
            except (IndexError, KeyError) as e:
                print(f"Error retrieving metadata for result {rank}: {e}")
                continue
        
        return results
    
    def build_index_from_document_store(self, document_store: 'DocumentStore',
                                      embedding_engine: 'EnhancedEmbeddingEngine',
                                      model_names: Optional[List[str]] = None,
                                      batch_size: int = 32) -> Dict[str, bool]:
        """
        Build FAISS indices from existing embeddings in document store.
        
        Args:
            document_store: Document store with embeddings
            embedding_engine: Embedding engine for getting model info
            model_names: List of models to build indices for (None = all available)
            batch_size: Batch size for processing
        
        Returns:
            Dictionary indicating success/failure for each model
        """
        if not FAISS_AVAILABLE:
            print("FAISS not available. Cannot build indices.")
            return {}
        
        results = {}
        
        # Get all chunks with embeddings
        chunks_with_embeddings = []
        for chunk in document_store.get_all_chunks():
            if chunk.get('has_embedding', False):
                chunks_with_embeddings.append(chunk)
        
        print(f"Found {len(chunks_with_embeddings)} chunks with embeddings")
        
        if not chunks_with_embeddings:
            print("No chunks with embeddings found. Cannot build indices.")
            return results
        
        # Determine which models to process
        available_models = set()
        for chunk in chunks_with_embeddings:
            available_models.update(chunk.get('embedding_models', []))
        
        if model_names is None:
            model_names = list(available_models)
        else:
            model_names = [m for m in model_names if m in available_models]
        
        print(f"Building indices for models: {model_names}")
        
        # Build index for each model
        for model_name in model_names:
            try:
                print(f"\nBuilding index for {model_name}...")
                
                # Collect embeddings and chunk IDs for this model
                embeddings = []
                chunk_ids = []
                
                for chunk in chunks_with_embeddings:
                    if model_name in chunk.get('embedding_models', []):
                        # Load embedding from document store
                        embedding = document_store.get_chunk_embedding(chunk['chunk_id'], model_name)
                        if embedding is not None:
                            embeddings.append(embedding)
                            chunk_ids.append(chunk['chunk_id'])
                
                if not embeddings:
                    print(f"No embeddings found for {model_name}")
                    results[model_name] = False
                    continue
                
                # Get embedding dimension
                embedding_dim = len(embeddings[0])
                
                # Create index
                success = self.create_index(model_name, embedding_dim)
                if not success:
                    results[model_name] = False
                    continue
                
                # Add embeddings in batches
                for i in range(0, len(embeddings), batch_size):
                    batch_embeddings = embeddings[i:i + batch_size]
                    batch_chunk_ids = chunk_ids[i:i + batch_size]
                    
                    success = self.add_embeddings(model_name, batch_embeddings, batch_chunk_ids)
                    if not success:
                        results[model_name] = False
                        break
                
                if success:
                    results[model_name] = True
                    print(f"Successfully built {model_name} index with {len(embeddings)} vectors")
                
            except Exception as e:
                print(f"Error building index for {model_name}: {e}")
                results[model_name] = False
        
        # Save indices to disk
        if any(results.values()):
            self.save_indices()
        
        return results
    
    def save_indices(self):
        """Save all FAISS indices and metadata to disk."""
        if not FAISS_AVAILABLE:
            return
        
        try:
            # Save metadata
            metadata_dict = {}
            for model_name, metadata in self.metadata.items():
                metadata_dict[model_name] = metadata.to_dict()
            
            metadata_file = self.storage_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata_dict, f, indent=2)
            
            # Save each FAISS index and chunk mappings
            for model_name, index in self.indices.items():
                # Save FAISS index
                index_file = self.storage_dir / f"{model_name}_index.faiss"
                faiss.write_index(index, str(index_file))
                
                # Save chunk mappings
                mapping_file = self.storage_dir / f"{model_name}_mappings.json"
                with open(mapping_file, 'w') as f:
                    json.dump(self.chunk_mappings[model_name], f, indent=2)
            
            print(f"Saved {len(self.indices)} FAISS indices to {self.storage_dir}")
            
        except Exception as e:
            print(f"Error saving indices: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        stats = {
            'num_indices': len(self.indices),
            'total_vectors': sum(meta.num_vectors for meta in self.metadata.values()),
            'storage_dir': str(self.storage_dir),
            'faiss_available': FAISS_AVAILABLE,
            'indices': {}
        }
        
        for model_name, metadata in self.metadata.items():
            stats['indices'][model_name] = {
                'embedding_dim': metadata.embedding_dim,
                'num_vectors': metadata.num_vectors,
                'index_type': metadata.index_type,
                'created_at': metadata.created_at,
                'updated_at': metadata.updated_at
            }
        
        return stats
    
    def remove_index(self, model_name: str) -> bool:
        """
        Remove a FAISS index and its associated files.
        
        Args:
            model_name: Name of the model index to remove
        
        Returns:
            True if removed successfully, False otherwise
        """
        try:
            # Remove from memory
            if model_name in self.indices:
                del self.indices[model_name]
            if model_name in self.metadata:
                del self.metadata[model_name]
            if model_name in self.chunk_mappings:
                del self.chunk_mappings[model_name]
            
            # Remove files
            index_file = self.storage_dir / f"{model_name}_index.faiss"
            mapping_file = self.storage_dir / f"{model_name}_mappings.json"
            
            if index_file.exists():
                index_file.unlink()
            if mapping_file.exists():
                mapping_file.unlink()
            
            # Update metadata file
            self.save_indices()
            
            print(f"Removed index for {model_name}")
            return True
            
        except Exception as e:
            print(f"Error removing index for {model_name}: {e}")
            return False


def create_vector_store(storage_dir: str = "vector_indices") -> FAISSVectorStore:
    """
    Create a FAISS vector store with the specified storage directory.
    
    Args:
        storage_dir: Directory to store FAISS indices
    
    Returns:
        Configured FAISSVectorStore instance
    """
    return FAISSVectorStore(storage_dir)


if __name__ == "__main__":
    # Test the vector store
    print("FAISS Vector Store Test")
    print("=" * 50)
    
    if not FAISS_AVAILABLE:
        print("FAISS not available. Please install with: pip install faiss-cpu")
        exit(1)
    
    # Create test vector store
    vector_store = create_vector_store("test_vector_store")
    
    # Create test data
    model_name = "test_model"
    embedding_dim = 128
    
    # Create index
    success = vector_store.create_index(model_name, embedding_dim)
    print(f"Index creation success: {success}")
    
    # Generate test embeddings
    num_vectors = 100
    test_embeddings = [np.random.randn(embedding_dim) for _ in range(num_vectors)]
    test_chunk_ids = [f"chunk_{i}" for i in range(num_vectors)]
    
    # Add embeddings
    success = vector_store.add_embeddings(model_name, test_embeddings, test_chunk_ids)
    print(f"Add embeddings success: {success}")
    
    # Test search
    query_embedding = np.random.randn(embedding_dim)
    results = vector_store.search(model_name, query_embedding, k=5)
    print(f"Search results: {len(results)} found")
    
    for i, (idx, score) in enumerate(results):
        print(f"  {i+1}. Index: {idx}, Score: {score:.4f}, Chunk ID: {test_chunk_ids[idx]}")
    
    # Get statistics
    stats = vector_store.get_stats()
    print(f"\nVector Store Statistics:")
    print(f"  Number of indices: {stats['num_indices']}")
    print(f"  Total vectors: {stats['total_vectors']}")
    print(f"  Storage directory: {stats['storage_dir']}")
    
    # Save indices
    vector_store.save_indices()
    print("\nIndices saved to disk")