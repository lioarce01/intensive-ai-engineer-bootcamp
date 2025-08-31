#!/usr/bin/env python3
"""
Document Store for Semantic Search
==================================

This module provides a JSON-based document storage system for the semantic search engine.
It handles storage and retrieval of documents, chunks, embeddings, and metadata with 
efficient indexing capabilities.

Features:
- JSON-based storage for documents and metadata
- Efficient chunk and embedding storage
- Fast retrieval by document ID and chunk ID
- Batch operations for performance
- Index building and management
- Storage statistics and optimization

Author: AI Bootcamp Week 3-4
Date: 2025
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any, Iterator
from dataclasses import dataclass, asdict
import time
from datetime import datetime

# Core libraries
import numpy as np

# Import document processing components
try:
    from document_processor import Document, DocumentChunk
    DOCUMENT_PROCESSOR_AVAILABLE = True
except ImportError:
    # Create minimal fallbacks
    DOCUMENT_PROCESSOR_AVAILABLE = False
    print("Warning: Document processor not available. Using minimal fallbacks.")
    
    @dataclass
    class DocumentChunk:
        chunk_id: str
        document_id: str
        content: str
        chunk_index: int
        start_char: int
        end_char: int
        metadata: Dict[str, Any]
    
    @dataclass
    class Document:
        document_id: str
        title: str
        content: str
        file_path: Optional[str]
        file_type: str
        metadata: Dict[str, Any]
        chunks: List[DocumentChunk]

# Import embedding components
try:
    from embedding_engine import EmbeddingResult
    EMBEDDING_ENGINE_AVAILABLE = True
except ImportError:
    EMBEDDING_ENGINE_AVAILABLE = False
    print("Warning: Embedding engine not available. Using minimal fallbacks.")
    
    @dataclass
    class EmbeddingResult:
        embedding: np.ndarray
        model_name: str
        embedding_dim: int
        metadata: Dict[str, Any]


@dataclass
class StoredDocument:
    """Document stored in the document store."""
    document_id: str
    title: str
    content: str
    file_path: Optional[str]
    file_type: str
    metadata: Dict[str, Any]
    num_chunks: int
    created_at: str
    updated_at: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StoredDocument':
        """Create from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_document(cls, document: Document) -> 'StoredDocument':
        """Create from Document object."""
        now = datetime.now().isoformat()
        return cls(
            document_id=document.document_id,
            title=document.title,
            content=document.content,
            file_path=document.file_path,
            file_type=document.file_type,
            metadata=document.metadata,
            num_chunks=len(document.chunks),
            created_at=now,
            updated_at=now
        )


@dataclass
class StoredChunk:
    """Chunk stored in the document store."""
    chunk_id: str
    document_id: str
    content: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any]
    has_embedding: bool = False
    embedding_models: List[str] = None
    
    def __post_init__(self):
        if self.embedding_models is None:
            self.embedding_models = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StoredChunk':
        """Create from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_chunk(cls, chunk: DocumentChunk) -> 'StoredChunk':
        """Create from DocumentChunk object."""
        return cls(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            content=chunk.content,
            chunk_index=chunk.chunk_index,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
            metadata=chunk.metadata
        )


@dataclass
class StoredEmbedding:
    """Embedding stored in the document store."""
    chunk_id: str
    model_name: str
    embedding: np.ndarray
    embedding_dim: int
    metadata: Dict[str, Any]
    created_at: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization (embedding as list)."""
        data = asdict(self)
        data['embedding'] = self.embedding.tolist()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StoredEmbedding':
        """Create from dictionary."""
        data['embedding'] = np.array(data['embedding'])
        return cls(**data)
    
    @classmethod
    def from_embedding_result(cls, chunk_id: str, result: EmbeddingResult) -> 'StoredEmbedding':
        """Create from EmbeddingResult."""
        return cls(
            chunk_id=chunk_id,
            model_name=result.model_name,
            embedding=result.embedding,
            embedding_dim=result.embedding_dim,
            metadata=result.metadata,
            created_at=datetime.now().isoformat()
        )


class DocumentStore:
    """
    JSON-based document store for efficient storage and retrieval of documents,
    chunks, and embeddings.
    """
    
    def __init__(self, storage_dir: str):
        """
        Initialize document store.
        
        Args:
            storage_dir: Directory for storing all data files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Define file paths
        self.documents_file = self.storage_dir / "documents.json"
        self.chunks_file = self.storage_dir / "chunks.json"
        self.embeddings_dir = self.storage_dir / "embeddings"
        self.embeddings_dir.mkdir(exist_ok=True)
        self.index_file = self.storage_dir / "index.json"
        
        # In-memory indexes for fast lookup
        self.documents_index = {}  # document_id -> StoredDocument
        self.chunks_index = {}     # chunk_id -> StoredChunk
        self.doc_chunks_index = {} # document_id -> List[chunk_id]
        self.embeddings_index = {} # (chunk_id, model_name) -> file_path
        
        # Load existing data
        self._load_data()
        
        print(f"Initialized document store at: {self.storage_dir}")
        print(f"  Documents: {len(self.documents_index)}")
        print(f"  Chunks: {len(self.chunks_index)}")
        print(f"  Embeddings: {len(self.embeddings_index)}")
    
    def _load_data(self):
        """Load existing data from storage files."""
        # Load documents
        if self.documents_file.exists():
            try:
                with open(self.documents_file, 'r') as f:
                    docs_data = json.load(f)
                
                for doc_data in docs_data:
                    doc = StoredDocument.from_dict(doc_data)
                    self.documents_index[doc.document_id] = doc
            except Exception as e:
                print(f"Error loading documents: {e}")
        
        # Load chunks
        if self.chunks_file.exists():
            try:
                with open(self.chunks_file, 'r') as f:
                    chunks_data = json.load(f)
                
                for chunk_data in chunks_data:
                    chunk = StoredChunk.from_dict(chunk_data)
                    self.chunks_index[chunk.chunk_id] = chunk
                    
                    # Build document->chunks mapping
                    if chunk.document_id not in self.doc_chunks_index:
                        self.doc_chunks_index[chunk.document_id] = []
                    self.doc_chunks_index[chunk.document_id].append(chunk.chunk_id)
            except Exception as e:
                print(f"Error loading chunks: {e}")
        
        # Load embedding index
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    index_data = json.load(f)
                    self.embeddings_index = {
                        (item['chunk_id'], item['model_name']): item['file_path']
                        for item in index_data.get('embeddings', [])
                    }
            except Exception as e:
                print(f"Error loading embedding index: {e}")
    
    def _save_documents(self):
        """Save documents to file."""
        try:
            docs_data = [doc.to_dict() for doc in self.documents_index.values()]
            with open(self.documents_file, 'w') as f:
                json.dump(docs_data, f, indent=2)
        except Exception as e:
            print(f"Error saving documents: {e}")
    
    def _save_chunks(self):
        """Save chunks to file."""
        try:
            chunks_data = [chunk.to_dict() for chunk in self.chunks_index.values()]
            with open(self.chunks_file, 'w') as f:
                json.dump(chunks_data, f, indent=2)
        except Exception as e:
            print(f"Error saving chunks: {e}")
    
    def _save_index(self):
        """Save embedding index to file."""
        try:
            index_data = {
                'embeddings': [
                    {'chunk_id': chunk_id, 'model_name': model_name, 'file_path': file_path}
                    for (chunk_id, model_name), file_path in self.embeddings_index.items()
                ]
            }
            with open(self.index_file, 'w') as f:
                json.dump(index_data, f, indent=2)
        except Exception as e:
            print(f"Error saving index: {e}")
    
    def add_document(self, document: Document) -> bool:
        """
        Add a document with its chunks to the store.
        
        Args:
            document: Document object to store
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if document already exists
            if document.document_id in self.documents_index:
                print(f"Document {document.document_id} already exists. Use update_document() to modify.")
                return False
            
            # Store document
            stored_doc = StoredDocument.from_document(document)
            self.documents_index[document.document_id] = stored_doc
            
            # Store chunks
            chunk_ids = []
            for chunk in document.chunks:
                stored_chunk = StoredChunk.from_chunk(chunk)
                self.chunks_index[chunk.chunk_id] = stored_chunk
                chunk_ids.append(chunk.chunk_id)
            
            # Update document->chunks mapping
            self.doc_chunks_index[document.document_id] = chunk_ids
            
            # Save to disk
            self._save_documents()
            self._save_chunks()
            
            print(f"Added document: {document.document_id} with {len(document.chunks)} chunks")
            return True
            
        except Exception as e:
            print(f"Error adding document {document.document_id}: {e}")
            return False
    
    def add_documents(self, documents: List[Document]) -> int:
        """
        Add multiple documents in batch.
        
        Args:
            documents: List of Document objects to store
        
        Returns:
            Number of successfully added documents
        """
        success_count = 0
        
        for document in documents:
            if self.add_document(document):
                success_count += 1
        
        print(f"Successfully added {success_count}/{len(documents)} documents")
        return success_count
    
    def get_document(self, document_id: str) -> Optional[StoredDocument]:
        """
        Get a document by ID.
        
        Args:
            document_id: Document identifier
        
        Returns:
            StoredDocument if found, None otherwise
        """
        return self.documents_index.get(document_id)
    
    def get_chunk(self, chunk_id: str) -> Optional[StoredChunk]:
        """
        Get a chunk by ID.
        
        Args:
            chunk_id: Chunk identifier
        
        Returns:
            StoredChunk if found, None otherwise
        """
        return self.chunks_index.get(chunk_id)
    
    def get_document_chunks(self, document_id: str) -> List[StoredChunk]:
        """
        Get all chunks for a document.
        
        Args:
            document_id: Document identifier
        
        Returns:
            List of StoredChunk objects
        """
        chunk_ids = self.doc_chunks_index.get(document_id, [])
        return [self.chunks_index[chunk_id] for chunk_id in chunk_ids if chunk_id in self.chunks_index]
    
    def add_embedding(self, chunk_id: str, embedding_result: EmbeddingResult) -> bool:
        """
        Add embedding for a chunk.
        
        Args:
            chunk_id: Chunk identifier
            embedding_result: EmbeddingResult object
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if chunk exists
            if chunk_id not in self.chunks_index:
                print(f"Chunk {chunk_id} not found")
                return False
            
            # Create stored embedding
            stored_embedding = StoredEmbedding.from_embedding_result(chunk_id, embedding_result)
            
            # Save embedding to file
            embedding_filename = f"{chunk_id}_{embedding_result.model_name}.pkl"
            embedding_path = self.embeddings_dir / embedding_filename
            
            with open(embedding_path, 'wb') as f:
                pickle.dump(stored_embedding, f)
            
            # Update index
            key = (chunk_id, embedding_result.model_name)
            self.embeddings_index[key] = str(embedding_path)
            
            # Update chunk metadata
            chunk = self.chunks_index[chunk_id]
            chunk.has_embedding = True
            if embedding_result.model_name not in chunk.embedding_models:
                chunk.embedding_models.append(embedding_result.model_name)
            
            # Save updated index and chunks
            self._save_index()
            self._save_chunks()
            
            return True
            
        except Exception as e:
            print(f"Error adding embedding for chunk {chunk_id}: {e}")
            return False
    
    def get_embedding(self, chunk_id: str, model_name: str) -> Optional[StoredEmbedding]:
        """
        Get embedding for a chunk and model.
        
        Args:
            chunk_id: Chunk identifier
            model_name: Model name
        
        Returns:
            StoredEmbedding if found, None otherwise
        """
        key = (chunk_id, model_name)
        embedding_path = self.embeddings_index.get(key)
        
        if embedding_path and os.path.exists(embedding_path):
            try:
                with open(embedding_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading embedding for {chunk_id}, {model_name}: {e}")
        
        return None
    
    def get_all_embeddings(self, model_name: str) -> List[Tuple[str, StoredEmbedding]]:
        """
        Get all embeddings for a specific model.
        
        Args:
            model_name: Model name
        
        Returns:
            List of (chunk_id, StoredEmbedding) tuples
        """
        embeddings = []
        
        for (chunk_id, emb_model_name), embedding_path in self.embeddings_index.items():
            if emb_model_name == model_name:
                stored_embedding = self.get_embedding(chunk_id, model_name)
                if stored_embedding:
                    embeddings.append((chunk_id, stored_embedding))
        
        return embeddings
    
    def add_embeddings_batch(self, chunk_embeddings: List[Tuple[str, EmbeddingResult]]) -> int:
        """
        Add multiple embeddings in batch.
        
        Args:
            chunk_embeddings: List of (chunk_id, EmbeddingResult) tuples
        
        Returns:
            Number of successfully added embeddings
        """
        success_count = 0
        
        for chunk_id, embedding_result in chunk_embeddings:
            if self.add_embedding(chunk_id, embedding_result):
                success_count += 1
        
        print(f"Successfully added {success_count}/{len(chunk_embeddings)} embeddings")
        return success_count
    
    def search_documents(self, query: str = None, file_type: str = None, 
                        metadata_filter: Dict[str, Any] = None) -> List[StoredDocument]:
        """
        Search documents by various criteria.
        
        Args:
            query: Text query to search in title and content
            file_type: Filter by file type
            metadata_filter: Filter by metadata fields
        
        Returns:
            List of matching StoredDocument objects
        """
        results = []
        
        for doc in self.documents_index.values():
            # Apply filters
            if file_type and doc.file_type != file_type:
                continue
            
            if metadata_filter:
                match = True
                for key, value in metadata_filter.items():
                    if key not in doc.metadata or doc.metadata[key] != value:
                        match = False
                        break
                if not match:
                    continue
            
            if query:
                query_lower = query.lower()
                if (query_lower not in doc.title.lower() and 
                    query_lower not in doc.content.lower()):
                    continue
            
            results.append(doc)
        
        return results
    
    def get_chunks_with_embeddings(self, model_name: str) -> List[StoredChunk]:
        """
        Get all chunks that have embeddings for a specific model.
        
        Args:
            model_name: Model name
        
        Returns:
            List of StoredChunk objects
        """
        chunks = []
        
        for chunk in self.chunks_index.values():
            if model_name in chunk.embedding_models:
                chunks.append(chunk)
        
        return chunks
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and all its chunks and embeddings.
        
        Args:
            document_id: Document identifier
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if document_id not in self.documents_index:
                print(f"Document {document_id} not found")
                return False
            
            # Get chunk IDs
            chunk_ids = self.doc_chunks_index.get(document_id, [])
            
            # Delete embeddings
            for chunk_id in chunk_ids:
                self._delete_chunk_embeddings(chunk_id)
            
            # Delete chunks
            for chunk_id in chunk_ids:
                if chunk_id in self.chunks_index:
                    del self.chunks_index[chunk_id]
            
            # Delete document
            del self.documents_index[document_id]
            
            # Update document->chunks mapping
            if document_id in self.doc_chunks_index:
                del self.doc_chunks_index[document_id]
            
            # Save changes
            self._save_documents()
            self._save_chunks()
            self._save_index()
            
            print(f"Deleted document: {document_id}")
            return True
            
        except Exception as e:
            print(f"Error deleting document {document_id}: {e}")
            return False
    
    def _delete_chunk_embeddings(self, chunk_id: str):
        """Delete all embeddings for a chunk."""
        keys_to_delete = []
        
        for (cid, model_name), embedding_path in self.embeddings_index.items():
            if cid == chunk_id:
                # Delete embedding file
                if os.path.exists(embedding_path):
                    try:
                        os.remove(embedding_path)
                    except Exception as e:
                        print(f"Error deleting embedding file {embedding_path}: {e}")
                
                keys_to_delete.append((cid, model_name))
        
        # Remove from index
        for key in keys_to_delete:
            del self.embeddings_index[key]
    
    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """
        Get all chunks as dictionaries.
        
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        for chunk in self.chunks_index.values():
            chunk_dict = chunk.to_dict()
            chunks.append(chunk_dict)
        return chunks
    
    def get_chunk_embedding(self, chunk_id: str, model_name: str) -> Optional[np.ndarray]:
        """
        Get embedding array for a specific chunk and model.
        
        Args:
            chunk_id: Chunk identifier
            model_name: Model name
        
        Returns:
            Numpy array of embedding or None if not found
        """
        embedding = self.get_embedding(chunk_id, model_name)
        if embedding:
            return embedding.embedding
        return None
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        # Calculate storage sizes
        def get_dir_size(path):
            total = 0
            try:
                for entry in os.scandir(path):
                    if entry.is_file():
                        total += entry.stat().st_size
                    elif entry.is_dir():
                        total += get_dir_size(entry.path)
            except:
                pass
            return total
        
        total_size = get_dir_size(self.storage_dir)
        embeddings_size = get_dir_size(self.embeddings_dir)
        
        # Model statistics
        model_stats = {}
        for (chunk_id, model_name), _ in self.embeddings_index.items():
            if model_name not in model_stats:
                model_stats[model_name] = 0
            model_stats[model_name] += 1
        
        # Chunk statistics
        chunks_per_doc = []
        for doc_id in self.doc_chunks_index:
            chunks_per_doc.append(len(self.doc_chunks_index[doc_id]))
        
        avg_chunks_per_doc = np.mean(chunks_per_doc) if chunks_per_doc else 0
        
        return {
            'storage_directory': str(self.storage_dir),
            'total_storage_mb': total_size / (1024 * 1024),
            'embeddings_storage_mb': embeddings_size / (1024 * 1024),
            'documents': len(self.documents_index),
            'chunks': len(self.chunks_index),
            'embeddings': len(self.embeddings_index),
            'models': list(model_stats.keys()),
            'embeddings_per_model': model_stats,
            'avg_chunks_per_doc': avg_chunks_per_doc,
            'files': {
                'documents_file_exists': self.documents_file.exists(),
                'chunks_file_exists': self.chunks_file.exists(),
                'index_file_exists': self.index_file.exists(),
                'embeddings_dir_exists': self.embeddings_dir.exists()
            }
        }
    
    def optimize_storage(self):
        """Optimize storage by rebuilding indexes and cleaning up."""
        print("Optimizing document store...")
        
        # Rebuild doc->chunks mapping
        self.doc_chunks_index = {}
        for chunk in self.chunks_index.values():
            if chunk.document_id not in self.doc_chunks_index:
                self.doc_chunks_index[chunk.document_id] = []
            self.doc_chunks_index[chunk.document_id].append(chunk.chunk_id)
        
        # Clean up orphaned embeddings
        valid_chunk_ids = set(self.chunks_index.keys())
        keys_to_remove = []
        
        for (chunk_id, model_name), embedding_path in self.embeddings_index.items():
            if chunk_id not in valid_chunk_ids:
                keys_to_remove.append((chunk_id, model_name))
                # Delete orphaned embedding file
                if os.path.exists(embedding_path):
                    try:
                        os.remove(embedding_path)
                        print(f"Removed orphaned embedding: {embedding_path}")
                    except Exception as e:
                        print(f"Error removing {embedding_path}: {e}")
        
        for key in keys_to_remove:
            del self.embeddings_index[key]
        
        # Save all data
        self._save_documents()
        self._save_chunks()
        self._save_index()
        
        print(f"Optimization complete. Removed {len(keys_to_remove)} orphaned embeddings.")


def create_document_store(storage_dir: str = "document_store") -> DocumentStore:
    """
    Create a document store in the specified directory.
    
    Args:
        storage_dir: Directory for storing data
    
    Returns:
        DocumentStore instance
    """
    return DocumentStore(storage_dir)


if __name__ == "__main__":
    # Test the document store
    print("Document Store Test")
    print("=" * 50)
    
    # Create test store
    store = create_document_store("test_store")
    
    # Create test document (minimal version for testing)
    test_doc = Document(
        document_id="test_doc_1",
        title="Test Document",
        content="This is a test document for the semantic search system. It contains multiple sentences to test chunking.",
        file_path=None,
        file_type="text",
        metadata={"source": "test", "topic": "testing"},
        chunks=[
            DocumentChunk(
                chunk_id="test_doc_1_chunk_0",
                document_id="test_doc_1",
                content="This is a test document for the semantic search system.",
                chunk_index=0,
                start_char=0,
                end_char=57,
                metadata={"chunk_type": "test"}
            ),
            DocumentChunk(
                chunk_id="test_doc_1_chunk_1",
                document_id="test_doc_1",
                content="It contains multiple sentences to test chunking.",
                chunk_index=1,
                start_char=58,
                end_char=105,
                metadata={"chunk_type": "test"}
            )
        ]
    )
    
    # Test adding document
    success = store.add_document(test_doc)
    print(f"Added document: {success}")
    
    # Test retrieval
    retrieved_doc = store.get_document("test_doc_1")
    print(f"Retrieved document: {retrieved_doc.title if retrieved_doc else 'None'}")
    
    # Test chunk retrieval
    chunks = store.get_document_chunks("test_doc_1")
    print(f"Document chunks: {len(chunks)}")
    
    for chunk in chunks:
        print(f"  - {chunk.chunk_id}: {chunk.content}")
    
    # Test storage stats
    stats = store.get_storage_stats()
    print("\nStorage Statistics:")
    for key, value in stats.items():
        if key not in ['files']:
            print(f"  {key}: {value}")
    
    print(f"Test completed successfully!")