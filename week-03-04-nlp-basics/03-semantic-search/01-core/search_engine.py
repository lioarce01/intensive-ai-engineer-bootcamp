#!/usr/bin/env python3
"""
Enhanced Semantic Search Engine
===============================

This module provides a comprehensive semantic search engine that integrates
FAISS vector search, TF-IDF baseline, and hybrid ranking approaches.
Built on top of the embedding engine and document store.

Features:
- FAISS-powered vector similarity search
- Support for multiple embedding models
- Query embedding and search
- Batch query processing
- Search result ranking and filtering
- Integration with document store
- Performance metrics and analysis

Author: AI Bootcamp Week 3-4
Date: 2025
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
import time
from datetime import datetime

# Core libraries
import numpy as np

# Import components
try:
    from document_store import DocumentStore
    from embedding_engine import EnhancedEmbeddingEngine, EmbeddingResult
    from vector_store import FAISSVectorStore, SearchResult
    COMPONENTS_AVAILABLE = True
except ImportError:
    COMPONENTS_AVAILABLE = False
    print("Warning: Could not import local components. Some features will be limited.")


@dataclass
class QueryResult:
    """Complete search result for a query."""
    query: str
    model_name: str
    search_time: float
    total_results: int
    results: List[SearchResult]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'query': self.query,
            'model_name': self.model_name,
            'search_time': self.search_time,
            'total_results': self.total_results,
            'results': [result.to_dict() for result in self.results],
            'metadata': self.metadata
        }
    
    def get_top_k(self, k: int) -> List[SearchResult]:
        """Get top K search results."""
        return self.results[:k]
    
    def filter_by_score(self, min_score: float) -> List[SearchResult]:
        """Filter results by minimum score."""
        return [result for result in self.results if result.score >= min_score]


@dataclass
class SearchConfig:
    """Configuration for search operations."""
    model_name: str = "sentence_transformer"
    top_k: int = 10
    score_threshold: float = 0.0
    include_metadata: bool = True
    normalize_scores: bool = True
    filter_duplicates: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'model_name': self.model_name,
            'top_k': self.top_k,
            'score_threshold': self.score_threshold,
            'include_metadata': self.include_metadata,
            'normalize_scores': self.normalize_scores,
            'filter_duplicates': self.filter_duplicates
        }


class SemanticSearchEngine:
    """
    Main semantic search engine that orchestrates embedding generation,
    vector search, and result processing.
    """
    
    def __init__(self, document_store: DocumentStore,
                 embedding_engine: EnhancedEmbeddingEngine,
                 vector_store: FAISSVectorStore):
        """
        Initialize semantic search engine.
        
        Args:
            document_store: Document store for metadata and content
            embedding_engine: Engine for generating embeddings
            vector_store: FAISS vector store for similarity search
        """
        self.document_store = document_store
        self.embedding_engine = embedding_engine
        self.vector_store = vector_store
        
        # Search statistics
        self.search_stats = {
            'total_searches': 0,
            'total_search_time': 0.0,
            'searches_by_model': {},
            'average_results_per_search': 0.0
        }
        
        print(f"Semantic Search Engine initialized with:")
        doc_stats = document_store.get_storage_stats()
        print(f"  - Document store: {doc_stats['documents']} documents, {doc_stats['chunks']} chunks")
        print(f"  - Vector store: {vector_store.get_stats()['num_indices']} indices, {vector_store.get_stats()['total_vectors']} vectors")
        print(f"  - Embedding engine: {len(embedding_engine.models)} models")
    
    def search(self, query: str, config: Optional[SearchConfig] = None) -> QueryResult:
        """
        Perform semantic search for a query.
        
        Args:
            query: Search query string
            config: Search configuration (uses default if None)
        
        Returns:
            QueryResult with search results and metadata
        """
        if config is None:
            config = SearchConfig()
        
        start_time = time.time()
        
        try:
            # Generate query embedding
            embedding_result = self.embedding_engine.embed_text(query, config.model_name)
            query_embedding = embedding_result.embedding
            
            # Perform vector search
            search_results = self.vector_store.search_with_metadata(
                model_name=config.model_name,
                query_embedding=query_embedding,
                document_store=self.document_store,
                k=config.top_k,
                threshold=config.score_threshold
            )
            
            # Process results
            if config.normalize_scores and search_results:
                search_results = self._normalize_scores(search_results)
            
            if config.filter_duplicates:
                search_results = self._filter_duplicates(search_results)
            
            search_time = time.time() - start_time
            
            # Update statistics
            self._update_search_stats(config.model_name, search_time, len(search_results))
            
            # Create result object
            result = QueryResult(
                query=query,
                model_name=config.model_name,
                search_time=search_time,
                total_results=len(search_results),
                results=search_results,
                metadata={
                    'embedding_dim': embedding_result.embedding_dim,
                    'from_cache': embedding_result.metadata.get('from_cache', False),
                    'config': config.to_dict()
                }
            )
            
            return result
            
        except Exception as e:
            print(f"Error during search: {e}")
            # Return empty result
            return QueryResult(
                query=query,
                model_name=config.model_name,
                search_time=time.time() - start_time,
                total_results=0,
                results=[],
                metadata={'error': str(e)}
            )
    
    def batch_search(self, queries: List[str], config: Optional[SearchConfig] = None) -> List[QueryResult]:
        """
        Perform batch search for multiple queries.
        
        Args:
            queries: List of search query strings
            config: Search configuration
        
        Returns:
            List of QueryResult objects
        """
        if config is None:
            config = SearchConfig()
        
        results = []
        
        print(f"Performing batch search for {len(queries)} queries...")
        
        for i, query in enumerate(queries):
            if i % 10 == 0 and i > 0:
                print(f"  Processed {i}/{len(queries)} queries...")
            
            result = self.search(query, config)
            results.append(result)
        
        print(f"Completed batch search: {len(results)} results")
        return results
    
    def compare_models(self, query: str, model_names: Optional[List[str]] = None,
                      top_k: int = 5) -> Dict[str, QueryResult]:
        """
        Compare search results across different embedding models.
        
        Args:
            query: Search query
            model_names: List of model names to compare (None = all available)
            top_k: Number of results per model
        
        Returns:
            Dictionary of model_name -> QueryResult
        """
        if model_names is None:
            model_names = list(self.vector_store.indices.keys())
        
        results = {}
        
        print(f"Comparing models for query: '{query}'")
        
        for model_name in model_names:
            try:
                config = SearchConfig(model_name=model_name, top_k=top_k)
                result = self.search(query, config)
                results[model_name] = result
                
                print(f"  {model_name}: {result.total_results} results, {result.search_time:.3f}s")
                
            except Exception as e:
                print(f"  Error with {model_name}: {e}")
                continue
        
        return results
    
    def get_similar_documents(self, document_id: str, model_name: str,
                            top_k: int = 5) -> List[SearchResult]:
        """
        Find documents similar to a given document.
        
        Args:
            document_id: ID of the reference document
            model_name: Embedding model to use
            top_k: Number of similar documents to return
        
        Returns:
            List of similar SearchResult objects
        """
        try:
            # Get document content
            document = self.document_store.get_document(document_id)
            if not document:
                print(f"Document {document_id} not found")
                return []
            
            # Use document title and content as query
            query = f"{document.get('title', '')} {document.get('content', '')[:500]}"
            
            config = SearchConfig(model_name=model_name, top_k=top_k + 5)  # Get extra to filter out self
            result = self.search(query, config)
            
            # Filter out the same document
            similar_results = []
            for search_result in result.results:
                if search_result.document_id != document_id:
                    similar_results.append(search_result)
                    if len(similar_results) >= top_k:
                        break
            
            return similar_results
            
        except Exception as e:
            print(f"Error finding similar documents: {e}")
            return []
    
    def explain_search(self, query: str, model_name: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Provide detailed explanation of search results.
        
        Args:
            query: Search query
            model_name: Embedding model to use
            top_k: Number of results to explain
        
        Returns:
            Dictionary with detailed search explanation
        """
        try:
            # Perform search
            config = SearchConfig(model_name=model_name, top_k=top_k)
            result = self.search(query, config)
            
            # Get query embedding
            query_embedding = self.embedding_engine.embed_text(query, model_name)
            
            explanation = {
                'query': query,
                'model_name': model_name,
                'query_embedding_info': {
                    'dimension': query_embedding.embedding_dim,
                    'norm': float(np.linalg.norm(query_embedding.embedding)),
                    'mean': float(np.mean(query_embedding.embedding)),
                    'std': float(np.std(query_embedding.embedding))
                },
                'search_info': {
                    'search_time': result.search_time,
                    'total_results': result.total_results,
                    'score_range': {
                        'min': min([r.score for r in result.results]) if result.results else 0,
                        'max': max([r.score for r in result.results]) if result.results else 0
                    }
                },
                'top_results': []
            }
            
            # Analyze top results
            for i, search_result in enumerate(result.results[:top_k]):
                # Get result embedding for comparison
                chunk_embedding = self.document_store.get_chunk_embedding(
                    search_result.chunk_id, model_name
                )
                
                result_analysis = {
                    'rank': i + 1,
                    'chunk_id': search_result.chunk_id,
                    'document_id': search_result.document_id,
                    'score': search_result.score,
                    'content_preview': search_result.content[:200] + "..." if len(search_result.content) > 200 else search_result.content,
                    'content_length': len(search_result.content)
                }
                
                if chunk_embedding is not None:
                    # Compute similarity metrics
                    cosine_sim = self.embedding_engine.compute_similarity(
                        query_embedding.embedding, chunk_embedding, 'cosine'
                    )
                    dot_product = self.embedding_engine.compute_similarity(
                        query_embedding.embedding, chunk_embedding, 'dot'
                    )
                    
                    result_analysis['similarity_metrics'] = {
                        'cosine': float(cosine_sim),
                        'dot_product': float(dot_product),
                        'faiss_score': search_result.score
                    }
                    
                    # Embedding comparison
                    result_analysis['embedding_info'] = {
                        'norm': float(np.linalg.norm(chunk_embedding)),
                        'mean': float(np.mean(chunk_embedding)),
                        'std': float(np.std(chunk_embedding))
                    }
                
                explanation['top_results'].append(result_analysis)
            
            return explanation
            
        except Exception as e:
            print(f"Error explaining search: {e}")
            return {'error': str(e)}
    
    def _normalize_scores(self, results: List[SearchResult]) -> List[SearchResult]:
        """Normalize scores to 0-1 range."""
        if not results:
            return results
        
        scores = [r.score for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            # All scores are the same
            for result in results:
                result.score = 1.0
        else:
            # Normalize to 0-1 range
            for result in results:
                result.score = (result.score - min_score) / (max_score - min_score)
        
        return results
    
    def _filter_duplicates(self, results: List[SearchResult]) -> List[SearchResult]:
        """Filter duplicate results based on document ID."""
        seen_documents = set()
        filtered_results = []
        
        for result in results:
            if result.document_id not in seen_documents:
                filtered_results.append(result)
                seen_documents.add(result.document_id)
        
        return filtered_results
    
    def _update_search_stats(self, model_name: str, search_time: float, num_results: int):
        """Update search statistics."""
        self.search_stats['total_searches'] += 1
        self.search_stats['total_search_time'] += search_time
        
        if model_name not in self.search_stats['searches_by_model']:
            self.search_stats['searches_by_model'][model_name] = {
                'count': 0,
                'total_time': 0.0,
                'total_results': 0
            }
        
        model_stats = self.search_stats['searches_by_model'][model_name]
        model_stats['count'] += 1
        model_stats['total_time'] += search_time
        model_stats['total_results'] += num_results
        
        # Update average
        total_results = sum(stats['total_results'] for stats in self.search_stats['searches_by_model'].values())
        self.search_stats['average_results_per_search'] = total_results / self.search_stats['total_searches']
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get comprehensive search statistics."""
        stats = dict(self.search_stats)
        
        # Add derived statistics
        if self.search_stats['total_searches'] > 0:
            stats['average_search_time'] = self.search_stats['total_search_time'] / self.search_stats['total_searches']
        else:
            stats['average_search_time'] = 0.0
        
        # Add per-model averages
        for model_name, model_stats in stats['searches_by_model'].items():
            if model_stats['count'] > 0:
                model_stats['average_time'] = model_stats['total_time'] / model_stats['count']
                model_stats['average_results'] = model_stats['total_results'] / model_stats['count']
            else:
                model_stats['average_time'] = 0.0
                model_stats['average_results'] = 0.0
        
        return stats
    
    def reset_stats(self):
        """Reset search statistics."""
        self.search_stats = {
            'total_searches': 0,
            'total_search_time': 0.0,
            'searches_by_model': {},
            'average_results_per_search': 0.0
        }


def create_search_engine(document_store_path: str, embedding_cache_file: str,
                        vector_store_path: str) -> SemanticSearchEngine:
    """
    Create a complete semantic search engine from file paths.
    
    Args:
        document_store_path: Path to document store
        embedding_cache_file: Path to embedding cache
        vector_store_path: Path to vector store
    
    Returns:
        Configured SemanticSearchEngine
    """
    if not COMPONENTS_AVAILABLE:
        raise ImportError("Required components not available")
    
    # Initialize components
    document_store = DocumentStore(document_store_path)
    
    from embedding_engine import create_embedding_engine
    embedding_engine = create_embedding_engine(cache_file=embedding_cache_file)
    
    vector_store = FAISSVectorStore(vector_store_path)
    
    # Create and return search engine
    return SemanticSearchEngine(document_store, embedding_engine, vector_store)


if __name__ == "__main__":
    # Test the search engine
    print("Semantic Search Engine Test")
    print("=" * 50)
    
    if not COMPONENTS_AVAILABLE:
        print("Required components not available. Cannot run test.")
        exit(1)
    
    # You would typically initialize this with real data
    print("Note: This test requires existing document store, embeddings, and vector indices.")
    print("Run the integration demo script to see full functionality.")