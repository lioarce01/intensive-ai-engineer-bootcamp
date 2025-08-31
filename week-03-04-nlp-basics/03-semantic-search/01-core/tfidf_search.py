#!/usr/bin/env python3
"""
TF-IDF Baseline Search Engine
============================

This module provides a classical TF-IDF search implementation using scikit-learn
for comparison with semantic search approaches. It processes the same document
chunks and provides compatible search results.

Features:
- TF-IDF vectorization using scikit-learn
- Document preprocessing and tokenization
- Cosine similarity search
- Batch query processing
- Compatible interface with semantic search
- Performance benchmarking
- Result ranking and filtering

Author: AI Bootcamp Week 3-4
Date: 2025
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
import time
import re
from collections import Counter

# Core libraries
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download NLTK data if not available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

# Import components
try:
    from document_store import DocumentStore
    COMPONENTS_AVAILABLE = True
except ImportError:
    COMPONENTS_AVAILABLE = False
    print("Warning: Could not import document store. Some features will be limited.")


@dataclass
class TFIDFResult:
    """Result from TF-IDF search."""
    chunk_id: str
    document_id: str
    content: str
    score: float
    rank: int
    matched_terms: List[str]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'chunk_id': self.chunk_id,
            'document_id': self.document_id,
            'content': self.content,
            'score': self.score,
            'rank': self.rank,
            'matched_terms': self.matched_terms,
            'metadata': self.metadata
        }


@dataclass
class TFIDFQueryResult:
    """Complete TF-IDF search result for a query."""
    query: str
    processed_query: str
    search_time: float
    total_results: int
    results: List[TFIDFResult]
    query_terms: List[str]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'query': self.query,
            'processed_query': self.processed_query,
            'search_time': self.search_time,
            'total_results': self.total_results,
            'results': [result.to_dict() for result in self.results],
            'query_terms': self.query_terms,
            'metadata': self.metadata
        }
    
    def get_top_k(self, k: int) -> List[TFIDFResult]:
        """Get top K search results."""
        return self.results[:k]
    
    def filter_by_score(self, min_score: float) -> List[TFIDFResult]:
        """Filter results by minimum score."""
        return [result for result in self.results if result.score >= min_score]


class TextPreprocessor:
    """Text preprocessing for TF-IDF search."""
    
    def __init__(self, use_stemming: bool = True, remove_stopwords: bool = True,
                 min_word_length: int = 2):
        """
        Initialize text preprocessor.
        
        Args:
            use_stemming: Whether to use stemming
            remove_stopwords: Whether to remove stopwords
            min_word_length: Minimum word length to include
        """
        self.use_stemming = use_stemming
        self.remove_stopwords = remove_stopwords
        self.min_word_length = min_word_length
        
        # Initialize stemmer and stopwords
        self.stemmer = PorterStemmer() if use_stemming else None
        self.stopwords = set(stopwords.words('english')) if remove_stopwords else set()
        
        # Additional stopwords for technical content
        self.stopwords.update([
            'also', 'however', 'therefore', 'moreover', 'furthermore',
            'additionally', 'consequently', 'nonetheless', 'meanwhile'
        ])
    
    def preprocess(self, text: str) -> str:
        """
        Preprocess text for TF-IDF.
        
        Args:
            text: Input text
        
        Returns:
            Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers (keep letters and spaces)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Tokenize
        try:
            tokens = word_tokenize(text)
        except:
            # Fallback to simple split if NLTK fails
            tokens = text.split()
        
        # Process tokens
        processed_tokens = []
        for token in tokens:
            # Skip if too short
            if len(token) < self.min_word_length:
                continue
            
            # Skip stopwords
            if self.remove_stopwords and token in self.stopwords:
                continue
            
            # Apply stemming
            if self.use_stemming and self.stemmer:
                token = self.stemmer.stem(token)
            
            processed_tokens.append(token)
        
        return ' '.join(processed_tokens)
    
    def extract_terms(self, text: str) -> List[str]:
        """Extract and return individual terms from text."""
        processed_text = self.preprocess(text)
        return processed_text.split() if processed_text else []


class TFIDFSearchEngine:
    """
    TF-IDF based search engine for baseline comparison.
    """
    
    def __init__(self, document_store: Optional[DocumentStore] = None,
                 max_features: int = 10000, ngram_range: Tuple[int, int] = (1, 2),
                 use_stemming: bool = True, remove_stopwords: bool = True):
        """
        Initialize TF-IDF search engine.
        
        Args:
            document_store: Document store for content and metadata
            max_features: Maximum number of features for TF-IDF
            ngram_range: N-gram range for TF-IDF (default: unigrams and bigrams)
            use_stemming: Whether to use stemming
            remove_stopwords: Whether to remove stopwords
        """
        self.document_store = document_store
        self.max_features = max_features
        self.ngram_range = ngram_range
        
        # Initialize preprocessor
        self.preprocessor = TextPreprocessor(use_stemming, remove_stopwords)
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            preprocessor=self.preprocessor.preprocess,
            tokenizer=str.split,  # Use simple split since preprocessing handles tokenization
            lowercase=False,  # Already handled in preprocessing
            stop_words=None  # Already handled in preprocessing
        )
        
        # Store document data
        self.documents = []  # List of document texts
        self.chunk_metadata = []  # Corresponding metadata
        self.tfidf_matrix = None
        self.is_fitted = False
        
        # Search statistics
        self.search_stats = {
            'total_searches': 0,
            'total_search_time': 0.0,
            'total_results': 0
        }
        
        print(f"TF-IDF Search Engine initialized with:")
        print(f"  - Max features: {max_features}")
        print(f"  - N-gram range: {ngram_range}")
        print(f"  - Stemming: {use_stemming}")
        print(f"  - Remove stopwords: {remove_stopwords}")
    
    def build_index(self, chunks: Optional[List[Dict]] = None) -> bool:
        """
        Build TF-IDF index from document chunks.
        
        Args:
            chunks: List of chunk dictionaries (uses document store if None)
        
        Returns:
            True if index built successfully, False otherwise
        """
        try:
            # Get chunks from document store if not provided
            if chunks is None:
                if not self.document_store:
                    print("No chunks provided and no document store available")
                    return False
                chunks = list(self.document_store.get_all_chunks())
            
            if not chunks:
                print("No chunks available for indexing")
                return False
            
            print(f"Building TF-IDF index from {len(chunks)} chunks...")
            
            # Extract text content and metadata
            self.documents = []
            self.chunk_metadata = []
            
            for chunk in chunks:
                content = chunk.get('content', '')
                if content and isinstance(content, str):
                    self.documents.append(content)
                    self.chunk_metadata.append({
                        'chunk_id': chunk.get('chunk_id', ''),
                        'document_id': chunk.get('document_id', ''),
                        'content': content,
                        'metadata': chunk.get('metadata', {})
                    })
            
            if not self.documents:
                print("No valid document content found")
                return False
            
            # Fit TF-IDF vectorizer and transform documents
            print("Fitting TF-IDF vectorizer...")
            self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
            self.is_fitted = True
            
            print(f"TF-IDF index built successfully:")
            print(f"  - Documents: {len(self.documents)}")
            print(f"  - Vocabulary size: {len(self.vectorizer.vocabulary_)}")
            print(f"  - Matrix shape: {self.tfidf_matrix.shape}")
            print(f"  - Matrix density: {self.tfidf_matrix.nnz / (self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[1]):.4f}")
            
            return True
            
        except Exception as e:
            print(f"Error building TF-IDF index: {e}")
            return False
    
    def search(self, query: str, top_k: int = 10, min_score: float = 0.0) -> TFIDFQueryResult:
        """
        Search using TF-IDF similarity.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            min_score: Minimum similarity score threshold
        
        Returns:
            TFIDFQueryResult with search results
        """
        start_time = time.time()
        
        if not self.is_fitted:
            print("TF-IDF index not built. Call build_index() first.")
            return TFIDFQueryResult(
                query=query,
                processed_query="",
                search_time=0.0,
                total_results=0,
                results=[],
                query_terms=[],
                metadata={'error': 'Index not built'}
            )
        
        try:
            # Preprocess query
            processed_query = self.preprocessor.preprocess(query)
            query_terms = processed_query.split() if processed_query else []
            
            if not processed_query:
                print("Query preprocessing resulted in empty query")
                return TFIDFQueryResult(
                    query=query,
                    processed_query=processed_query,
                    search_time=0.0,
                    total_results=0,
                    results=[],
                    query_terms=query_terms,
                    metadata={'warning': 'Empty processed query'}
                )
            
            # Transform query to TF-IDF vector
            query_vector = self.vectorizer.transform([processed_query])
            
            # Compute cosine similarity with all documents
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Get top results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Filter by minimum score and create results
            results = []
            for rank, idx in enumerate(top_indices):
                score = float(similarities[idx])
                
                if score < min_score:
                    break
                
                # Get matched terms
                matched_terms = self._get_matched_terms(query_terms, idx)
                
                # Create result
                result = TFIDFResult(
                    chunk_id=self.chunk_metadata[idx]['chunk_id'],
                    document_id=self.chunk_metadata[idx]['document_id'],
                    content=self.chunk_metadata[idx]['content'],
                    score=score,
                    rank=rank + 1,
                    matched_terms=matched_terms,
                    metadata=self.chunk_metadata[idx]['metadata']
                )
                results.append(result)
            
            search_time = time.time() - start_time
            
            # Update statistics
            self._update_search_stats(search_time, len(results))
            
            # Create query result
            query_result = TFIDFQueryResult(
                query=query,
                processed_query=processed_query,
                search_time=search_time,
                total_results=len(results),
                results=results,
                query_terms=query_terms,
                metadata={
                    'vocabulary_size': len(self.vectorizer.vocabulary_),
                    'max_similarity': max(similarities) if len(similarities) > 0 else 0.0,
                    'mean_similarity': float(np.mean(similarities)) if len(similarities) > 0 else 0.0
                }
            )
            
            return query_result
            
        except Exception as e:
            print(f"Error during TF-IDF search: {e}")
            return TFIDFQueryResult(
                query=query,
                processed_query="",
                search_time=time.time() - start_time,
                total_results=0,
                results=[],
                query_terms=[],
                metadata={'error': str(e)}
            )
    
    def batch_search(self, queries: List[str], top_k: int = 10, 
                    min_score: float = 0.0) -> List[TFIDFQueryResult]:
        """
        Perform batch search for multiple queries.
        
        Args:
            queries: List of search queries
            top_k: Number of results per query
            min_score: Minimum score threshold
        
        Returns:
            List of TFIDFQueryResult objects
        """
        results = []
        
        print(f"Performing TF-IDF batch search for {len(queries)} queries...")
        
        for i, query in enumerate(queries):
            if i % 10 == 0 and i > 0:
                print(f"  Processed {i}/{len(queries)} queries...")
            
            result = self.search(query, top_k, min_score)
            results.append(result)
        
        print(f"Completed TF-IDF batch search: {len(results)} results")
        return results
    
    def explain_search(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Provide detailed explanation of TF-IDF search results.
        
        Args:
            query: Search query
            top_k: Number of results to explain
        
        Returns:
            Dictionary with detailed search explanation
        """
        if not self.is_fitted:
            return {'error': 'Index not built'}
        
        try:
            # Perform search
            result = self.search(query, top_k)
            
            # Get query vector and feature names
            query_vector = self.vectorizer.transform([result.processed_query])
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Get top query terms by TF-IDF weight
            query_scores = query_vector.toarray()[0]
            query_term_scores = [(feature_names[i], score) for i, score in enumerate(query_scores) if score > 0]
            query_term_scores.sort(key=lambda x: x[1], reverse=True)
            
            explanation = {
                'query': query,
                'processed_query': result.processed_query,
                'query_terms': result.query_terms,
                'query_tfidf_terms': query_term_scores[:10],  # Top 10 TF-IDF terms
                'search_info': {
                    'search_time': result.search_time,
                    'total_results': result.total_results,
                    'vocabulary_size': len(self.vectorizer.vocabulary_)
                },
                'top_results': []
            }
            
            # Analyze top results
            for search_result in result.results[:top_k]:
                # Get document vector
                doc_idx = next(i for i, meta in enumerate(self.chunk_metadata) 
                             if meta['chunk_id'] == search_result.chunk_id)
                doc_vector = self.tfidf_matrix[doc_idx].toarray()[0]
                
                # Get top terms in document
                doc_term_scores = [(feature_names[i], score) for i, score in enumerate(doc_vector) if score > 0]
                doc_term_scores.sort(key=lambda x: x[1], reverse=True)
                
                result_analysis = {
                    'rank': search_result.rank,
                    'chunk_id': search_result.chunk_id,
                    'document_id': search_result.document_id,
                    'score': search_result.score,
                    'matched_terms': search_result.matched_terms,
                    'content_preview': search_result.content[:200] + "..." if len(search_result.content) > 200 else search_result.content,
                    'top_tfidf_terms': doc_term_scores[:10],  # Top 10 document terms
                    'content_length': len(search_result.content)
                }
                
                explanation['top_results'].append(result_analysis)
            
            return explanation
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_vocabulary_stats(self) -> Dict[str, Any]:
        """Get statistics about the TF-IDF vocabulary."""
        if not self.is_fitted:
            return {'error': 'Index not built'}
        
        try:
            vocabulary = self.vectorizer.vocabulary_
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Get IDF values
            idf_values = self.vectorizer.idf_
            
            # Find terms with highest/lowest IDF
            idf_pairs = list(zip(feature_names, idf_values))
            idf_pairs.sort(key=lambda x: x[1])
            
            # Get document frequency statistics
            doc_freq = np.array((self.tfidf_matrix > 0).sum(axis=0)).flatten()
            
            stats = {
                'vocabulary_size': len(vocabulary),
                'document_count': self.tfidf_matrix.shape[0],
                'feature_count': self.tfidf_matrix.shape[1],
                'matrix_density': self.tfidf_matrix.nnz / (self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[1]),
                'ngram_range': self.ngram_range,
                'most_common_terms': idf_pairs[:10],  # Lowest IDF = most common
                'rarest_terms': idf_pairs[-10:],  # Highest IDF = rarest
                'doc_freq_stats': {
                    'mean': float(np.mean(doc_freq)),
                    'std': float(np.std(doc_freq)),
                    'min': int(np.min(doc_freq)),
                    'max': int(np.max(doc_freq))
                }
            }
            
            return stats
            
        except Exception as e:
            return {'error': str(e)}
    
    def _get_matched_terms(self, query_terms: List[str], doc_idx: int) -> List[str]:
        """Get terms that match between query and document."""
        if not query_terms:
            return []
        
        try:
            # Get document vector
            doc_vector = self.tfidf_matrix[doc_idx].toarray()[0]
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Find non-zero features in document
            doc_terms = set(feature_names[i] for i, score in enumerate(doc_vector) if score > 0)
            
            # Find intersection with query terms
            matched = []
            for term in query_terms:
                if term in doc_terms:
                    matched.append(term)
                # Also check if term is part of any n-gram in document
                for doc_term in doc_terms:
                    if term in doc_term and term not in matched:
                        matched.append(term)
            
            return list(set(matched))  # Remove duplicates
            
        except Exception as e:
            print(f"Error getting matched terms: {e}")
            return []
    
    def _update_search_stats(self, search_time: float, num_results: int):
        """Update search statistics."""
        self.search_stats['total_searches'] += 1
        self.search_stats['total_search_time'] += search_time
        self.search_stats['total_results'] += num_results
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get search statistics."""
        stats = dict(self.search_stats)
        
        if self.search_stats['total_searches'] > 0:
            stats['average_search_time'] = self.search_stats['total_search_time'] / self.search_stats['total_searches']
            stats['average_results_per_search'] = self.search_stats['total_results'] / self.search_stats['total_searches']
        else:
            stats['average_search_time'] = 0.0
            stats['average_results_per_search'] = 0.0
        
        return stats
    
    def save_index(self, filepath: str) -> bool:
        """Save the TF-IDF index to disk."""
        try:
            index_data = {
                'vectorizer': self.vectorizer,
                'tfidf_matrix': self.tfidf_matrix,
                'documents': self.documents,
                'chunk_metadata': self.chunk_metadata,
                'is_fitted': self.is_fitted,
                'search_stats': self.search_stats
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(index_data, f)
            
            print(f"TF-IDF index saved to {filepath}")
            return True
            
        except Exception as e:
            print(f"Error saving TF-IDF index: {e}")
            return False
    
    def load_index(self, filepath: str) -> bool:
        """Load TF-IDF index from disk."""
        try:
            with open(filepath, 'rb') as f:
                index_data = pickle.load(f)
            
            self.vectorizer = index_data['vectorizer']
            self.tfidf_matrix = index_data['tfidf_matrix']
            self.documents = index_data['documents']
            self.chunk_metadata = index_data['chunk_metadata']
            self.is_fitted = index_data['is_fitted']
            self.search_stats = index_data.get('search_stats', self.search_stats)
            
            print(f"TF-IDF index loaded from {filepath}")
            print(f"  - Documents: {len(self.documents)}")
            print(f"  - Vocabulary size: {len(self.vectorizer.vocabulary_) if self.vectorizer else 0}")
            
            return True
            
        except Exception as e:
            print(f"Error loading TF-IDF index: {e}")
            return False


if __name__ == "__main__":
    # Test the TF-IDF search engine
    print("TF-IDF Search Engine Test")
    print("=" * 50)
    
    # Create test data
    test_documents = [
        "Artificial intelligence is transforming the world through machine learning",
        "Machine learning algorithms can learn patterns from data automatically",
        "Deep learning uses neural networks with multiple layers",
        "Natural language processing helps computers understand human language",
        "Computer vision enables machines to interpret visual information",
        "Data science combines statistics and programming for insights",
        "The cat sat on the mat in the sunny garden"
    ]
    
    test_chunks = []
    for i, doc in enumerate(test_documents):
        chunk = {
            'chunk_id': f'test_chunk_{i}',
            'document_id': f'test_doc_{i}',
            'content': doc,
            'metadata': {'test': True}
        }
        test_chunks.append(chunk)
    
    # Initialize search engine
    tfidf_engine = TFIDFSearchEngine()
    
    # Build index
    success = tfidf_engine.build_index(test_chunks)
    print(f"Index built: {success}")
    
    if success:
        # Test search
        test_queries = [
            "machine learning algorithms",
            "neural networks",
            "artificial intelligence",
            "cat garden"
        ]
        
        print(f"\nTesting {len(test_queries)} queries:")
        
        for query in test_queries:
            result = tfidf_engine.search(query, top_k=3)
            print(f"\nQuery: '{query}'")
            print(f"  Processed: '{result.processed_query}'")
            print(f"  Results: {result.total_results}, Time: {result.search_time:.3f}s")
            
            for search_result in result.results:
                print(f"    {search_result.rank}. Score: {search_result.score:.3f}")
                print(f"       Content: {search_result.content[:100]}...")
                print(f"       Matched: {search_result.matched_terms}")
        
        # Get statistics
        stats = tfidf_engine.get_search_stats()
        vocab_stats = tfidf_engine.get_vocabulary_stats()
        
        print(f"\nSearch Statistics:")
        print(f"  Total searches: {stats['total_searches']}")
        print(f"  Average time: {stats['average_search_time']:.3f}s")
        print(f"  Average results: {stats['average_results_per_search']:.1f}")
        
        print(f"\nVocabulary Statistics:")
        print(f"  Vocabulary size: {vocab_stats['vocabulary_size']}")
        print(f"  Matrix density: {vocab_stats['matrix_density']:.4f}")
        print(f"  Most common terms: {vocab_stats['most_common_terms'][:5]}")