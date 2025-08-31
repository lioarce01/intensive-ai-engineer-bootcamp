#!/usr/bin/env python3
"""
Enhanced Embedding Engine for Semantic Search
=============================================

This module extends the existing word embedding capabilities with document-level and 
chunk-level embeddings. It integrates both custom Word2Vec models and modern 
sentence transformers for comparison and educational purposes.

Features:
- Integration with existing Word2Vec models from word embeddings module
- Sentence-Transformers for state-of-the-art embeddings
- Multiple embedding strategies (word averaging, sentence transformers)
- Embedding caching and batch processing
- Similarity computation and ranking
- Educational comparisons between different embedding approaches

Author: AI Bootcamp Week 3-4
Date: 2025
"""

import os
import sys
import pickle
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Core libraries
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Import existing word embeddings components
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '02-word-embeddings'))

try:
    from word_embeddings_comprehensive import (
        EmbeddingAnalyzer, 
        SemanticSearchEngine,
        TextPreprocessor,
        SkipGramModel
    )
    WORD_EMBEDDINGS_AVAILABLE = True
except ImportError:
    WORD_EMBEDDINGS_AVAILABLE = False
    print("Warning: Could not import from word embeddings module. Some features will be limited.")

# Import document processing components
try:
    from document_processor import Document, DocumentChunk
    DOCUMENT_PROCESSOR_AVAILABLE = True
except ImportError:
    DOCUMENT_PROCESSOR_AVAILABLE = False
    print("Warning: Could not import document processor. Using minimal fallbacks.")

# Sentence Transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Install with: pip install sentence-transformers")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


@dataclass
class EmbeddingResult:
    """Container for embedding results with metadata."""
    embedding: np.ndarray
    model_name: str
    embedding_dim: int
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'embedding': self.embedding.tolist(),
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmbeddingResult':
        """Create from dictionary."""
        return cls(
            embedding=np.array(data['embedding']),
            model_name=data['model_name'],
            embedding_dim=data['embedding_dim'],
            metadata=data['metadata']
        )


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""
    
    @abstractmethod
    def encode(self, texts: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """Encode text(s) into embeddings."""
        pass
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Get the dimensionality of embeddings produced by this model."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the name/identifier of this model."""
        pass


class Word2VecAveragingModel(EmbeddingModel):
    """Embedding model that averages Word2Vec embeddings for sentences/documents."""
    
    def __init__(self, word2vec_model: Optional[SkipGramModel] = None, 
                 analyzer: Optional[EmbeddingAnalyzer] = None,
                 preprocessor: Optional[TextPreprocessor] = None):
        """
        Initialize with existing Word2Vec model components.
        
        Args:
            word2vec_model: Trained Word2Vec model
            analyzer: EmbeddingAnalyzer instance
            preprocessor: TextPreprocessor instance
        """
        self.word2vec_model = word2vec_model
        self.analyzer = analyzer
        self.preprocessor = preprocessor
        
        if not WORD_EMBEDDINGS_AVAILABLE:
            print("Warning: Word embeddings module not available. Word2Vec features limited.")
    
    def encode(self, texts: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Encode text(s) by averaging Word2Vec embeddings.
        
        Args:
            texts: Single text string or list of text strings
        
        Returns:
            Numpy array(s) of embeddings
        """
        if isinstance(texts, str):
            return self._encode_single(texts)
        else:
            return [self._encode_single(text) for text in texts]
    
    def _encode_single(self, text: str) -> np.ndarray:
        """Encode a single text by averaging word embeddings."""
        if not self.analyzer or not self.preprocessor:
            # Fallback to random embedding if components not available
            return np.random.randn(100) * 0.1
        
        # Clean and tokenize text
        clean_text = self.preprocessor.clean_text(text)
        words = clean_text.split()
        
        # Collect embeddings for words in vocabulary
        embeddings = []
        for word in words:
            embedding = self.analyzer.get_embedding(word)
            if embedding is not None:
                embeddings.append(embedding.numpy())
        
        if not embeddings:
            # Return zero vector if no words found in vocabulary
            return np.zeros(self.get_embedding_dim())
        
        # Average the embeddings
        avg_embedding = np.mean(embeddings, axis=0)
        return avg_embedding
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        if self.word2vec_model:
            return self.word2vec_model.embedding_dim
        return 100  # Default fallback
    
    def get_model_name(self) -> str:
        """Get model name."""
        return "Word2Vec-Averaging"


class SentenceTransformerModel(EmbeddingModel):
    """Wrapper for Sentence-Transformers models."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize Sentence-Transformers model.
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
                self.available = True
                print(f"Loaded SentenceTransformer model: {model_name}")
            except Exception as e:
                print(f"Error loading SentenceTransformer {model_name}: {e}")
                self.model = None
                self.available = False
        else:
            print("SentenceTransformers not available")
            self.model = None
            self.available = False
    
    def encode(self, texts: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """Encode texts using sentence transformers."""
        if not self.available:
            # Fallback to random embeddings
            if isinstance(texts, str):
                return np.random.randn(384) * 0.1  # MiniLM dimension
            else:
                return [np.random.randn(384) * 0.1 for _ in texts]
        
        # Use sentence transformer
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        if isinstance(texts, str):
            return embeddings
        else:
            return list(embeddings)
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        if self.available:
            return self.model.get_sentence_embedding_dimension()
        return 384  # Default MiniLM dimension
    
    def get_model_name(self) -> str:
        """Get model name."""
        return f"SentenceTransformer-{self.model_name}"


class HybridEmbeddingModel(EmbeddingModel):
    """Model that combines multiple embedding approaches."""
    
    def __init__(self, models: List[EmbeddingModel], weights: Optional[List[float]] = None):
        """
        Initialize hybrid model.
        
        Args:
            models: List of embedding models to combine
            weights: Optional weights for each model (defaults to equal weighting)
        """
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        
        if len(self.weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")
    
    def encode(self, texts: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """Encode using weighted combination of models."""
        # Get embeddings from all models
        all_embeddings = []
        for model in self.models:
            embeddings = model.encode(texts)
            all_embeddings.append(embeddings)
        
        # Combine embeddings
        if isinstance(texts, str):
            # Single text
            combined = np.concatenate([emb * weight for emb, weight in zip(all_embeddings, self.weights)])
            return combined
        else:
            # Multiple texts
            combined_list = []
            for i in range(len(texts)):
                combined = np.concatenate([emb[i] * weight for emb, weight in zip(all_embeddings, self.weights)])
                combined_list.append(combined)
            return combined_list
    
    def get_embedding_dim(self) -> int:
        """Get combined embedding dimension."""
        return sum(model.get_embedding_dim() for model in self.models)
    
    def get_model_name(self) -> str:
        """Get combined model name."""
        model_names = [model.get_model_name() for model in self.models]
        return f"Hybrid-{'-'.join(model_names)}"


class EmbeddingCache:
    """Cache for storing and retrieving embeddings to avoid recomputation."""
    
    def __init__(self, cache_file: Optional[str] = None):
        """
        Initialize embedding cache.
        
        Args:
            cache_file: Optional file to persist cache to disk
        """
        self.cache = {}
        self.cache_file = cache_file
        self.hits = 0
        self.misses = 0
        
        # Load existing cache if file exists
        if cache_file and os.path.exists(cache_file):
            self.load_cache()
    
    def _get_cache_key(self, text: str, model_name: str) -> str:
        """Generate cache key for text and model."""
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"{model_name}:{text_hash}"
    
    def get(self, text: str, model_name: str) -> Optional[np.ndarray]:
        """Get embedding from cache if available."""
        key = self._get_cache_key(text, model_name)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        else:
            self.misses += 1
            return None
    
    def put(self, text: str, model_name: str, embedding: np.ndarray):
        """Store embedding in cache."""
        key = self._get_cache_key(text, model_name)
        self.cache[key] = embedding.copy()
        
        # Periodically save to disk
        if self.cache_file and len(self.cache) % 100 == 0:
            self.save_cache()
    
    def save_cache(self):
        """Save cache to disk."""
        if not self.cache_file:
            return
        
        try:
            # Convert numpy arrays to lists for JSON serialization
            serializable_cache = {}
            for key, embedding in self.cache.items():
                serializable_cache[key] = embedding.tolist()
            
            with open(self.cache_file, 'w') as f:
                json.dump(serializable_cache, f)
            
            print(f"Saved embedding cache with {len(self.cache)} entries to {self.cache_file}")
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def load_cache(self):
        """Load cache from disk."""
        if not self.cache_file or not os.path.exists(self.cache_file):
            return
        
        try:
            with open(self.cache_file, 'r') as f:
                serializable_cache = json.load(f)
            
            # Convert lists back to numpy arrays
            for key, embedding_list in serializable_cache.items():
                self.cache[key] = np.array(embedding_list)
            
            print(f"Loaded embedding cache with {len(self.cache)} entries from {self.cache_file}")
        except Exception as e:
            print(f"Error loading cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_size': len(self.cache),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate
        }


class EnhancedEmbeddingEngine:
    """
    Main embedding engine that orchestrates multiple embedding models
    and provides high-level functionality for document embedding.
    """
    
    def __init__(self, models: Optional[Dict[str, EmbeddingModel]] = None,
                 cache_file: Optional[str] = None):
        """
        Initialize enhanced embedding engine.
        
        Args:
            models: Dictionary of embedding models by name
            cache_file: Optional file for embedding cache
        """
        # Initialize cache
        self.cache = EmbeddingCache(cache_file)
        
        # Initialize models
        if models is None:
            models = self._create_default_models()
        
        self.models = models
        print(f"Initialized embedding engine with {len(self.models)} models:")
        for name, model in self.models.items():
            print(f"  - {name}: {model.get_embedding_dim()}D")
    
    def _create_default_models(self) -> Dict[str, EmbeddingModel]:
        """Create default set of embedding models."""
        models = {}
        
        # Add Word2Vec averaging model if available
        if WORD_EMBEDDINGS_AVAILABLE:
            try:
                # Try to create a basic Word2Vec model (would need to be pre-trained)
                word2vec_model = Word2VecAveragingModel()
                models['word2vec_avg'] = word2vec_model
            except Exception as e:
                print(f"Could not initialize Word2Vec model: {e}")
        
        # Add Sentence Transformer model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            st_model = SentenceTransformerModel('all-MiniLM-L6-v2')
            if st_model.available:
                models['sentence_transformer'] = st_model
        
        # Fallback: create a dummy model if no real models available
        if not models:
            print("No embedding models available, creating dummy model for testing")
            models['dummy'] = DummyEmbeddingModel()
        
        return models
    
    def embed_text(self, text: str, model_name: str = None) -> EmbeddingResult:
        """
        Embed a single text using specified model.
        
        Args:
            text: Text to embed
            model_name: Name of model to use (uses first available if None)
        
        Returns:
            EmbeddingResult containing embedding and metadata
        """
        if model_name is None:
            model_name = next(iter(self.models.keys()))
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not available. Available models: {list(self.models.keys())}")
        
        model = self.models[model_name]
        
        # Check cache first
        cached_embedding = self.cache.get(text, model_name)
        if cached_embedding is not None:
            return EmbeddingResult(
                embedding=cached_embedding,
                model_name=model_name,
                embedding_dim=model.get_embedding_dim(),
                metadata={'from_cache': True}
            )
        
        # Compute embedding
        embedding = model.encode(text)
        
        # Store in cache
        self.cache.put(text, model_name, embedding)
        
        return EmbeddingResult(
            embedding=embedding,
            model_name=model_name,
            embedding_dim=model.get_embedding_dim(),
            metadata={'from_cache': False}
        )
    
    def embed_documents(self, documents: List[Union[str, Document, DocumentChunk]], 
                       model_name: str = None,
                       batch_size: int = 32) -> List[EmbeddingResult]:
        """
        Embed multiple documents/chunks efficiently.
        
        Args:
            documents: List of texts, Documents, or DocumentChunks
            model_name: Name of model to use
            batch_size: Batch size for processing
        
        Returns:
            List of EmbeddingResult objects
        """
        if model_name is None:
            model_name = next(iter(self.models.keys()))
        
        model = self.models[model_name]
        
        # Extract text content
        texts = []
        for doc in documents:
            if isinstance(doc, str):
                texts.append(doc)
            elif hasattr(doc, 'content'):  # Document or DocumentChunk
                texts.append(doc.content)
            else:
                texts.append(str(doc))
        
        # Check cache for each text
        results = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            cached_embedding = self.cache.get(text, model_name)
            if cached_embedding is not None:
                results.append(EmbeddingResult(
                    embedding=cached_embedding,
                    model_name=model_name,
                    embedding_dim=model.get_embedding_dim(),
                    metadata={'from_cache': True, 'batch_index': i}
                ))
            else:
                results.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Process uncached texts in batches
        if uncached_texts:
            for batch_start in range(0, len(uncached_texts), batch_size):
                batch_end = min(batch_start + batch_size, len(uncached_texts))
                batch_texts = uncached_texts[batch_start:batch_end]
                
                # Compute embeddings for batch
                batch_embeddings = model.encode(batch_texts)
                if not isinstance(batch_embeddings, list):
                    batch_embeddings = [batch_embeddings]
                
                # Store results and cache
                for j, embedding in enumerate(batch_embeddings):
                    text_index = batch_start + j
                    original_index = uncached_indices[text_index]
                    text = batch_texts[j]
                    
                    # Store in cache
                    self.cache.put(text, model_name, embedding)
                    
                    # Store result
                    results[original_index] = EmbeddingResult(
                        embedding=embedding,
                        model_name=model_name,
                        embedding_dim=model.get_embedding_dim(),
                        metadata={'from_cache': False, 'batch_index': original_index}
                    )
        
        return results
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray, 
                          metric: str = 'cosine') -> float:
        """
        Compute similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            metric: Similarity metric ('cosine', 'euclidean', 'dot')
        
        Returns:
            Similarity score
        """
        if metric == 'cosine':
            # Cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return dot_product / (norm1 * norm2)
        
        elif metric == 'euclidean':
            # Euclidean distance (converted to similarity)
            distance = np.linalg.norm(embedding1 - embedding2)
            return 1.0 / (1.0 + distance)
        
        elif metric == 'dot':
            # Dot product
            return np.dot(embedding1, embedding2)
        
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")
    
    def find_similar(self, query_embedding: np.ndarray, 
                    candidate_embeddings: List[np.ndarray],
                    top_k: int = 5, metric: str = 'cosine') -> List[Tuple[int, float]]:
        """
        Find most similar embeddings to a query.
        
        Args:
            query_embedding: Query embedding
            candidate_embeddings: List of candidate embeddings
            top_k: Number of top results to return
            metric: Similarity metric
        
        Returns:
            List of (index, similarity_score) tuples, sorted by similarity
        """
        similarities = []
        for i, candidate in enumerate(candidate_embeddings):
            sim = self.compute_similarity(query_embedding, candidate, metric)
            similarities.append((i, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def compare_models(self, texts: List[str]) -> Dict[str, Any]:
        """
        Compare different embedding models on a set of texts.
        
        Args:
            texts: List of texts to embed with all models
        
        Returns:
            Comparison results with embeddings and analysis
        """
        results = {}
        
        for model_name in self.models.keys():
            print(f"Embedding with {model_name}...")
            
            model_results = self.embed_documents(texts, model_name)
            embeddings = [result.embedding for result in model_results]
            
            # Compute some basic statistics
            embedding_matrix = np.array(embeddings)
            
            stats = {
                'embedding_dim': model_results[0].embedding_dim,
                'mean_norm': np.mean([np.linalg.norm(emb) for emb in embeddings]),
                'std_norm': np.std([np.linalg.norm(emb) for emb in embeddings]),
                'embeddings': embeddings
            }
            
            # Compute pairwise similarities
            if len(embeddings) > 1:
                similarities = []
                for i in range(len(embeddings)):
                    for j in range(i + 1, len(embeddings)):
                        sim = self.compute_similarity(embeddings[i], embeddings[j])
                        similarities.append(sim)
                
                stats['mean_similarity'] = np.mean(similarities)
                stats['std_similarity'] = np.std(similarities)
                stats['pairwise_similarities'] = similarities
            
            results[model_name] = stats
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        cache_stats = self.cache.get_stats()
        
        model_info = {}
        for name, model in self.models.items():
            model_info[name] = {
                'model_class': model.__class__.__name__,
                'embedding_dim': model.get_embedding_dim(),
                'model_name': model.get_model_name()
            }
        
        return {
            'num_models': len(self.models),
            'models': model_info,
            'cache_stats': cache_stats
        }


class DummyEmbeddingModel(EmbeddingModel):
    """Dummy embedding model for testing when no real models are available."""
    
    def __init__(self, dim: int = 100):
        self.dim = dim
    
    def encode(self, texts: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """Generate random embeddings."""
        if isinstance(texts, str):
            return np.random.randn(self.dim) * 0.1
        else:
            return [np.random.randn(self.dim) * 0.1 for _ in texts]
    
    def get_embedding_dim(self) -> int:
        return self.dim
    
    def get_model_name(self) -> str:
        return "Dummy"


def create_embedding_engine(use_word2vec: bool = True, 
                           use_sentence_transformers: bool = True,
                           cache_file: Optional[str] = None) -> EnhancedEmbeddingEngine:
    """
    Create an embedding engine with specified models.
    
    Args:
        use_word2vec: Whether to include Word2Vec averaging model
        use_sentence_transformers: Whether to include Sentence Transformers
        cache_file: Optional cache file path
    
    Returns:
        Configured EnhancedEmbeddingEngine
    """
    models = {}
    
    if use_word2vec and WORD_EMBEDDINGS_AVAILABLE:
        models['word2vec_avg'] = Word2VecAveragingModel()
    
    if use_sentence_transformers and SENTENCE_TRANSFORMERS_AVAILABLE:
        st_model = SentenceTransformerModel('all-MiniLM-L6-v2')
        if st_model.available:
            models['sentence_transformer'] = st_model
    
    # Add hybrid model if multiple models available
    if len(models) > 1:
        hybrid_models = list(models.values())
        models['hybrid'] = HybridEmbeddingModel(hybrid_models)
    
    # Fallback to dummy if no models
    if not models:
        models['dummy'] = DummyEmbeddingModel()
    
    return EnhancedEmbeddingEngine(models, cache_file)


if __name__ == "__main__":
    # Test the embedding engine
    print("Enhanced Embedding Engine Test")
    print("=" * 50)
    
    # Create engine
    engine = create_embedding_engine(cache_file="test_cache.json")
    
    # Test texts
    test_texts = [
        "Artificial intelligence is transforming the world",
        "Machine learning algorithms can learn from data",
        "Deep learning uses neural networks",
        "Natural language processing helps computers understand text",
        "The cat sat on the mat"
    ]
    
    print(f"Testing with {len(test_texts)} texts")
    print(f"Available models: {list(engine.models.keys())}")
    print()
    
    # Test single embedding
    model_name = next(iter(engine.models.keys()))
    result = engine.embed_text(test_texts[0], model_name)
    print(f"Single embedding result:")
    print(f"  Model: {result.model_name}")
    print(f"  Dimension: {result.embedding_dim}")
    print(f"  Embedding shape: {result.embedding.shape}")
    print(f"  From cache: {result.metadata.get('from_cache', False)}")
    print()
    
    # Test batch embedding
    batch_results = engine.embed_documents(test_texts, model_name)
    print(f"Batch embedding results:")
    print(f"  Number of results: {len(batch_results)}")
    print(f"  Cache hits: {sum(1 for r in batch_results if r.metadata.get('from_cache', False))}")
    print()
    
    # Test similarity
    emb1 = batch_results[0].embedding
    emb2 = batch_results[1].embedding
    emb3 = batch_results[4].embedding  # "The cat sat on the mat"
    
    sim_ai_ml = engine.compute_similarity(emb1, emb2)
    sim_ai_cat = engine.compute_similarity(emb1, emb3)
    
    print(f"Similarity tests:")
    print(f"  AI vs ML: {sim_ai_ml:.3f}")
    print(f"  AI vs Cat: {sim_ai_cat:.3f}")
    print()
    
    # Test model comparison if multiple models
    if len(engine.models) > 1:
        print("Model comparison:")
        comparison = engine.compare_models(test_texts[:3])
        for model_name, stats in comparison.items():
            print(f"  {model_name}:")
            print(f"    Embedding dim: {stats['embedding_dim']}")
            print(f"    Mean norm: {stats['mean_norm']:.3f}")
            if 'mean_similarity' in stats:
                print(f"    Mean pairwise similarity: {stats['mean_similarity']:.3f}")
    
    # Show statistics
    print()
    print("Engine statistics:")
    stats = engine.get_stats()
    print(f"  Models: {stats['num_models']}")
    print(f"  Cache size: {stats['cache_stats']['cache_size']}")
    print(f"  Cache hit rate: {stats['cache_stats']['hit_rate']:.2%}")