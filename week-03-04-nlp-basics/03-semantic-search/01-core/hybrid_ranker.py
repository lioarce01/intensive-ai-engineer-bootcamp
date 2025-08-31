#!/usr/bin/env python3
"""
Hybrid Search Ranker
====================

This module combines TF-IDF and semantic search results using various
fusion strategies. It provides educational comparison metrics and
demonstrates the strengths of different search approaches.

Features:
- Multiple score fusion strategies (linear, RRF, CombSUM, CombMNZ)
- Score normalization techniques
- Result deduplication and merging
- Performance comparison metrics
- Educational analysis of search approaches
- Configurable weighting strategies

Author: AI Bootcamp Week 3-4
Date: 2025
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any, Set
from dataclasses import dataclass
import time
from datetime import datetime
import math
from collections import defaultdict

# Core libraries
import numpy as np
from scipy import stats

# Import components
try:
    from tfidf_search import TFIDFSearchEngine, TFIDFResult, TFIDFQueryResult
    from search_engine import SemanticSearchEngine, SearchResult, QueryResult
    COMPONENTS_AVAILABLE = True
except ImportError:
    COMPONENTS_AVAILABLE = False
    print("Warning: Could not import search engine components. Some features will be limited.")


@dataclass
class HybridResult:
    """Result from hybrid search combining multiple approaches."""
    chunk_id: str
    document_id: str
    content: str
    final_score: float
    rank: int
    scores: Dict[str, float]  # Scores from different methods
    source_ranks: Dict[str, int]  # Original ranks from different methods
    matched_terms: List[str]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'chunk_id': self.chunk_id,
            'document_id': self.document_id,
            'content': self.content,
            'final_score': self.final_score,
            'rank': self.rank,
            'scores': self.scores,
            'source_ranks': self.source_ranks,
            'matched_terms': self.matched_terms,
            'metadata': self.metadata
        }


@dataclass
class HybridQueryResult:
    """Complete hybrid search result for a query."""
    query: str
    fusion_strategy: str
    search_time: float
    total_results: int
    results: List[HybridResult]
    component_results: Dict[str, Any]  # Results from individual methods
    fusion_metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'query': self.query,
            'fusion_strategy': self.fusion_strategy,
            'search_time': self.search_time,
            'total_results': self.total_results,
            'results': [result.to_dict() for result in self.results],
            'component_results': self.component_results,
            'fusion_metadata': self.fusion_metadata
        }
    
    def get_top_k(self, k: int) -> List[HybridResult]:
        """Get top K hybrid results."""
        return self.results[:k]


class ScoreNormalizer:
    """Utility class for normalizing scores from different search methods."""
    
    @staticmethod
    def min_max_normalize(scores: List[float]) -> List[float]:
        """Min-max normalization to [0, 1] range."""
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return [(score - min_score) / (max_score - min_score) for score in scores]
    
    @staticmethod
    def z_score_normalize(scores: List[float]) -> List[float]:
        """Z-score normalization (mean=0, std=1)."""
        if not scores or len(scores) == 1:
            return [0.0] * len(scores)
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        if std_score == 0:
            return [0.0] * len(scores)
        
        return [(score - mean_score) / std_score for score in scores]
    
    @staticmethod
    def rank_normalize(scores: List[float], method: str = 'reciprocal') -> List[float]:
        """Rank-based normalization."""
        if not scores:
            return []
        
        # Get ranks (1 = highest score)
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        ranks = [0] * len(scores)
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = rank + 1
        
        if method == 'reciprocal':
            # Reciprocal rank: 1/rank
            return [1.0 / rank for rank in ranks]
        elif method == 'linear':
            # Linear decay: (max_rank - rank + 1) / max_rank
            max_rank = len(scores)
            return [(max_rank - rank + 1) / max_rank for rank in ranks]
        else:
            raise ValueError(f"Unknown rank normalization method: {method}")
    
    @staticmethod
    def sigmoid_normalize(scores: List[float], center: float = 0.5) -> List[float]:
        """Sigmoid normalization."""
        if not scores:
            return []
        
        # Shift scores so center maps to 0
        shifted_scores = [score - center for score in scores]
        return [1.0 / (1.0 + math.exp(-score)) for score in shifted_scores]


class HybridRanker:
    """
    Hybrid ranking system that combines TF-IDF and semantic search results.
    """
    
    def __init__(self, tfidf_engine: TFIDFSearchEngine, semantic_engine: SemanticSearchEngine):
        """
        Initialize hybrid ranker.
        
        Args:
            tfidf_engine: TF-IDF search engine
            semantic_engine: Semantic search engine
        """
        self.tfidf_engine = tfidf_engine
        self.semantic_engine = semantic_engine
        
        # Search statistics
        self.search_stats = {
            'total_searches': 0,
            'fusion_strategies_used': defaultdict(int),
            'total_search_time': 0.0,
            'average_fusion_gain': 0.0
        }
        
        print("Hybrid Ranker initialized with TF-IDF and Semantic search engines")
    
    def search(self, query: str, fusion_strategy: str = 'linear', 
              semantic_model: str = 'sentence_transformer',
              semantic_weight: float = 0.6, tfidf_weight: float = 0.4,
              top_k: int = 10, normalization: str = 'min_max',
              semantic_top_k: int = 20, tfidf_top_k: int = 20) -> HybridQueryResult:
        """
        Perform hybrid search combining TF-IDF and semantic approaches.
        
        Args:
            query: Search query
            fusion_strategy: Strategy for combining scores ('linear', 'rrf', 'combsum', 'combmnz')
            semantic_model: Semantic model to use
            semantic_weight: Weight for semantic scores
            tfidf_weight: Weight for TF-IDF scores
            top_k: Final number of results to return
            normalization: Score normalization method
            semantic_top_k: Number of semantic results to retrieve
            tfidf_top_k: Number of TF-IDF results to retrieve
        
        Returns:
            HybridQueryResult with combined results
        """
        start_time = time.time()
        
        try:
            # Get results from both engines
            print(f"Hybrid search for: '{query}'")
            
            # Semantic search
            from search_engine import SearchConfig
            semantic_config = SearchConfig(
                model_name=semantic_model,
                top_k=semantic_top_k,
                score_threshold=0.0
            )
            semantic_result = self.semantic_engine.search(query, semantic_config)
            
            # TF-IDF search
            tfidf_result = self.tfidf_engine.search(query, top_k=tfidf_top_k, min_score=0.0)
            
            print(f"  Semantic: {semantic_result.total_results} results")
            print(f"  TF-IDF: {tfidf_result.total_results} results")
            
            # Combine results using specified strategy
            hybrid_results = self._fuse_results(
                semantic_result, tfidf_result, fusion_strategy,
                semantic_weight, tfidf_weight, normalization, top_k
            )
            
            search_time = time.time() - start_time
            
            # Update statistics
            self._update_search_stats(fusion_strategy, search_time)
            
            # Create hybrid query result
            result = HybridQueryResult(
                query=query,
                fusion_strategy=fusion_strategy,
                search_time=search_time,
                total_results=len(hybrid_results),
                results=hybrid_results,
                component_results={
                    'semantic': {
                        'model': semantic_model,
                        'total_results': semantic_result.total_results,
                        'search_time': semantic_result.search_time,
                        'top_scores': [r.score for r in semantic_result.results[:5]]
                    },
                    'tfidf': {
                        'total_results': tfidf_result.total_results,
                        'search_time': tfidf_result.search_time,
                        'processed_query': tfidf_result.processed_query,
                        'query_terms': tfidf_result.query_terms,
                        'top_scores': [r.score for r in tfidf_result.results[:5]]
                    }
                },
                fusion_metadata={
                    'semantic_weight': semantic_weight,
                    'tfidf_weight': tfidf_weight,
                    'normalization': normalization,
                    'semantic_top_k': semantic_top_k,
                    'tfidf_top_k': tfidf_top_k
                }
            )
            
            return result
            
        except Exception as e:
            print(f"Error during hybrid search: {e}")
            return HybridQueryResult(
                query=query,
                fusion_strategy=fusion_strategy,
                search_time=time.time() - start_time,
                total_results=0,
                results=[],
                component_results={},
                fusion_metadata={'error': str(e)}
            )
    
    def _fuse_results(self, semantic_result: QueryResult, tfidf_result: TFIDFQueryResult,
                     fusion_strategy: str, semantic_weight: float, tfidf_weight: float,
                     normalization: str, top_k: int) -> List[HybridResult]:
        """Fuse results from semantic and TF-IDF searches."""
        
        # Create mappings of chunk_id to results
        semantic_results = {r.chunk_id: r for r in semantic_result.results}
        tfidf_results = {r.chunk_id: r for r in tfidf_result.results}
        
        # Get all unique chunk IDs
        all_chunk_ids = set(semantic_results.keys()) | set(tfidf_results.keys())
        
        if not all_chunk_ids:
            return []
        
        # Collect scores for normalization
        semantic_scores = [r.score for r in semantic_result.results]
        tfidf_scores = [r.score for r in tfidf_result.results]
        
        # Normalize scores
        if semantic_scores:
            semantic_scores_norm = self._normalize_scores(semantic_scores, normalization)
            semantic_score_map = {r.chunk_id: norm_score 
                                for r, norm_score in zip(semantic_result.results, semantic_scores_norm)}
        else:
            semantic_score_map = {}
        
        if tfidf_scores:
            tfidf_scores_norm = self._normalize_scores(tfidf_scores, normalization)
            tfidf_score_map = {r.chunk_id: norm_score 
                              for r, norm_score in zip(tfidf_result.results, tfidf_scores_norm)}
        else:
            tfidf_score_map = {}
        
        # Fuse scores for each chunk
        fused_results = []
        
        for chunk_id in all_chunk_ids:
            # Get normalized scores (0 if not found)
            semantic_score = semantic_score_map.get(chunk_id, 0.0)
            tfidf_score = tfidf_score_map.get(chunk_id, 0.0)
            
            # Get original ranks
            semantic_rank = next((i+1 for i, r in enumerate(semantic_result.results) 
                                if r.chunk_id == chunk_id), 0)
            tfidf_rank = next((i+1 for i, r in enumerate(tfidf_result.results) 
                             if r.chunk_id == chunk_id), 0)
            
            # Apply fusion strategy
            if fusion_strategy == 'linear':
                final_score = semantic_weight * semantic_score + tfidf_weight * tfidf_score
            
            elif fusion_strategy == 'rrf':  # Reciprocal Rank Fusion
                k = 60  # RRF parameter
                semantic_rrf = 1.0 / (k + semantic_rank) if semantic_rank > 0 else 0.0
                tfidf_rrf = 1.0 / (k + tfidf_rank) if tfidf_rank > 0 else 0.0
                final_score = semantic_weight * semantic_rrf + tfidf_weight * tfidf_rrf
            
            elif fusion_strategy == 'combsum':
                final_score = semantic_score + tfidf_score
            
            elif fusion_strategy == 'combmnz':
                # CombMNZ: sum of scores * number of non-zero scores
                num_nonzero = int(semantic_score > 0) + int(tfidf_score > 0)
                final_score = (semantic_score + tfidf_score) * num_nonzero
            
            else:
                raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
            
            # Get content and metadata (prefer semantic result, fallback to TF-IDF)
            if chunk_id in semantic_results:
                base_result = semantic_results[chunk_id]
                content = base_result.content
                document_id = base_result.document_id
                metadata = base_result.metadata
            else:
                base_result = tfidf_results[chunk_id]
                content = base_result.content
                document_id = base_result.document_id
                metadata = base_result.metadata
            
            # Get matched terms (combine from both sources)
            matched_terms = []
            if chunk_id in semantic_results:
                matched_terms.extend(getattr(semantic_results[chunk_id], 'matched_terms', []))
            if chunk_id in tfidf_results:
                matched_terms.extend(tfidf_results[chunk_id].matched_terms)
            matched_terms = list(set(matched_terms))  # Remove duplicates
            
            # Create hybrid result
            hybrid_result = HybridResult(
                chunk_id=chunk_id,
                document_id=document_id,
                content=content,
                final_score=final_score,
                rank=0,  # Will be set after sorting
                scores={
                    'semantic': semantic_score,
                    'tfidf': tfidf_score,
                    'semantic_raw': semantic_results[chunk_id].score if chunk_id in semantic_results else 0.0,
                    'tfidf_raw': tfidf_results[chunk_id].score if chunk_id in tfidf_results else 0.0
                },
                source_ranks={
                    'semantic': semantic_rank,
                    'tfidf': tfidf_rank
                },
                matched_terms=matched_terms,
                metadata=metadata
            )
            
            fused_results.append(hybrid_result)
        
        # Sort by final score and assign ranks
        fused_results.sort(key=lambda x: x.final_score, reverse=True)
        
        for rank, result in enumerate(fused_results):
            result.rank = rank + 1
        
        # Return top K results
        return fused_results[:top_k]
    
    def _normalize_scores(self, scores: List[float], method: str) -> List[float]:
        """Normalize scores using specified method."""
        if method == 'min_max':
            return ScoreNormalizer.min_max_normalize(scores)
        elif method == 'z_score':
            return ScoreNormalizer.z_score_normalize(scores)
        elif method == 'rank_reciprocal':
            return ScoreNormalizer.rank_normalize(scores, 'reciprocal')
        elif method == 'rank_linear':
            return ScoreNormalizer.rank_normalize(scores, 'linear')
        elif method == 'sigmoid':
            return ScoreNormalizer.sigmoid_normalize(scores)
        elif method == 'none':
            return scores
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def compare_strategies(self, query: str, semantic_model: str = 'sentence_transformer',
                          top_k: int = 10) -> Dict[str, HybridQueryResult]:
        """
        Compare different fusion strategies for a query.
        
        Args:
            query: Search query
            semantic_model: Semantic model to use
            top_k: Number of results per strategy
        
        Returns:
            Dictionary of strategy -> HybridQueryResult
        """
        strategies = ['linear', 'rrf', 'combsum', 'combmnz']
        results = {}
        
        print(f"Comparing fusion strategies for: '{query}'")
        
        for strategy in strategies:
            try:
                result = self.search(
                    query, fusion_strategy=strategy, semantic_model=semantic_model,
                    top_k=top_k
                )
                results[strategy] = result
                print(f"  {strategy}: {result.total_results} results, {result.search_time:.3f}s")
                
            except Exception as e:
                print(f"  Error with {strategy}: {e}")
                continue
        
        return results
    
    def analyze_result_overlap(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Analyze overlap between semantic and TF-IDF results.
        
        Args:
            query: Search query
            top_k: Number of top results to analyze
        
        Returns:
            Analysis of result overlap and differences
        """
        try:
            # Get individual results
            semantic_config = SearchConfig(top_k=top_k)
            semantic_result = self.semantic_engine.search(query, semantic_config)
            tfidf_result = self.tfidf_engine.search(query, top_k=top_k)
            
            # Extract chunk IDs
            semantic_chunks = {r.chunk_id for r in semantic_result.results}
            tfidf_chunks = {r.chunk_id for r in tfidf_result.results}
            
            # Calculate overlap metrics
            intersection = semantic_chunks & tfidf_chunks
            union = semantic_chunks | tfidf_chunks
            
            semantic_only = semantic_chunks - tfidf_chunks
            tfidf_only = tfidf_chunks - semantic_chunks
            
            # Jaccard similarity
            jaccard = len(intersection) / len(union) if union else 0.0
            
            # Rank correlation for common results
            rank_correlation = 0.0
            if len(intersection) > 1:
                semantic_ranks = []
                tfidf_ranks = []
                
                for chunk_id in intersection:
                    sem_rank = next(i+1 for i, r in enumerate(semantic_result.results) 
                                  if r.chunk_id == chunk_id)
                    tfidf_rank = next(i+1 for i, r in enumerate(tfidf_result.results) 
                                    if r.chunk_id == chunk_id)
                    semantic_ranks.append(sem_rank)
                    tfidf_ranks.append(tfidf_rank)
                
                if len(semantic_ranks) > 1:
                    rank_correlation, _ = stats.spearmanr(semantic_ranks, tfidf_ranks)
                    if math.isnan(rank_correlation):
                        rank_correlation = 0.0
            
            analysis = {
                'query': query,
                'semantic_results': len(semantic_chunks),
                'tfidf_results': len(tfidf_chunks),
                'overlap_count': len(intersection),
                'semantic_only_count': len(semantic_only),
                'tfidf_only_count': len(tfidf_only),
                'jaccard_similarity': jaccard,
                'rank_correlation': rank_correlation,
                'overlap_details': {
                    'common_chunks': list(intersection),
                    'semantic_only': list(semantic_only),
                    'tfidf_only': list(tfidf_only)
                }
            }
            
            return analysis
            
        except Exception as e:
            return {'error': str(e)}
    
    def evaluate_fusion_effectiveness(self, queries: List[str], 
                                    ground_truth: Optional[Dict[str, Set[str]]] = None) -> Dict[str, Any]:
        """
        Evaluate effectiveness of different fusion strategies.
        
        Args:
            queries: List of test queries
            ground_truth: Optional ground truth relevance judgments
        
        Returns:
            Evaluation metrics for different strategies
        """
        strategies = ['linear', 'rrf', 'combsum', 'combmnz']
        evaluation = {
            'strategies': {},
            'queries_tested': len(queries),
            'overlap_analysis': []
        }
        
        print(f"Evaluating fusion effectiveness on {len(queries)} queries...")
        
        # Test each strategy
        for strategy in strategies:
            strategy_results = []
            total_time = 0.0
            
            for query in queries:
                try:
                    result = self.search(query, fusion_strategy=strategy, top_k=10)
                    strategy_results.append(result)
                    total_time += result.search_time
                except Exception as e:
                    print(f"Error with {strategy} on '{query}': {e}")
                    continue
            
            if strategy_results:
                # Calculate metrics
                avg_time = total_time / len(strategy_results)
                avg_results = np.mean([r.total_results for r in strategy_results])
                
                # Score distribution
                all_scores = []
                for result in strategy_results:
                    all_scores.extend([r.final_score for r in result.results])
                
                evaluation['strategies'][strategy] = {
                    'average_search_time': avg_time,
                    'average_results_count': avg_results,
                    'score_stats': {
                        'mean': float(np.mean(all_scores)) if all_scores else 0.0,
                        'std': float(np.std(all_scores)) if all_scores else 0.0,
                        'min': float(np.min(all_scores)) if all_scores else 0.0,
                        'max': float(np.max(all_scores)) if all_scores else 0.0
                    }
                }
        
        # Analyze result overlap for each query
        for query in queries[:5]:  # Analyze first 5 queries in detail
            try:
                overlap_analysis = self.analyze_result_overlap(query)
                evaluation['overlap_analysis'].append(overlap_analysis)
            except Exception as e:
                print(f"Error analyzing overlap for '{query}': {e}")
                continue
        
        return evaluation
    
    def _update_search_stats(self, fusion_strategy: str, search_time: float):
        """Update search statistics."""
        self.search_stats['total_searches'] += 1
        self.search_stats['fusion_strategies_used'][fusion_strategy] += 1
        self.search_stats['total_search_time'] += search_time
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get search statistics."""
        stats = dict(self.search_stats)
        
        if self.search_stats['total_searches'] > 0:
            stats['average_search_time'] = self.search_stats['total_search_time'] / self.search_stats['total_searches']
        else:
            stats['average_search_time'] = 0.0
        
        # Convert defaultdict to regular dict
        stats['fusion_strategies_used'] = dict(stats['fusion_strategies_used'])
        
        return stats


def create_hybrid_ranker(tfidf_engine: TFIDFSearchEngine, 
                        semantic_engine: SemanticSearchEngine) -> HybridRanker:
    """
    Create a hybrid ranker with TF-IDF and semantic search engines.
    
    Args:
        tfidf_engine: TF-IDF search engine
        semantic_engine: Semantic search engine
    
    Returns:
        Configured HybridRanker
    """
    return HybridRanker(tfidf_engine, semantic_engine)


if __name__ == "__main__":
    # Test the hybrid ranker
    print("Hybrid Ranker Test")
    print("=" * 50)
    
    if not COMPONENTS_AVAILABLE:
        print("Required components not available. Cannot run test.")
        exit(1)
    
    print("Note: This test requires initialized TF-IDF and semantic search engines.")
    print("Run the integration demo script to see full functionality.")