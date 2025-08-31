#!/usr/bin/env python3
"""
Search Integration Demo
======================

This script demonstrates the complete FAISS vector search functionality
integrated with the existing semantic search architecture. It showcases:

- Loading existing embeddings and building FAISS indices
- TF-IDF baseline search implementation
- Semantic search using FAISS vector similarity
- Hybrid ranking combining both approaches
- Performance comparison and evaluation
- Educational analysis of different search methods

Author: AI Bootcamp Week 3-4
Date: 2025
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Core libraries
import numpy as np

# Import components
try:
    # Add subdirectories to path
    sys.path.append(str(current_dir / "02-indexing"))
    sys.path.append(str(current_dir / "01-core"))
    
    from document_store import DocumentStore
    from embedding_engine import EnhancedEmbeddingEngine, create_embedding_engine
    from vector_store import FAISSVectorStore, create_vector_store
    from search_engine import SemanticSearchEngine, SearchConfig
    from tfidf_search import TFIDFSearchEngine
    from hybrid_ranker import HybridRanker, create_hybrid_ranker
    
    COMPONENTS_AVAILABLE = True
    print("All search components imported successfully")
except ImportError as e:
    print(f"Error importing components: {e}")
    COMPONENTS_AVAILABLE = False
    sys.exit(1)


def setup_demo_environment(base_path: Path) -> Dict[str, Any]:
    """
    Setup the demo environment with existing data.
    
    Args:
        base_path: Base path to the semantic search directory
    
    Returns:
        Dictionary with initialized components
    """
    print("Setting up demo environment...")
    
    # Paths
    document_store_path = base_path / "demo_storage" / "document_store"
    embeddings_cache_path = base_path / "demo_storage" / "embeddings_cache.json"
    vector_store_path = base_path / "demo_storage" / "vector_indices"
    
    print(f"Document store path: {document_store_path}")
    print(f"Embeddings cache: {embeddings_cache_path}")
    print(f"Vector store path: {vector_store_path}")
    
    # Initialize components
    print("\n1. Initializing Document Store...")
    document_store = DocumentStore(str(document_store_path))
    doc_stats = document_store.get_storage_stats()
    print(f"   - Documents: {doc_stats['documents']}")
    print(f"   - Chunks: {doc_stats['chunks']}")
    print(f"   - Embeddings: {doc_stats['embeddings']}")
    
    # Calculate chunks with embeddings
    chunks_with_embeddings = 0
    for chunk in document_store.get_all_chunks():
        if chunk.get('has_embedding', False):
            chunks_with_embeddings += 1
    print(f"   - Chunks with embeddings: {chunks_with_embeddings}")
    
    print("\n2. Initializing Embedding Engine...")
    embedding_engine = create_embedding_engine(
        use_word2vec=True,
        use_sentence_transformers=True,
        cache_file=str(embeddings_cache_path)
    )
    print(f"   - Available models: {list(embedding_engine.models.keys())}")
    
    print("\n3. Initializing Vector Store...")
    vector_store = create_vector_store(str(vector_store_path))
    
    print("\n4. Initializing TF-IDF Search Engine...")
    tfidf_engine = TFIDFSearchEngine(document_store)
    
    components = {
        'document_store': document_store,
        'embedding_engine': embedding_engine,
        'vector_store': vector_store,
        'tfidf_engine': tfidf_engine,
        'base_path': base_path
    }
    
    return components


def build_faiss_indices(components: Dict[str, Any]) -> bool:
    """
    Build FAISS indices from existing embeddings.
    
    Args:
        components: Dictionary with initialized components
    
    Returns:
        True if indices built successfully, False otherwise
    """
    print("\n" + "="*60)
    print("BUILDING FAISS INDICES")
    print("="*60)
    
    vector_store = components['vector_store']
    document_store = components['document_store']
    embedding_engine = components['embedding_engine']
    
    # Check existing indices
    existing_stats = vector_store.get_stats()
    if existing_stats['total_vectors'] > 0:
        print("Existing FAISS indices found:")
        for model_name, info in existing_stats['indices'].items():
            print(f"  - {model_name}: {info['num_vectors']} vectors ({info['embedding_dim']}D)")
        
        print("\nUsing existing indices...")
        return True
    
    # Build indices from document store embeddings
    print("Building FAISS indices from existing embeddings...")
    
    results = vector_store.build_index_from_document_store(
        document_store=document_store,
        embedding_engine=embedding_engine,
        model_names=None,  # Build for all available models
        batch_size=32
    )
    
    print(f"\nIndex building results:")
    for model_name, success in results.items():
        status = "Success" if success else "Failed"
        print(f"  - {model_name}: {status}")
    
    # Display final statistics
    final_stats = vector_store.get_stats()
    print(f"\nFinal Vector Store Statistics:")
    print(f"  - Total indices: {final_stats['num_indices']}")
    print(f"  - Total vectors: {final_stats['total_vectors']}")
    
    return any(results.values())


def build_tfidf_index(components: Dict[str, Any]) -> bool:
    """
    Build TF-IDF index from document chunks.
    
    Args:
        components: Dictionary with initialized components
    
    Returns:
        True if index built successfully, False otherwise
    """
    print("\n" + "="*60)
    print("BUILDING TF-IDF INDEX")
    print("="*60)
    
    tfidf_engine = components['tfidf_engine']
    
    # Build TF-IDF index
    success = tfidf_engine.build_index()
    
    if success:
        # Get vocabulary statistics
        vocab_stats = tfidf_engine.get_vocabulary_stats()
        print(f"\nTF-IDF Index Statistics:")
        print(f"  - Documents indexed: {vocab_stats['document_count']}")
        print(f"  - Vocabulary size: {vocab_stats['vocabulary_size']}")
        print(f"  - Matrix density: {vocab_stats['matrix_density']:.4f}")
        print(f"  - Most common terms: {[term for term, _ in vocab_stats['most_common_terms'][:5]]}")
        
        # Save index for future use
        tfidf_index_path = components['base_path'] / "demo_storage" / "tfidf_index.pkl"
        tfidf_engine.save_index(str(tfidf_index_path))
        print(f"  - Saved to: {tfidf_index_path}")
    
    return success


def demonstrate_search_capabilities(components: Dict[str, Any]):
    """
    Demonstrate different search capabilities.
    
    Args:
        components: Dictionary with initialized components
    """
    print("\n" + "="*60)
    print("SEARCH CAPABILITIES DEMONSTRATION")
    print("="*60)
    
    # Initialize search engines
    semantic_engine = SemanticSearchEngine(
        components['document_store'],
        components['embedding_engine'],
        components['vector_store']
    )
    
    hybrid_ranker = create_hybrid_ranker(
        components['tfidf_engine'],
        semantic_engine
    )
    
    # Test queries - mix of factual, conceptual, and specific
    test_queries = [
        "machine learning algorithms",
        "artificial intelligence applications",
        "neural networks deep learning",
        "data science statistics",
        "computer vision image recognition",
        "natural language processing NLP"
    ]
    
    print(f"Testing with {len(test_queries)} queries:")
    for i, query in enumerate(test_queries, 1):
        print(f"  {i}. {query}")
    
    # Test each query with different approaches
    for query in test_queries[:3]:  # Test first 3 queries in detail
        print(f"\n" + "-"*50)
        print(f"QUERY: '{query}'")
        print("-"*50)
        
        # 1. Semantic Search
        print("\n1. SEMANTIC SEARCH:")
        try:
            semantic_config = SearchConfig(
                model_name='sentence_transformer',  # Use sentence transformer if available
                top_k=5
            )
            semantic_result = semantic_engine.search(query, semantic_config)
            
            print(f"   Time: {semantic_result.search_time:.3f}s")
            print(f"   Results: {semantic_result.total_results}")
            
            for i, result in enumerate(semantic_result.results[:3], 1):
                print(f"   {i}. Score: {result.score:.3f}")
                print(f"      Content: {result.content[:100]}...")
                print(f"      Doc: {result.document_id}")
        
        except Exception as e:
            print(f"   Error: {e}")
        
        # 2. TF-IDF Search
        print("\n2. TF-IDF SEARCH:")
        try:
            tfidf_result = components['tfidf_engine'].search(query, top_k=5)
            
            print(f"   Time: {tfidf_result.search_time:.3f}s")
            print(f"   Results: {tfidf_result.total_results}")
            print(f"   Processed query: '{tfidf_result.processed_query}'")
            
            for i, result in enumerate(tfidf_result.results[:3], 1):
                print(f"   {i}. Score: {result.score:.3f}")
                print(f"      Content: {result.content[:100]}...")
                print(f"      Matched terms: {result.matched_terms[:5]}")
        
        except Exception as e:
            print(f"   Error: {e}")
        
        # 3. Hybrid Search
        print("\n3. HYBRID SEARCH:")
        try:
            hybrid_result = hybrid_ranker.search(
                query=query,
                fusion_strategy='linear',
                semantic_weight=0.6,
                tfidf_weight=0.4,
                top_k=5
            )
            
            print(f"   Time: {hybrid_result.search_time:.3f}s")
            print(f"   Results: {hybrid_result.total_results}")
            print(f"   Strategy: {hybrid_result.fusion_strategy}")
            
            for i, result in enumerate(hybrid_result.results[:3], 1):
                print(f"   {i}. Final Score: {result.final_score:.3f}")
                print(f"      Semantic: {result.scores.get('semantic', 0):.3f}")
                print(f"      TF-IDF: {result.scores.get('tfidf', 0):.3f}")
                print(f"      Content: {result.content[:100]}...")
        
        except Exception as e:
            print(f"   Error: {e}")


def compare_search_approaches(components: Dict[str, Any]):
    """
    Compare different search approaches systematically.
    
    Args:
        components: Dictionary with initialized components
    """
    print("\n" + "="*60)
    print("SEARCH APPROACH COMPARISON")
    print("="*60)
    
    # Initialize engines
    semantic_engine = SemanticSearchEngine(
        components['document_store'],
        components['embedding_engine'],
        components['vector_store']
    )
    
    hybrid_ranker = create_hybrid_ranker(
        components['tfidf_engine'],
        semantic_engine
    )
    
    # Comparison queries
    comparison_queries = [
        "machine learning algorithms",
        "neural networks",
        "artificial intelligence",
        "data analysis"
    ]
    
    print(f"Comparing approaches on {len(comparison_queries)} queries:\n")
    
    # Performance comparison
    performance_results = {
        'semantic': {'times': [], 'results_counts': []},
        'tfidf': {'times': [], 'results_counts': []},
        'hybrid': {'times': [], 'results_counts': []}
    }
    
    for query in comparison_queries:
        print(f"Query: '{query}'")
        
        # Semantic search
        try:
            config = SearchConfig(model_name='sentence_transformer', top_k=10)
            semantic_result = semantic_engine.search(query, config)
            performance_results['semantic']['times'].append(semantic_result.search_time)
            performance_results['semantic']['results_counts'].append(semantic_result.total_results)
            print(f"  Semantic: {semantic_result.search_time:.3f}s, {semantic_result.total_results} results")
        except Exception as e:
            print(f"  Semantic: Error - {e}")
        
        # TF-IDF search
        try:
            tfidf_result = components['tfidf_engine'].search(query, top_k=10)
            performance_results['tfidf']['times'].append(tfidf_result.search_time)
            performance_results['tfidf']['results_counts'].append(tfidf_result.total_results)
            print(f"  TF-IDF: {tfidf_result.search_time:.3f}s, {tfidf_result.total_results} results")
        except Exception as e:
            print(f"  TF-IDF: Error - {e}")
        
        # Hybrid search
        try:
            hybrid_result = hybrid_ranker.search(query, top_k=10)
            performance_results['hybrid']['times'].append(hybrid_result.search_time)
            performance_results['hybrid']['results_counts'].append(hybrid_result.total_results)
            print(f"  Hybrid: {hybrid_result.search_time:.3f}s, {hybrid_result.total_results} results")
        except Exception as e:
            print(f"  Hybrid: Error - {e}")
        
        print()
    
    # Calculate and display performance statistics
    print("PERFORMANCE SUMMARY:")
    print("-" * 30)
    
    for approach, data in performance_results.items():
        if data['times']:
            avg_time = np.mean(data['times'])
            avg_results = np.mean(data['results_counts'])
            print(f"{approach.title()}:")
            print(f"  Average time: {avg_time:.3f}s")
            print(f"  Average results: {avg_results:.1f}")
        else:
            print(f"{approach.title()}: No successful queries")
    
    # Fusion strategy comparison
    print(f"\nFUSION STRATEGY COMPARISON:")
    print("-" * 30)
    
    test_query = comparison_queries[0]
    try:
        strategies_result = hybrid_ranker.compare_strategies(test_query, top_k=5)
        
        for strategy, result in strategies_result.items():
            print(f"{strategy.upper()}:")
            print(f"  Time: {result.search_time:.3f}s")
            print(f"  Results: {result.total_results}")
            if result.results:
                print(f"  Top score: {result.results[0].final_score:.3f}")
    except Exception as e:
        print(f"Error comparing fusion strategies: {e}")


def analyze_result_quality(components: Dict[str, Any]):
    """
    Analyze the quality and characteristics of search results.
    
    Args:
        components: Dictionary with initialized components
    """
    print("\n" + "="*60)
    print("RESULT QUALITY ANALYSIS")
    print("="*60)
    
    # Initialize engines
    semantic_engine = SemanticSearchEngine(
        components['document_store'],
        components['embedding_engine'],
        components['vector_store']
    )
    
    hybrid_ranker = create_hybrid_ranker(
        components['tfidf_engine'],
        semantic_engine
    )
    
    # Analysis query
    analysis_query = "machine learning algorithms"
    print(f"Analyzing results for: '{analysis_query}'\n")
    
    # 1. Semantic search explanation
    print("1. SEMANTIC SEARCH ANALYSIS:")
    try:
        explanation = semantic_engine.explain_search(analysis_query, 'sentence_transformer', top_k=3)
        
        print(f"   Query embedding info:")
        print(f"     Dimension: {explanation['query_embedding_info']['dimension']}")
        print(f"     Norm: {explanation['query_embedding_info']['norm']:.3f}")
        
        print(f"   Top results analysis:")
        for result in explanation['top_results']:
            print(f"     Rank {result['rank']}: Score {result['score']:.3f}")
            print(f"       Content length: {result['content_length']}")
            if 'similarity_metrics' in result:
                print(f"       Cosine similarity: {result['similarity_metrics']['cosine']:.3f}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 2. TF-IDF search explanation
    print(f"\n2. TF-IDF SEARCH ANALYSIS:")
    try:
        tfidf_explanation = components['tfidf_engine'].explain_search(analysis_query, top_k=3)
        
        print(f"   Processed query: '{tfidf_explanation['processed_query']}'")
        print(f"   Query terms: {tfidf_explanation['query_terms']}")
        print(f"   Top query TF-IDF terms: {[f'{term}:{score:.3f}' for term, score in tfidf_explanation['query_tfidf_terms'][:3]]}")
        
        print(f"   Top results analysis:")
        for result in tfidf_explanation['top_results']:
            print(f"     Rank {result['rank']}: Score {result['score']:.3f}")
            print(f"       Matched terms: {result['matched_terms']}")
            print(f"       Top TF-IDF terms: {[f'{term}:{score:.3f}' for term, score in result['top_tfidf_terms'][:3]]}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 3. Result overlap analysis
    print(f"\n3. RESULT OVERLAP ANALYSIS:")
    try:
        overlap_analysis = hybrid_ranker.analyze_result_overlap(analysis_query, top_k=10)
        
        print(f"   Semantic results: {overlap_analysis['semantic_results']}")
        print(f"   TF-IDF results: {overlap_analysis['tfidf_results']}")
        print(f"   Overlap: {overlap_analysis['overlap_count']} results")
        print(f"   Jaccard similarity: {overlap_analysis['jaccard_similarity']:.3f}")
        print(f"   Rank correlation: {overlap_analysis['rank_correlation']:.3f}")
        print(f"   Semantic-only: {overlap_analysis['semantic_only_count']}")
        print(f"   TF-IDF-only: {overlap_analysis['tfidf_only_count']}")
    except Exception as e:
        print(f"   Error: {e}")


def display_final_statistics(components: Dict[str, Any]):
    """
    Display final statistics and summary.
    
    Args:
        components: Dictionary with initialized components
    """
    print("\n" + "="*60)
    print("FINAL STATISTICS AND SUMMARY")
    print("="*60)
    
    # Document store statistics
    print("DOCUMENT STORE:")
    doc_stats = components['document_store'].get_storage_stats()
    print(f"  - Documents: {doc_stats['documents']}")
    print(f"  - Total chunks: {doc_stats['chunks']}")
    print(f"  - Total embeddings: {doc_stats['embeddings']}")
    print(f"  - Models: {doc_stats['models']}")
    print(f"  - Storage: {doc_stats['total_storage_mb']:.2f} MB")
    
    # Vector store statistics
    print(f"\nVECTOR STORE:")
    vector_stats = components['vector_store'].get_stats()
    print(f"  - FAISS indices: {vector_stats['num_indices']}")
    print(f"  - Total vectors: {vector_stats['total_vectors']}")
    print(f"  - Storage directory: {vector_stats['storage_dir']}")
    
    for model_name, info in vector_stats['indices'].items():
        print(f"    - {model_name}: {info['num_vectors']} vectors ({info['embedding_dim']}D)")
    
    # TF-IDF statistics
    print(f"\nTF-IDF ENGINE:")
    tfidf_stats = components['tfidf_engine'].get_search_stats()
    vocab_stats = components['tfidf_engine'].get_vocabulary_stats()
    
    if 'error' not in vocab_stats:
        print(f"  - Vocabulary size: {vocab_stats['vocabulary_size']}")
        print(f"  - Documents indexed: {vocab_stats['document_count']}")
        print(f"  - Matrix density: {vocab_stats['matrix_density']:.4f}")
    
    print(f"  - Searches performed: {tfidf_stats['total_searches']}")
    if tfidf_stats['total_searches'] > 0:
        print(f"  - Average search time: {tfidf_stats['average_search_time']:.3f}s")
    
    # Embedding engine statistics
    print(f"\nEMBEDDING ENGINE:")
    embedding_stats = components['embedding_engine'].get_stats()
    print(f"  - Available models: {embedding_stats['num_models']}")
    print(f"  - Cache size: {embedding_stats['cache_stats']['cache_size']}")
    print(f"  - Cache hit rate: {embedding_stats['cache_stats']['hit_rate']:.2%}")
    
    for model_name, info in embedding_stats['models'].items():
        print(f"    - {model_name}: {info['embedding_dim']}D ({info['model_class']})")


def main():
    """Main demonstration function."""
    print("FAISS Vector Search Integration Demo")
    print("=" * 60)
    print("This demo showcases the complete semantic search system with:")
    print("- FAISS vector similarity search")
    print("- TF-IDF baseline search")
    print("- Hybrid ranking combining both approaches")
    print("- Performance comparison and analysis")
    
    if not COMPONENTS_AVAILABLE:
        print("\nError: Required components not available. Please check imports.")
        return
    
    # Setup
    base_path = Path(__file__).parent
    components = setup_demo_environment(base_path)
    
    if not components:
        print("Failed to setup demo environment")
        return
    
    try:
        # Build indices
        faiss_success = build_faiss_indices(components)
        tfidf_success = build_tfidf_index(components)
        
        if not faiss_success and not tfidf_success:
            print("Failed to build any search indices")
            return
        
        # Demonstrate capabilities
        demonstrate_search_capabilities(components)
        
        # Compare approaches
        compare_search_approaches(components)
        
        # Analyze result quality
        analyze_result_quality(components)
        
        # Final statistics
        display_final_statistics(components)
        
        print(f"\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("="*60)
        print("The semantic search system is now ready for use!")
        print("You can integrate these components into a web interface or API.")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()