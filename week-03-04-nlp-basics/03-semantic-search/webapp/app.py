#!/usr/bin/env python3
"""
FastAPI Semantic Search Web Interface
====================================

A complete web interface for the semantic search engine showcasing:
- RESTful API endpoints for all search types (semantic, TF-IDF, hybrid)
- Educational comparison interface
- Performance metrics and analytics
- Production-ready with proper error handling

Author: AI Bootcamp Week 3-4
Date: 2025
"""

import os
import sys
import json
import time
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add current directory and subdirectories to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir / "02-indexing"))
sys.path.append(str(current_dir / "01-core"))

# FastAPI and web framework imports
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import Pydantic models
from models import (
    SearchRequest, SearchResponse, DocumentInfo, HealthResponse,
    MetricsResponse, ComparisonRequest, ComparisonResponse,
    SearchResult as SearchResultModel
)

# Import search components
try:
    from document_store import DocumentStore
    from embedding_engine import EnhancedEmbeddingEngine, create_embedding_engine
    from vector_store import FAISSVectorStore, create_vector_store
    from search_engine import SemanticSearchEngine, SearchConfig
    from tfidf_search import TFIDFSearchEngine
    from hybrid_ranker import HybridRanker, create_hybrid_ranker
    
    COMPONENTS_AVAILABLE = True
    print("All search components imported successfully")
except ImportError as e:
    print(f"✗ Error importing components: {e}")
    COMPONENTS_AVAILABLE = False


class SearchService:
    """
    Main search service that manages all search components.
    """
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.components = {}
        self.search_stats = {
            'total_searches': 0,
            'search_times': [],
            'error_count': 0,
            'search_types': {'semantic': 0, 'tfidf': 0, 'hybrid': 0}
        }
        self.initialized = False
        
    async def initialize(self):
        """Initialize all search components."""
        if self.initialized:
            return True
            
        print("Initializing search service...")
        
        try:
            # Paths
            document_store_path = self.base_path / "demo_storage" / "document_store"
            embeddings_cache_path = self.base_path / "demo_storage" / "embeddings_cache.json"
            vector_store_path = self.base_path / "demo_storage" / "vector_indices"
            
            # Initialize document store
            print("1. Initializing Document Store...")
            self.components['document_store'] = DocumentStore(str(document_store_path))
            
            # Initialize embedding engine
            print("2. Initializing Embedding Engine...")
            self.components['embedding_engine'] = create_embedding_engine(
                use_word2vec=True,
                use_sentence_transformers=True,
                cache_file=str(embeddings_cache_path)
            )
            
            # Initialize vector store
            print("3. Initializing Vector Store...")
            self.components['vector_store'] = create_vector_store(str(vector_store_path))
            
            # Initialize TF-IDF engine
            print("4. Initializing TF-IDF Search Engine...")
            self.components['tfidf_engine'] = TFIDFSearchEngine(self.components['document_store'])
            
            # Build TF-IDF index if needed
            tfidf_index_path = self.base_path / "demo_storage" / "tfidf_index.pkl"
            if tfidf_index_path.exists():
                print("   Loading existing TF-IDF index...")
                self.components['tfidf_engine'].load_index(str(tfidf_index_path))
            else:
                print("   Building TF-IDF index...")
                self.components['tfidf_engine'].build_index()
                self.components['tfidf_engine'].save_index(str(tfidf_index_path))
            
            # Initialize search engines
            print("5. Initializing Search Engines...")
            self.components['semantic_engine'] = SemanticSearchEngine(
                self.components['document_store'],
                self.components['embedding_engine'],
                self.components['vector_store']
            )
            
            self.components['hybrid_ranker'] = create_hybrid_ranker(
                self.components['tfidf_engine'],
                self.components['semantic_engine']
            )
            
            self.initialized = True
            print("Search service initialized successfully")
            return True
            
        except Exception as e:
            print(f"✗ Error initializing search service: {e}")
            traceback.print_exc()
            return False
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get service health status."""
        if not self.initialized:
            return {
                "status": "unhealthy",
                "message": "Service not initialized",
                "components": {}
            }
        
        try:
            # Check each component
            components_status = {}
            
            # Document store
            doc_stats = self.components['document_store'].get_storage_stats()
            components_status['document_store'] = {
                "status": "healthy",
                "documents": doc_stats['documents'],
                "chunks": doc_stats['chunks']
            }
            
            # Vector store
            vector_stats = self.components['vector_store'].get_stats()
            components_status['vector_store'] = {
                "status": "healthy",
                "indices": vector_stats['num_indices'],
                "vectors": vector_stats['total_vectors']
            }
            
            # TF-IDF engine
            tfidf_stats = self.components['tfidf_engine'].get_search_stats()
            components_status['tfidf_engine'] = {
                "status": "healthy",
                "searches": tfidf_stats['total_searches']
            }
            
            return {
                "status": "healthy",
                "message": "All components operational",
                "components": components_status,
                "search_stats": self.search_stats
            }
            
        except Exception as e:
            return {
                "status": "degraded",
                "message": f"Component check failed: {str(e)}",
                "components": {}
            }
    
    async def search_semantic(self, query: str, top_k: int = 10, model_name: str = 'sentence_transformer') -> Dict[str, Any]:
        """Perform semantic search."""
        start_time = time.time()
        
        try:
            config = SearchConfig(model_name=model_name, top_k=top_k)
            result = self.components['semantic_engine'].search(query, config)
            
            search_time = time.time() - start_time
            self._update_stats('semantic', search_time)
            
            return {
                "results": [
                    {
                        "content": r.content,
                        "score": r.score,
                        "document_id": r.document_id,
                        "chunk_id": r.chunk_id,
                        "metadata": r.metadata
                    }
                    for r in result.results
                ],
                "total_results": result.total_results,
                "search_time": search_time,
                "model_used": model_name,
                "query": query
            }
            
        except Exception as e:
            self.search_stats['error_count'] += 1
            raise HTTPException(status_code=500, detail=f"Semantic search failed: {str(e)}")
    
    async def search_tfidf(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """Perform TF-IDF search."""
        start_time = time.time()
        
        try:
            result = self.components['tfidf_engine'].search(query, top_k=top_k)
            
            search_time = time.time() - start_time
            self._update_stats('tfidf', search_time)
            
            return {
                "results": [
                    {
                        "content": r.content,
                        "score": r.score,
                        "document_id": r.document_id,
                        "chunk_id": r.chunk_id,
                        "matched_terms": getattr(r, 'matched_terms', []),
                        "metadata": r.metadata
                    }
                    for r in result.results
                ],
                "total_results": result.total_results,
                "search_time": search_time,
                "processed_query": result.processed_query,
                "query": query
            }
            
        except Exception as e:
            self.search_stats['error_count'] += 1
            raise HTTPException(status_code=500, detail=f"TF-IDF search failed: {str(e)}")
    
    async def search_hybrid(self, query: str, top_k: int = 10, 
                           semantic_weight: float = 0.6, 
                           tfidf_weight: float = 0.4,
                           strategy: str = 'linear') -> Dict[str, Any]:
        """Perform hybrid search."""
        start_time = time.time()
        
        try:
            result = self.components['hybrid_ranker'].search(
                query=query,
                fusion_strategy=strategy,
                semantic_weight=semantic_weight,
                tfidf_weight=tfidf_weight,
                top_k=top_k
            )
            
            search_time = time.time() - start_time
            self._update_stats('hybrid', search_time)
            
            return {
                "results": [
                    {
                        "content": r.content,
                        "final_score": r.final_score,
                        "semantic_score": r.scores.get('semantic', 0),
                        "tfidf_score": r.scores.get('tfidf', 0),
                        "document_id": r.document_id,
                        "chunk_id": r.chunk_id,
                        "metadata": r.metadata
                    }
                    for r in result.results
                ],
                "total_results": result.total_results,
                "search_time": search_time,
                "fusion_strategy": strategy,
                "weights": {"semantic": semantic_weight, "tfidf": tfidf_weight},
                "query": query
            }
            
        except Exception as e:
            self.search_stats['error_count'] += 1
            raise HTTPException(status_code=500, detail=f"Hybrid search failed: {str(e)}")
    
    async def compare_search_methods(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Compare all search methods side by side."""
        comparison_results = {}
        
        try:
            # Semantic search
            semantic_result = await self.search_semantic(query, top_k)
            comparison_results['semantic'] = semantic_result
        except Exception as e:
            comparison_results['semantic'] = {"error": str(e)}
        
        try:
            # TF-IDF search
            tfidf_result = await self.search_tfidf(query, top_k)
            comparison_results['tfidf'] = tfidf_result
        except Exception as e:
            comparison_results['tfidf'] = {"error": str(e)}
        
        try:
            # Hybrid search
            hybrid_result = await self.search_hybrid(query, top_k)
            comparison_results['hybrid'] = hybrid_result
        except Exception as e:
            comparison_results['hybrid'] = {"error": str(e)}
        
        return {
            "query": query,
            "top_k": top_k,
            "results": comparison_results,
            "timestamp": time.time()
        }
    
    def get_documents(self) -> List[Dict[str, Any]]:
        """Get list of available documents."""
        try:
            doc_stats = self.components['document_store'].get_storage_stats()
            
            documents = []
            # Get document information from storage
            chunks = self.components['document_store'].get_all_chunks()
            doc_info = {}
            
            for chunk in chunks:
                doc_id = chunk.get('document_id', 'unknown')
                if doc_id not in doc_info:
                    doc_info[doc_id] = {
                        'document_id': doc_id,
                        'chunk_count': 0,
                        'has_embeddings': False,
                        'metadata': chunk.get('metadata', {})
                    }
                
                doc_info[doc_id]['chunk_count'] += 1
                if chunk.get('has_embedding', False):
                    doc_info[doc_id]['has_embeddings'] = True
            
            return list(doc_info.values())
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get documents: {str(e)}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        try:
            # Calculate averages
            avg_search_time = 0
            if self.search_stats['search_times']:
                avg_search_time = sum(self.search_stats['search_times']) / len(self.search_stats['search_times'])
            
            # Component statistics
            component_stats = {}
            
            if self.initialized:
                # Document store stats
                doc_stats = self.components['document_store'].get_storage_stats()
                component_stats['document_store'] = doc_stats
                
                # Vector store stats
                vector_stats = self.components['vector_store'].get_stats()
                component_stats['vector_store'] = vector_stats
                
                # TF-IDF stats
                tfidf_stats = self.components['tfidf_engine'].get_search_stats()
                component_stats['tfidf_engine'] = tfidf_stats
            
            return {
                "search_metrics": {
                    "total_searches": self.search_stats['total_searches'],
                    "average_search_time": avg_search_time,
                    "error_count": self.search_stats['error_count'],
                    "search_types": self.search_stats['search_types']
                },
                "component_stats": component_stats,
                "service_status": "initialized" if self.initialized else "not_initialized"
            }
            
        except Exception as e:
            return {
                "error": f"Failed to get metrics: {str(e)}",
                "search_metrics": self.search_stats
            }
    
    def _update_stats(self, search_type: str, search_time: float):
        """Update search statistics."""
        self.search_stats['total_searches'] += 1
        self.search_stats['search_times'].append(search_time)
        self.search_stats['search_types'][search_type] += 1
        
        # Keep only last 1000 search times to prevent memory issues
        if len(self.search_stats['search_times']) > 1000:
            self.search_stats['search_times'] = self.search_stats['search_times'][-1000:]


# Initialize FastAPI app
app = FastAPI(
    title="Semantic Search Engine",
    description="Educational semantic search interface showcasing TF-IDF, vector similarity, and hybrid ranking",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize search service
search_service = SearchService(Path(__file__).parent)

# Mount static files
static_path = Path(__file__).parent / "static"
static_path.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


@app.on_event("startup")
async def startup_event():
    """Initialize search service on startup."""
    if COMPONENTS_AVAILABLE:
        await search_service.initialize()
    else:
        print("Warning: Search components not available - running in demo mode")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface."""
    try:
        html_file = Path(__file__).parent / "static" / "index.html"
        if html_file.exists():
            return HTMLResponse(content=html_file.read_text(), status_code=200)
        else:
            return HTMLResponse(
                content="<h1>Semantic Search Engine</h1><p>Web interface not found. Please check static files.</p>",
                status_code=200
            )
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error</h1><p>{str(e)}</p>", status_code=500)


@app.post("/api/search", response_model=SearchResponse)
async def search_endpoint(request: SearchRequest):
    """Main search endpoint supporting all search types."""
    if not search_service.initialized:
        raise HTTPException(status_code=503, detail="Search service not initialized")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        if request.search_type == "semantic":
            result = await search_service.search_semantic(
                request.query, 
                request.top_k, 
                request.model_name or 'sentence_transformer'
            )
        elif request.search_type == "tfidf":
            result = await search_service.search_tfidf(request.query, request.top_k)
        elif request.search_type == "hybrid":
            result = await search_service.search_hybrid(
                request.query,
                request.top_k,
                request.semantic_weight or 0.6,
                request.tfidf_weight or 0.4,
                request.fusion_strategy or 'linear'
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid search type")
        
        return SearchResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/api/compare", response_model=ComparisonResponse)
async def compare_endpoint(request: ComparisonRequest):
    """Compare different search methods side by side."""
    if not search_service.initialized:
        raise HTTPException(status_code=503, detail="Search service not initialized")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        result = await search_service.compare_search_methods(request.query, request.top_k)
        return ComparisonResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@app.get("/api/documents")
async def get_documents():
    """Get list of available documents."""
    if not search_service.initialized:
        raise HTTPException(status_code=503, detail="Search service not initialized")
    
    try:
        documents = search_service.get_documents()
        return {"documents": documents}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get documents: {str(e)}")


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Service health check endpoint."""
    status = search_service.get_health_status()
    return HealthResponse(**status)


@app.get("/api/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get performance metrics and statistics."""
    try:
        metrics = search_service.get_metrics()
        return MetricsResponse(**metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@app.get("/api/models")
async def get_available_models():
    """Get available embedding models."""
    if not search_service.initialized:
        return {"models": [], "message": "Service not initialized"}
    
    try:
        embedding_stats = search_service.components['embedding_engine'].get_stats()
        models = []
        for model_name, info in embedding_stats['models'].items():
            models.append({
                "name": model_name,
                "dimension": info['embedding_dim'],
                "type": info['model_class']
            })
        
        return {"models": models}
        
    except Exception as e:
        return {"models": [], "error": str(e)}


if __name__ == "__main__":
    print("Starting Semantic Search Engine Web Interface...")
    print("=" * 50)
    print("Features:")
    print("- Semantic search with vector similarity")
    print("- Classical TF-IDF search")
    print("- Hybrid ranking combining both approaches")
    print("- Educational comparison interface")
    print("- Performance metrics and monitoring")
    print("=" * 50)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )