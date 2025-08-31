#!/usr/bin/env python3
"""
Semantic Search Demo Application
===============================

Simplified version of the semantic search web interface that runs without
full dependencies for demonstration purposes. Shows the complete UI and
API structure with mock data.

Usage:
    python demo_app.py

Author: AI Bootcamp Week 3-4
Date: 2025
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import time
import json

# FastAPI and web framework imports
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import Pydantic models
from models import (
    SearchRequest, SearchResponse, DocumentInfo, HealthResponse,
    MetricsResponse, ComparisonRequest, ComparisonResponse,
    SearchResult as SearchResultModel
)


class MockSearchService:
    """
    Mock search service for demonstration purposes.
    Returns realistic sample data without requiring full ML setup.
    """
    
    def __init__(self):
        self.search_stats = {
            'total_searches': 42,
            'search_times': [],
            'error_count': 1,
            'search_types': {'semantic': 15, 'tfidf': 12, 'hybrid': 15}
        }
        
        # Mock documents
        self.documents = [
            {
                'document_id': 'ml_fundamentals.pdf',
                'chunk_count': 25,
                'has_embeddings': True,
                'metadata': {'title': 'Machine Learning Fundamentals', 'pages': 180}
            },
            {
                'document_id': 'deep_learning_guide.pdf',
                'chunk_count': 35,
                'has_embeddings': True,
                'metadata': {'title': 'Deep Learning Guide', 'pages': 240}
            },
            {
                'document_id': 'ai_applications.pdf',
                'chunk_count': 18,
                'has_embeddings': True,
                'metadata': {'title': 'AI Applications in Industry', 'pages': 120}
            }
        ]
        
        # Mock search results
        self.sample_results = {
            'semantic': [
                {
                    'content': 'Machine learning algorithms are computational methods that allow systems to learn and improve from data without being explicitly programmed. These algorithms identify patterns, make predictions, and adapt to new information.',
                    'score': 0.89,
                    'document_id': 'ml_fundamentals.pdf',
                    'chunk_id': 'chunk_001',
                    'metadata': {'page': 15, 'section': 'Introduction to ML'}
                },
                {
                    'content': 'Deep learning represents a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data. It has revolutionized fields like computer vision and natural language processing.',
                    'score': 0.85,
                    'document_id': 'deep_learning_guide.pdf',
                    'chunk_id': 'chunk_042',
                    'metadata': {'page': 32, 'section': 'Deep Learning Basics'}
                },
                {
                    'content': 'Artificial intelligence applications span across various industries, from healthcare diagnosis systems to autonomous vehicles, demonstrating the transformative power of intelligent algorithms in solving real-world problems.',
                    'score': 0.82,
                    'document_id': 'ai_applications.pdf',
                    'chunk_id': 'chunk_015',
                    'metadata': {'page': 45, 'section': 'Industry Applications'}
                }
            ],
            'tfidf': [
                {
                    'content': 'Machine learning algorithms form the backbone of modern AI systems. Classification algorithms, regression algorithms, and clustering algorithms each serve different purposes in data analysis and pattern recognition.',
                    'score': 0.75,
                    'document_id': 'ml_fundamentals.pdf',
                    'chunk_id': 'chunk_008',
                    'matched_terms': ['machine', 'learning', 'algorithms'],
                    'metadata': {'page': 28, 'section': 'Algorithm Types'}
                },
                {
                    'content': 'Supervised learning algorithms require labeled training data to learn patterns, while unsupervised algorithms discover hidden structures in unlabeled datasets. Algorithm selection depends on the specific problem and available data.',
                    'score': 0.68,
                    'document_id': 'ml_fundamentals.pdf',
                    'chunk_id': 'chunk_012',
                    'matched_terms': ['algorithms', 'learning'],
                    'metadata': {'page': 52, 'section': 'Learning Paradigms'}
                },
                {
                    'content': 'Optimization algorithms play a crucial role in training machine learning models. Gradient descent and its variants are fundamental algorithms used to minimize loss functions and improve model performance.',
                    'score': 0.65,
                    'document_id': 'deep_learning_guide.pdf',
                    'chunk_id': 'chunk_028',
                    'matched_terms': ['algorithms', 'machine', 'learning'],
                    'metadata': {'page': 78, 'section': 'Optimization Techniques'}
                }
            ]
        }
    
    async def search_semantic(self, query: str, top_k: int = 10, model_name: str = 'sentence_transformer') -> Dict[str, Any]:
        """Mock semantic search."""
        search_time = 0.156  # Simulated search time
        self._update_stats('semantic', search_time)
        
        results = self.sample_results['semantic'][:top_k]
        
        return {
            "results": results,
            "total_results": len(results),
            "search_time": search_time,
            "model_used": model_name,
            "query": query
        }
    
    async def search_tfidf(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """Mock TF-IDF search."""
        search_time = 0.089  # Simulated search time
        self._update_stats('tfidf', search_time)
        
        results = self.sample_results['tfidf'][:top_k]
        
        return {
            "results": results,
            "total_results": len(results),
            "search_time": search_time,
            "processed_query": query.lower().strip(),
            "query": query
        }
    
    async def search_hybrid(self, query: str, top_k: int = 10, 
                           semantic_weight: float = 0.6, 
                           tfidf_weight: float = 0.4,
                           strategy: str = 'linear') -> Dict[str, Any]:
        """Mock hybrid search."""
        search_time = 0.203  # Simulated search time
        self._update_stats('hybrid', search_time)
        
        # Simulate hybrid results by combining and reranking
        semantic_results = self.sample_results['semantic']
        tfidf_results = self.sample_results['tfidf']
        
        hybrid_results = []
        for i, (sem, tfidf) in enumerate(zip(semantic_results, tfidf_results)):
            final_score = (sem['score'] * semantic_weight) + (tfidf['score'] * tfidf_weight)
            
            hybrid_results.append({
                'content': sem['content'],
                'final_score': final_score,
                'semantic_score': sem['score'],
                'tfidf_score': tfidf['score'],
                'document_id': sem['document_id'],
                'chunk_id': sem['chunk_id'],
                'metadata': sem['metadata']
            })
            
            if i + 1 >= top_k:
                break
        
        # Sort by final score
        hybrid_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return {
            "results": hybrid_results,
            "total_results": len(hybrid_results),
            "search_time": search_time,
            "fusion_strategy": strategy,
            "weights": {"semantic": semantic_weight, "tfidf": tfidf_weight},
            "query": query
        }
    
    async def compare_search_methods(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Compare all search methods."""
        comparison_results = {
            'semantic': await self.search_semantic(query, top_k),
            'tfidf': await self.search_tfidf(query, top_k),
            'hybrid': await self.search_hybrid(query, top_k)
        }
        
        return {
            "query": query,
            "top_k": top_k,
            "results": comparison_results,
            "timestamp": time.time()
        }
    
    def get_documents(self) -> List[Dict[str, Any]]:
        """Get mock document list."""
        return self.documents
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get mock health status."""
        return {
            "status": "healthy",
            "message": "Demo mode - all components operational",
            "components": {
                "document_store": {"status": "healthy", "documents": 3, "chunks": 78},
                "vector_store": {"status": "healthy", "indices": 2, "vectors": 78},
                "tfidf_engine": {"status": "healthy", "searches": self.search_stats['total_searches']}
            },
            "search_stats": self.search_stats
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get mock performance metrics."""
        return {
            "search_metrics": {
                "total_searches": self.search_stats['total_searches'],
                "average_search_time": 0.149,
                "error_count": self.search_stats['error_count'],
                "search_types": self.search_stats['search_types']
            },
            "component_stats": {
                "document_store": {"documents": 3, "chunks": 78, "embeddings": 78},
                "vector_store": {"num_indices": 2, "total_vectors": 78}
            },
            "service_status": "demo_mode"
        }
    
    def _update_stats(self, search_type: str, search_time: float):
        """Update search statistics."""
        self.search_stats['total_searches'] += 1
        self.search_stats['search_times'].append(search_time)
        self.search_stats['search_types'][search_type] += 1


# Initialize FastAPI app
app = FastAPI(
    title="Semantic Search Engine (Demo)",
    description="Educational semantic search interface - Demo Mode with mock data",
    version="1.0.0-demo",
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

# Initialize mock search service
search_service = MockSearchService()

# Mount static files
static_path = Path(__file__).parent / "static"
static_path.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface."""
    try:
        html_file = Path(__file__).parent / "static" / "index.html"
        if html_file.exists():
            content = html_file.read_text(encoding='utf-8')
            # Add demo banner
            demo_banner = """
            <div style="background: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; margin: 10px; text-align: center; border-radius: 5px;">
                <strong>🎭 DEMO MODE</strong> - This interface is running with mock data for demonstration purposes.
            </div>
            """
            content = content.replace('<header class="header">', demo_banner + '<header class="header">')
            return HTMLResponse(content=content, status_code=200)
        else:
            return HTMLResponse(
                content="<h1>Semantic Search Engine (Demo)</h1><p>Web interface not found. Please check static files.</p>",
                status_code=200
            )
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error</h1><p>{str(e)}</p>", status_code=500)


@app.post("/api/search", response_model=SearchResponse)
async def search_endpoint(request: SearchRequest):
    """Main search endpoint with mock data."""
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
    """Compare different search methods with mock data."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        result = await search_service.compare_search_methods(request.query, request.top_k)
        return ComparisonResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@app.get("/api/documents")
async def get_documents():
    """Get mock document list."""
    documents = search_service.get_documents()
    return {"documents": documents}


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Mock health check endpoint."""
    status = search_service.get_health_status()
    return HealthResponse(**status)


@app.get("/api/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get mock performance metrics."""
    metrics = search_service.get_metrics()
    return MetricsResponse(**metrics)


@app.get("/api/models")
async def get_available_models():
    """Get mock available models."""
    return {
        "models": [
            {
                "name": "sentence_transformer",
                "dimension": 384,
                "type": "SentenceTransformer"
            },
            {
                "name": "word2vec",
                "dimension": 300,
                "type": "Word2Vec"
            }
        ]
    }


if __name__ == "__main__":
    print("Semantic Search Engine - Demo Mode")
    print("=" * 50)
    print("Running with mock data for demonstration purposes.")
    print("This showcases the complete web interface functionality")
    print("without requiring full ML dependencies.")
    print("=" * 50)
    print()
    print("Access the interface at: http://localhost:8000")
    print("API documentation: http://localhost:8000/api/docs")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )