"""
Pydantic Models for Semantic Search API
======================================

Request and response models for the FastAPI semantic search interface.
Provides proper validation, documentation, and type hints for all API endpoints.

Author: AI Bootcamp Week 3-4
Date: 2025
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, validator
from enum import Enum


class SearchType(str, Enum):
    """Enumeration of available search types."""
    semantic = "semantic"
    tfidf = "tfidf" 
    hybrid = "hybrid"


class FusionStrategy(str, Enum):
    """Enumeration of fusion strategies for hybrid search."""
    linear = "linear"
    rank_fusion = "rank_fusion"
    weighted_sum = "weighted_sum"


class SearchRequest(BaseModel):
    """Request model for search endpoints."""
    query: str = Field(..., description="Search query", min_length=1, max_length=500)
    search_type: SearchType = Field(default=SearchType.semantic, description="Type of search to perform")
    top_k: int = Field(default=10, description="Number of results to return", ge=1, le=50)
    
    # Semantic search specific
    model_name: Optional[str] = Field(default="sentence_transformer", description="Embedding model to use")
    
    # Hybrid search specific
    semantic_weight: Optional[float] = Field(default=0.6, description="Weight for semantic search", ge=0.0, le=1.0)
    tfidf_weight: Optional[float] = Field(default=0.4, description="Weight for TF-IDF search", ge=0.0, le=1.0)
    fusion_strategy: Optional[FusionStrategy] = Field(default=FusionStrategy.linear, description="Fusion strategy for hybrid search")
    
    @validator('semantic_weight', 'tfidf_weight')
    def validate_weights(cls, v, values):
        """Validate that weights are reasonable."""
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError('Weights must be between 0.0 and 1.0')
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "query": "machine learning algorithms",
                "search_type": "hybrid",
                "top_k": 5,
                "semantic_weight": 0.6,
                "tfidf_weight": 0.4,
                "fusion_strategy": "linear"
            }
        }


class SearchResult(BaseModel):
    """Individual search result model."""
    content: str = Field(..., description="Document content/snippet")
    document_id: str = Field(..., description="Document identifier")
    chunk_id: Optional[str] = Field(default=None, description="Chunk identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    # Different score types based on search method
    score: Optional[float] = Field(default=None, description="Similarity score (semantic/TF-IDF)")
    final_score: Optional[float] = Field(default=None, description="Final hybrid score")
    semantic_score: Optional[float] = Field(default=None, description="Semantic similarity score")
    tfidf_score: Optional[float] = Field(default=None, description="TF-IDF relevance score")
    
    # TF-IDF specific
    matched_terms: Optional[List[str]] = Field(default=None, description="Terms that matched in TF-IDF search")
    
    class Config:
        json_schema_extra = {
            "example": {
                "content": "Machine learning algorithms are computational methods that...",
                "document_id": "ml_textbook.pdf",
                "chunk_id": "chunk_001",
                "score": 0.89,
                "metadata": {"page": 1, "section": "Introduction"}
            }
        }


class SearchResponse(BaseModel):
    """Response model for search endpoints."""
    query: str = Field(..., description="Original search query")
    results: List[SearchResult] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results found")
    search_time: float = Field(..., description="Search execution time in seconds")
    
    # Search type specific fields
    model_used: Optional[str] = Field(default=None, description="Embedding model used (semantic search)")
    processed_query: Optional[str] = Field(default=None, description="Processed query (TF-IDF search)")
    fusion_strategy: Optional[str] = Field(default=None, description="Fusion strategy used (hybrid search)")
    weights: Optional[Dict[str, float]] = Field(default=None, description="Weights used (hybrid search)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "machine learning algorithms",
                "results": [
                    {
                        "content": "Machine learning algorithms are computational methods...",
                        "document_id": "ml_textbook.pdf", 
                        "score": 0.89,
                        "metadata": {"page": 1}
                    }
                ],
                "total_results": 15,
                "search_time": 0.124,
                "model_used": "sentence_transformer"
            }
        }


class ComparisonRequest(BaseModel):
    """Request model for search comparison endpoint."""
    query: str = Field(..., description="Search query to compare", min_length=1, max_length=500)
    top_k: int = Field(default=5, description="Number of results per method", ge=1, le=20)
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "neural networks deep learning",
                "top_k": 5
            }
        }


class ComparisonResponse(BaseModel):
    """Response model for search comparison endpoint."""
    query: str = Field(..., description="Original search query")
    top_k: int = Field(..., description="Number of results per method")
    results: Dict[str, Union[SearchResponse, Dict[str, str]]] = Field(..., description="Results from each search method")
    timestamp: float = Field(..., description="Comparison timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "neural networks deep learning",
                "top_k": 5,
                "results": {
                    "semantic": {"query": "...", "results": [], "search_time": 0.15},
                    "tfidf": {"query": "...", "results": [], "search_time": 0.08},
                    "hybrid": {"query": "...", "results": [], "search_time": 0.22}
                },
                "timestamp": 1640995200.0
            }
        }


class DocumentInfo(BaseModel):
    """Document information model."""
    document_id: str = Field(..., description="Document identifier")
    chunk_count: int = Field(..., description="Number of chunks in document")
    has_embeddings: bool = Field(..., description="Whether document has embeddings")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "ml_textbook.pdf",
                "chunk_count": 45,
                "has_embeddings": True,
                "metadata": {"title": "Machine Learning Fundamentals", "pages": 200}
            }
        }


class ComponentStatus(BaseModel):
    """Status of a search component."""
    status: str = Field(..., description="Component status (healthy/unhealthy/degraded)")
    
    # Component-specific fields (optional)
    documents: Optional[int] = Field(default=None, description="Number of documents")
    chunks: Optional[int] = Field(default=None, description="Number of chunks")
    indices: Optional[int] = Field(default=None, description="Number of indices")
    vectors: Optional[int] = Field(default=None, description="Number of vectors")
    searches: Optional[int] = Field(default=None, description="Number of searches performed")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Overall service status")
    message: str = Field(..., description="Status message")
    components: Dict[str, ComponentStatus] = Field(default_factory=dict, description="Component statuses")
    search_stats: Optional[Dict[str, Any]] = Field(default=None, description="Search statistics")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "message": "All components operational",
                "components": {
                    "document_store": {"status": "healthy", "documents": 5, "chunks": 142},
                    "vector_store": {"status": "healthy", "indices": 2, "vectors": 142}
                }
            }
        }


class SearchMetrics(BaseModel):
    """Search performance metrics."""
    total_searches: int = Field(..., description="Total number of searches performed")
    average_search_time: float = Field(..., description="Average search time in seconds")
    error_count: int = Field(..., description="Number of search errors")
    search_types: Dict[str, int] = Field(..., description="Count by search type")


class MetricsResponse(BaseModel):
    """Metrics endpoint response model."""
    search_metrics: SearchMetrics = Field(..., description="Search performance metrics")
    component_stats: Dict[str, Any] = Field(default_factory=dict, description="Component statistics")
    service_status: str = Field(..., description="Service initialization status")
    error: Optional[str] = Field(default=None, description="Error message if any")
    
    class Config:
        json_schema_extra = {
            "example": {
                "search_metrics": {
                    "total_searches": 125,
                    "average_search_time": 0.156,
                    "error_count": 2,
                    "search_types": {"semantic": 45, "tfidf": 35, "hybrid": 45}
                },
                "service_status": "initialized"
            }
        }


class ModelInfo(BaseModel):
    """Embedding model information."""
    name: str = Field(..., description="Model name")
    dimension: int = Field(..., description="Embedding dimension")
    type: str = Field(..., description="Model type/class")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "sentence_transformer",
                "dimension": 384,
                "type": "SentenceTransformer"
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response model."""
    detail: str = Field(..., description="Error message")
    error_type: Optional[str] = Field(default=None, description="Type of error")
    timestamp: Optional[float] = Field(default=None, description="Error timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Search service not initialized",
                "error_type": "ServiceError",
                "timestamp": 1640995200.0
            }
        }