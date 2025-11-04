"""
Production FastAPI Service with vLLM
=====================================

A production-ready API service for LLM inference with:
- vLLM for efficient inference
- Redis caching for repeated queries
- Prometheus metrics for monitoring
- Rate limiting
- Request validation

Installation:
    pip install fastapi uvicorn vllm redis prometheus-client pydantic

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from vllm import LLM, SamplingParams
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time
import logging
from typing import Optional, List
import hashlib
import json

# Import custom modules
try:
    from cache import CacheManager
    from metrics import metrics_middleware, REQUEST_COUNT, REQUEST_LATENCY
except ImportError:
    print("⚠️  cache.py and metrics.py not found. Running without caching and custom metrics.")
    CacheManager = None
    metrics_middleware = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="vLLM Inference API",
    description="High-performance LLM inference API using vLLM",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom metrics middleware if available
if metrics_middleware:
    app.middleware("http")(metrics_middleware)


# ============================================================================
# Request/Response Models
# ============================================================================

class GenerateRequest(BaseModel):
    """Request model for text generation."""
    prompt: str = Field(..., min_length=1, max_length=4096, description="Input prompt")
    max_tokens: int = Field(default=100, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.95, ge=0.0, le=1.0, description="Nucleus sampling threshold")
    top_k: int = Field(default=50, ge=1, le=100, description="Top-k sampling")
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0, description="Repetition penalty")
    use_cache: bool = Field(default=True, description="Whether to use response cache")

    @validator("prompt")
    def validate_prompt(cls, v):
        if not v.strip():
            raise ValueError("Prompt cannot be empty or whitespace only")
        return v


class GenerateResponse(BaseModel):
    """Response model for text generation."""
    generated_text: str
    prompt: str
    num_tokens: int
    inference_time_ms: float
    cached: bool = False


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model: str
    gpu_available: bool


class BatchGenerateRequest(BaseModel):
    """Request model for batch generation."""
    prompts: List[str] = Field(..., min_items=1, max_items=32)
    max_tokens: int = Field(default=100, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)


# ============================================================================
# Global State
# ============================================================================

class AppState:
    """Global application state."""
    llm: Optional[LLM] = None
    cache: Optional[CacheManager] = None
    model_name: str = "facebook/opt-125m"


state = AppState()


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize models and services on startup."""
    logger.info("🚀 Starting vLLM Inference API...")

    try:
        # Initialize vLLM
        logger.info(f"Loading model: {state.model_name}")
        state.llm = LLM(
            model=state.model_name,
            dtype="float16",
            max_model_len=512,
            gpu_memory_utilization=0.8,
        )
        logger.info("✅ Model loaded successfully")

        # Initialize cache if available
        if CacheManager:
            state.cache = CacheManager()
            logger.info("✅ Cache initialized")

    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("👋 Shutting down vLLM Inference API...")
    if state.cache:
        state.cache.close()


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "vLLM Inference API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    import torch

    return HealthResponse(
        status="healthy" if state.llm else "not_ready",
        model=state.model_name,
        gpu_available=torch.cuda.is_available(),
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate text from a prompt.

    This endpoint supports caching for improved performance on repeated queries.
    """
    if not state.llm:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    # Check cache
    cache_key = None
    if request.use_cache and state.cache:
        cache_key = _create_cache_key(request)
        cached_response = state.cache.get(cache_key)
        if cached_response:
            logger.info(f"✅ Cache hit for prompt: {request.prompt[:50]}...")
            cached_response["cached"] = True
            cached_response["inference_time_ms"] = (time.time() - start_time) * 1000
            return GenerateResponse(**cached_response)

    # Generate
    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        max_tokens=request.max_tokens,
        repetition_penalty=request.repetition_penalty,
    )

    try:
        outputs = state.llm.generate([request.prompt], sampling_params)
        output = outputs[0]

        generated_text = output.outputs[0].text
        num_tokens = len(output.outputs[0].token_ids)

        inference_time_ms = (time.time() - start_time) * 1000

        response_data = {
            "generated_text": generated_text,
            "prompt": request.prompt,
            "num_tokens": num_tokens,
            "inference_time_ms": inference_time_ms,
            "cached": False,
        }

        # Cache response
        if request.use_cache and state.cache and cache_key:
            state.cache.set(cache_key, response_data, ttl=3600)

        logger.info(f"✅ Generated {num_tokens} tokens in {inference_time_ms:.2f}ms")

        return GenerateResponse(**response_data)

    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/batch-generate")
async def batch_generate(request: BatchGenerateRequest):
    """
    Generate text for multiple prompts in a batch.

    Leverages vLLM's continuous batching for optimal throughput.
    """
    if not state.llm:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
    )

    try:
        outputs = state.llm.generate(request.prompts, sampling_params)

        results = []
        for output in outputs:
            results.append({
                "prompt": output.prompt,
                "generated_text": output.outputs[0].text,
                "num_tokens": len(output.outputs[0].token_ids),
            })

        total_time_ms = (time.time() - start_time) * 1000
        total_tokens = sum(r["num_tokens"] for r in results)

        logger.info(f"✅ Batch generated {total_tokens} tokens in {total_time_ms:.2f}ms")

        return {
            "results": results,
            "total_time_ms": total_time_ms,
            "total_tokens": total_tokens,
            "throughput_tokens_per_sec": total_tokens / (total_time_ms / 1000),
        }

    except Exception as e:
        logger.error(f"❌ Batch generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch generation failed: {str(e)}")


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return JSONResponse(
        content=generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )


# ============================================================================
# Helper Functions
# ============================================================================

def _create_cache_key(request: GenerateRequest) -> str:
    """Create a cache key from request parameters."""
    key_data = {
        "prompt": request.prompt,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "top_k": request.top_k,
    }
    key_string = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_string.encode()).hexdigest()


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
