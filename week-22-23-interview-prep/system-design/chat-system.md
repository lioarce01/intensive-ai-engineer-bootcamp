# Case Study: ChatGPT-like Conversational AI System

## 🎯 Problem Statement

**Question**: Design a conversational AI system similar to ChatGPT that can handle multi-turn conversations with context awareness, supporting 100K concurrent users with P95 latency <2s.

## 📋 Step 1: Requirements Clarification (5 min)

### Functional Requirements
✅ Multi-turn conversations with context
✅ Streaming responses (token-by-token)
✅ Conversation history persistence
✅ Multiple conversations per user
✅ Support for various model sizes/capabilities
✅ Basic content moderation

### Non-functional Requirements
- **Scale**: 100K concurrent users, 500 QPS peak
- **Latency**: P95 <2s for first token, ~50 tokens/sec streaming
- **Availability**: 99.9% uptime
- **Cost**: <$0.01 per conversation turn
- **Quality**: Coherent, context-aware responses
- **Privacy**: Secure conversation storage, no cross-user leakage

### Out of Scope (para esta sesión)
❌ Voice input/output
❌ Image generation
❌ Fine-tuning por usuario
❌ Advanced RAG capabilities

## 🏗️ Step 2: High-level Architecture (12 min)

```
┌─────────────────────────────────────────────────────────────┐
│                        User (Web/Mobile)                     │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ↓
┌──────────────────────────────────────────────────────────────┐
│                    Load Balancer (AWS ALB)                    │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ↓
┌──────────────────────────────────────────────────────────────┐
│              API Gateway (Authentication + Rate Limiting)     │
└──────────────────┬──────────────────────────────────────────┘
                   │
           ┌───────┴────────┐
           ↓                ↓
    ┌───────────┐    ┌─────────────┐
    │  Cache    │    │ Application │
    │  (Redis)  │◄───┤   Servers   │
    └───────────┘    └──────┬──────┘
                            │
                     ┌──────┴──────┐
                     ↓             ↓
              ┌─────────────┐  ┌──────────────┐
              │ Conversation│  │  LLM Service │
              │  Database   │  │   (vLLM)     │
              │ (Postgres)  │  └──────┬───────┘
              └─────────────┘         │
                     │                ↓
                     │         ┌─────────────┐
                     │         │ Model Store │
                     │         │    (S3)     │
                     │         └─────────────┘
                     ↓
              ┌──────────────┐
              │  Monitoring  │
              │ (DataDog +   │
              │  Langfuse)   │
              └──────────────┘
```

### Core Components

1. **API Gateway**
   - Authentication (JWT tokens)
   - Rate limiting (per user/tier)
   - Request validation

2. **Application Servers** (Node.js/Python)
   - Business logic
   - Conversation management
   - Context assembly
   - Response streaming

3. **LLM Service** (vLLM cluster)
   - Model serving with continuous batching
   - Multiple model support (GPT-4, Claude, etc.)
   - GPU optimization

4. **Conversation Database** (Postgres)
   - User conversations
   - Message history
   - Metadata (timestamps, model used, etc.)

5. **Cache Layer** (Redis)
   - Session data
   - Recent conversation context
   - Common prompt templates

6. **Monitoring & Analytics**
   - Request/response logging
   - Model performance metrics
   - User analytics

## 🔍 Step 3: Deep Dive into Critical Components (25 min)

### 3.1 Conversation Context Management

**Challenge**: How to efficiently manage context for multi-turn conversations?

**Solution**:

```python
class ConversationManager:
    def __init__(self, max_context_tokens=4000):
        self.max_context_tokens = max_context_tokens
        self.cache = Redis()

    async def get_context(self, conversation_id: str) -> List[Message]:
        """
        Retrieve conversation context with smart truncation
        """
        # Try cache first
        cached = await self.cache.get(f"conv:{conversation_id}")
        if cached:
            return cached

        # Load from DB
        messages = await db.get_conversation_history(conversation_id)

        # Smart truncation strategy
        context = self._truncate_context(messages)

        # Cache for 1 hour
        await self.cache.setex(
            f"conv:{conversation_id}",
            3600,
            context
        )

        return context

    def _truncate_context(self, messages: List[Message]) -> List[Message]:
        """
        Truncate to fit within token limit while preserving:
        1. System message (always)
        2. Most recent messages (priority)
        3. Summarize older messages if needed
        """
        total_tokens = 0
        result = []

        # Always include system message
        if messages[0].role == "system":
            result.append(messages[0])
            total_tokens += messages[0].token_count
            messages = messages[1:]

        # Include recent messages (reverse order)
        for msg in reversed(messages):
            if total_tokens + msg.token_count > self.max_context_tokens:
                break
            result.insert(0, msg)
            total_tokens += msg.token_count

        return result
```

**Key Decisions**:
- **Caching**: Redis cache for hot conversations (reduces DB load)
- **Truncation**: Keep recent messages (recency bias for relevance)
- **Summarization**: For very long conversations, summarize older turns

### 3.2 Streaming Response Implementation

**Challenge**: How to stream tokens efficiently to users?

**Solution using Server-Sent Events (SSE)**:

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI()

@app.post("/v1/chat/completions")
async def chat_completion(request: ChatRequest):
    """
    Streaming chat endpoint
    """
    # Validate and get context
    context = await conversation_mgr.get_context(request.conversation_id)

    # Add user message
    context.append({
        "role": "user",
        "content": request.message
    })

    # Stream response
    return StreamingResponse(
        stream_llm_response(context, request),
        media_type="text/event-stream"
    )

async def stream_llm_response(context, request):
    """
    Generator that streams tokens from LLM
    """
    # Add to conversation DB immediately
    user_msg_id = await db.add_message(
        conversation_id=request.conversation_id,
        role="user",
        content=request.message
    )

    # Start streaming from LLM
    full_response = ""
    assistant_msg_id = None

    async for token in llm_service.generate_stream(
        messages=context,
        model=request.model,
        temperature=request.temperature
    ):
        full_response += token

        # Send SSE event
        yield f"data: {json.dumps({'token': token})}\n\n"

        # Periodic checkpoints (every 10 tokens)
        if len(full_response.split()) % 10 == 0:
            if assistant_msg_id is None:
                assistant_msg_id = await db.add_message(
                    conversation_id=request.conversation_id,
                    role="assistant",
                    content=full_response
                )
            else:
                await db.update_message(assistant_msg_id, full_response)

    # Final save
    await db.update_message(assistant_msg_id, full_response)

    # Update cache
    await cache.delete(f"conv:{request.conversation_id}")

    # Done
    yield f"data: {json.dumps({'done': True})}\n\n"
```

**Key Decisions**:
- **SSE vs WebSockets**: SSE simpler for unidirectional streaming
- **Checkpointing**: Save partial responses to DB (resilience)
- **Cache invalidation**: Clear cache after new message

### 3.3 LLM Service with vLLM

**Challenge**: Serve models efficiently with low latency and high throughput

**Architecture**:

```yaml
# vLLM Deployment Config
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-service
spec:
  replicas: 4  # Auto-scale based on GPU utilization
  template:
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        resources:
          limits:
            nvidia.com/gpu: 1  # A100 40GB
        env:
        - name: MODEL_NAME
          value: "meta-llama/Llama-2-70b-chat-hf"
        - name: TENSOR_PARALLEL_SIZE
          value: "1"
        - name: MAX_MODEL_LEN
          value: "4096"
        - name: GPU_MEMORY_UTILIZATION
          value: "0.9"
        args:
        - --host=0.0.0.0
        - --port=8000
        - --model=$(MODEL_NAME)
        - --tensor-parallel-size=$(TENSOR_PARALLEL_SIZE)
        - --max-model-len=$(MAX_MODEL_LEN)
        - --gpu-memory-utilization=$(GPU_MEMORY_UTILIZATION)
```

**Optimizations**:

1. **Continuous Batching**: vLLM automatically batches requests
2. **PagedAttention**: Efficient KV cache management
3. **Quantization**: Use AWQ or GPTQ for 4-bit (2x throughput)
4. **Multi-model Serving**: Different models for different tiers
   - Free tier: Llama 2 7B (fast, cheap)
   - Pro tier: Llama 2 70B (better quality)
   - Enterprise: Custom fine-tuned models

**Request Routing Logic**:

```python
class ModelRouter:
    def __init__(self):
        self.models = {
            "fast": "llama-2-7b",      # <1s latency
            "balanced": "llama-2-13b",  # <2s latency
            "quality": "llama-2-70b"    # <3s latency
        }

    async def route_request(self, user_tier: str, complexity: float):
        """
        Route to appropriate model based on user tier and query complexity
        """
        if user_tier == "free":
            return self.models["fast"]
        elif user_tier == "pro":
            # Use complexity heuristic
            if complexity < 0.3:
                return self.models["balanced"]
            else:
                return self.models["quality"]
        else:  # enterprise
            return "custom-fine-tuned-model"
```

### 3.4 Content Moderation

**Challenge**: Detect and filter inappropriate content in real-time

**Solution - Multi-layer Moderation**:

```python
class ContentModerator:
    def __init__(self):
        self.keyword_filter = KeywordFilter()
        self.classifier = ModerationClassifier()  # Distilled BERT

    async def moderate_input(self, text: str) -> ModerationResult:
        """
        Fast multi-layer moderation
        """
        # Layer 1: Keyword filter (instant)
        if self.keyword_filter.contains_banned_words(text):
            return ModerationResult(
                approved=False,
                reason="banned_keywords",
                latency_ms=1
            )

        # Layer 2: ML classifier (5-10ms)
        score = await self.classifier.predict(text)

        if score > 0.8:  # High confidence unsafe
            return ModerationResult(
                approved=False,
                reason="unsafe_content",
                confidence=score,
                latency_ms=8
            )
        elif score > 0.5:  # Uncertain - flag for human review
            await self.flag_for_review(text, score)

        return ModerationResult(approved=True, latency_ms=8)

    async def moderate_output(self, text: str) -> str:
        """
        Post-process LLM output
        """
        # Check for hallucinations, PII, harmful content
        # Can be more thorough since streaming already started

        issues = []

        # PII detection
        if self.detect_pii(text):
            issues.append("pii_detected")
            text = self.redact_pii(text)

        # Harmful content
        if self.detect_harmful(text):
            issues.append("harmful_content")
            text = "I cannot provide that information."

        if issues:
            await self.log_moderation_event(text, issues)

        return text
```

**Key Decisions**:
- **Input moderation**: Fast (<10ms) to not block user
- **Output moderation**: Can be slower, runs during streaming
- **Human-in-loop**: Flag uncertain cases for review

## ⚖️ Step 4: Trade-offs & Scaling (8 min)

### Key Trade-offs Made

| Decision | Option A | Option B | Chosen | Reasoning |
|----------|----------|----------|--------|-----------|
| Context Strategy | Load full history | Smart truncation | **Smart truncation** | Balances context quality with token efficiency |
| Streaming | Batch response | Token-by-token | **Token-by-token** | Better UX (perceived latency) |
| Model Serving | OpenAI API | Self-hosted vLLM | **Self-hosted vLLM** | Cost at scale (~10x cheaper) |
| Database | MongoDB | Postgres | **Postgres** | Strong consistency, relational data |
| Caching | Aggressive | Conservative | **Aggressive** | Hot conversations accessed frequently |

### Scaling Plan

**Current (100K users)**:
- 4 vLLM instances (A100 GPUs)
- 10 application servers
- 1 Postgres primary + 2 replicas
- 1 Redis cluster (3 nodes)
- **Cost**: ~$15K/month

**10x Scale (1M users)**:
- **Horizontal scaling**:
  - 40 vLLM instances with auto-scaling
  - 50 application servers
  - Postgres sharding by user_id
  - Redis cluster (10 nodes)
- **Optimizations**:
  - Add CDN for static assets
  - Implement multi-region deployment
  - Use smaller models for simple queries
- **Cost**: ~$120K/month (80% GPU cost)

**Bottleneck Analysis**:
1. **GPU inference** (70% of cost, main bottleneck at scale)
   - Solution: Aggressive caching, model distillation, speculative decoding
2. **Database writes** (conversation history)
   - Solution: Async writes, batch updates, sharding
3. **Network bandwidth** (streaming)
   - Solution: Compression, CDN, regional deployment

## 📊 Step 5: Monitoring & Observability (5 min)

### Key Metrics

**Performance**:
- `latency_first_token_p95`: <2s target
- `latency_full_response_p95`: <30s target
- `tokens_per_second`: ~50 target
- `request_success_rate`: >99.5%

**Quality**:
- `conversation_length_avg`: Engagement metric
- `user_feedback_score`: Thumbs up/down
- `moderation_flag_rate`: Track false positives

**Cost**:
- `cost_per_conversation`: <$0.01 target
- `gpu_utilization`: >70% (efficiency)
- `cache_hit_rate`: >60% (effectiveness)

### Alerts

```yaml
alerts:
  - name: HighLatency
    condition: latency_first_token_p95 > 3s
    severity: critical
    action: Scale up GPU instances

  - name: HighErrorRate
    condition: error_rate > 1%
    severity: critical
    action: Page on-call engineer

  - name: LowCacheHitRate
    condition: cache_hit_rate < 40%
    severity: warning
    action: Review caching strategy

  - name: HighModerationRate
    condition: moderation_flag_rate > 5%
    severity: warning
    action: Review moderation thresholds
```

### Observability Stack

```python
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
import langfuse

# Distributed tracing
tracer = trace.get_tracer(__name__)

@app.post("/v1/chat/completions")
@tracer.start_as_current_span("chat_completion")
async def chat_completion(request: ChatRequest):
    span = trace.get_current_span()
    span.set_attribute("conversation_id", request.conversation_id)
    span.set_attribute("model", request.model)

    # Langfuse for LLM observability
    langfuse_trace = langfuse.trace(
        name="chat_completion",
        user_id=request.user_id,
        metadata={"model": request.model}
    )

    # ... rest of logic

    langfuse_trace.score(
        name="latency",
        value=response_time
    )
```

## ✅ Interview Success Criteria

**Architecture**:
- ✅ Clear separation of concerns (API, business logic, ML service)
- ✅ Scalable design (horizontal scaling, caching, async processing)
- ✅ Resilient (fallbacks, retries, graceful degradation)

**ML-Specific**:
- ✅ Efficient context management
- ✅ Model serving optimization (vLLM, batching)
- ✅ Content moderation (safety)

**Trade-offs**:
- ✅ Articulated cost vs quality decisions
- ✅ Discussed latency vs throughput
- ✅ Explained scaling strategy

**Communication**:
- ✅ Think aloud during design
- ✅ Asked clarifying questions
- ✅ Justified technical decisions

## 🚀 Advanced Extensions (if time permits)

1. **Multi-modal Support**:
   - Image input via CLIP embeddings
   - Vision-language models (LLaVA)

2. **Personalization**:
   - User preference learning
   - Custom system prompts per user

3. **Advanced RAG**:
   - Integrate vector DB for knowledge retrieval
   - Hybrid search for factual grounding

4. **A/B Testing**:
   - Model comparison framework
   - Gradual rollout of new models

5. **Cost Optimization**:
   - Prompt caching (exact + semantic)
   - Speculative decoding (2x speedup)
   - Model distillation (smaller models)

---

**Next**: [Case Study: Recommendation Engine](./recommendation-engine.md)
**Back**: [System Design Framework](./framework.md)
