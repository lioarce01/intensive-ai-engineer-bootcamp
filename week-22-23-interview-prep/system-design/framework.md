# Framework for AI System Design Interviews

## 📋 Overview

Este framework te guía paso a paso para abordar cualquier pregunta de system design de AI/ML. La clave es ser **estructurado**, **comunicativo** y demostrar **profundidad técnica**.

## ⏱️ Time Management (45-60 min típico)

| Fase | Tiempo | Objetivo |
|------|--------|----------|
| 1. Clarification & Requirements | 5-8 min | Entender el problema completamente |
| 2. High-level Design | 10-15 min | Arquitectura general y componentes |
| 3. Deep Dive | 20-30 min | Detalles técnicos de 2-3 componentes clave |
| 4. Trade-offs & Scaling | 5-10 min | Discutir alternativas y scaling |
| 5. Q&A | 5 min | Responder preguntas del entrevistador |

## 🎯 Phase 1: Clarification & Requirements (5-8 min)

**Objetivo**: Entender exactamente qué estás construyendo

### Functional Requirements
- ¿Qué debe hacer el sistema exactamente?
- ¿Cuáles son los casos de uso principales?
- ¿Qué inputs y outputs espera el sistema?
- ¿Hay requisitos de UI/UX?

### Non-functional Requirements
- **Scale**: ¿Cuántos usuarios? ¿QPS esperado?
- **Latency**: ¿P50/P95/P99 targets? (ej: <100ms, <500ms)
- **Availability**: ¿99.9% uptime? ¿24/7?
- **Cost**: ¿Budget constraints? ¿Cost per request?
- **Quality**: ¿Qué accuracy/precision esperamos?
- **Privacy**: ¿PII handling? ¿GDPR compliance?

### ML-Specific Questions
- ¿El modelo ya existe o hay que entrenarlo?
- ¿Offline o online inference?
- ¿Batch o real-time processing?
- ¿Feedback loop para mejora continua?
- ¿Qué pasa si el modelo no está disponible? (fallback)

**Ejemplo**:
```
Pregunta: "Design a RAG system for customer support"

Buenos clarifications:
- ¿Cuántos documentos? (10K vs 10M es muy diferente)
- ¿Latency target? (<300ms, <1s, <5s?)
- ¿Solo texto o también imágenes/PDFs?
- ¿Qué idiomas? (multilingüe agrega complejidad)
- ¿Updates en tiempo real o batch diario?
- ¿Cuántos queries por segundo esperamos?
```

## 🏗️ Phase 2: High-level Design (10-15 min)

**Objetivo**: Diseñar la arquitectura end-to-end

### Step 1: Identify Major Components

Componentes típicos en sistemas de AI:

1. **API Layer**: REST/GraphQL/gRPC
2. **Authentication/Authorization**: API keys, JWT, OAuth
3. **Rate Limiting**: Token bucket, leaky bucket
4. **Request Processing**: Input validation, preprocessing
5. **Model Serving**: Inference engine (vLLM, TGI, SageMaker)
6. **Caching**: Redis, Memcached (results, embeddings)
7. **Database**: Vector DB (Pinecone, Weaviate), SQL/NoSQL
8. **Message Queue**: Kafka, RabbitMQ (async processing)
9. **Monitoring**: Logs, metrics, traces (Datadog, Prometheus)
10. **Storage**: S3, GCS (models, datasets)

### Step 2: Draw High-level Architecture

**Template típico**:

```
User → Load Balancer → API Gateway → Application Servers
                            ↓
                       Rate Limiter
                            ↓
                    ┌───────┴───────┐
                    ↓               ↓
              Cache (Redis)    ML Service
                    ↓               ↓
              Vector DB      Model Serving
                    ↓               ↓
             Monitoring & Logging ←──┘
                    ↓
             Analytics & Feedback
```

### Step 3: Define Data Flow

Describe el flujo end-to-end:

**Request Path**:
1. User sends request → API Gateway
2. Authentication & rate limiting
3. Check cache (cache hit → return immediately)
4. Cache miss → Process request
5. Call ML service / vector search
6. Post-process results
7. Store in cache
8. Return to user
9. Log metrics & traces

**Training/Update Path** (si aplica):
1. Collect user feedback
2. Store in data warehouse
3. Periodic retraining / fine-tuning
4. A/B testing new model
5. Gradual rollout

## 🔍 Phase 3: Deep Dive (20-30 min)

**Objetivo**: Demostrar profundidad técnica en 2-3 componentes críticos

### Cómo elegir qué deep dive hacer:
- Pregunta al entrevistador: "¿Hay algún componente que le gustaría que profundice?"
- Si no, elige los **más críticos para el sistema**: model serving, vector search, caching, etc.

### Deep Dive Topics Comunes

#### 1. Model Serving
- **Framework**: vLLM, TGI, TensorRT, TorchServe
- **Optimizations**:
  - Quantization (4-bit, 8-bit)
  - Batching (continuous batching para LLMs)
  - KV cache management
  - Speculative decoding
- **Deployment**:
  - Kubernetes con autoscaling
  - GPU allocation y scheduling
  - Health checks y graceful shutdown
- **Fallbacks**:
  - Cascade a smaller model
  - Cached responses
  - Rule-based fallback

#### 2. Vector Search / RAG
- **Indexing**:
  - Algorithm: HNSW, IVF, Product Quantization
  - Incremental updates vs full rebuild
  - Multi-tenancy (namespace per user/company)
- **Query**:
  - Hybrid search (semantic + keyword)
  - Query expansion / rewriting
  - Reranking with cross-encoder
- **Optimization**:
  - Embedding caching
  - Pre-filtering metadata
  - Top-k optimization (nprobe tuning)

#### 3. Caching Strategy
- **What to cache**:
  - Embeddings (queries, documents)
  - LLM responses (exact match + semantic match)
  - Intermediate results
- **Cache invalidation**:
  - TTL (Time to Live)
  - LRU (Least Recently Used)
  - Manual invalidation on updates
- **Implementation**:
  - Redis with clustering
  - Multi-level cache (L1: in-memory, L2: Redis)
  - Cache warming strategies

#### 4. Data Pipeline
- **Ingestion**:
  - Batch (S3 → ETL → Vector DB)
  - Streaming (Kafka → Processing → Vector DB)
  - CDC (Change Data Capture) for databases
- **Processing**:
  - Chunking strategies (fixed, semantic, recursive)
  - Metadata extraction
  - Deduplication
- **Monitoring**:
  - Data quality checks
  - Schema validation
  - Anomaly detection

## ⚖️ Phase 4: Trade-offs & Scaling (5-10 min)

**Objetivo**: Demostrar que entiendes los trade-offs

### Common Trade-offs

| Decision | Option A | Option B | Trade-off |
|----------|----------|----------|-----------|
| Model size | Small (7B) | Large (70B) | Latency vs quality |
| Caching | Aggressive | Conservative | Memory vs freshness |
| Indexing | HNSW | IVF | Speed vs memory |
| Search | Semantic only | Hybrid | Simplicity vs quality |
| Deployment | Single region | Multi-region | Cost vs latency |
| Database | SQL | NoSQL | Consistency vs scale |

### Scaling Considerations

**Horizontal Scaling**:
- Stateless API servers → scale easily
- Load balancing (round-robin, least connections)
- Auto-scaling based on metrics (CPU, QPS)

**Vertical Scaling**:
- Larger GPU instances for models
- More memory for vector DB
- Eventually hits limits → need horizontal

**Bottleneck Analysis**:
1. **If high latency**: Profile the critical path
   - Is it model inference? → Optimize model
   - Is it vector search? → Optimize index
   - Is it network? → Add caching

2. **If high cost**: Analyze cost breakdown
   - Model serving most expensive? → Use smaller model or caching
   - Vector DB expensive? → Optimize storage
   - API calls expensive? → Batch processing

3. **If low quality**: Debug the quality
   - Bad retrieval? → Improve chunking/embeddings
   - Bad generation? → Better prompts or fine-tune
   - Hallucinations? → Add verification layer

### Example Scaling Path

```
Phase 1 (MVP): Monolith, single server, SQLite
   ↓ (1K users)
Phase 2: Separate API + ML service, Postgres, Redis
   ↓ (10K users)
Phase 3: Microservices, managed vector DB, load balancer
   ↓ (100K users)
Phase 4: Multi-region, auto-scaling, CDN
   ↓ (1M+ users)
```

## 🎤 Phase 5: Q&A (5 min)

Preguntas comunes del entrevistador:

**Architecture**:
- "¿Por qué elegiste X en lugar de Y?"
- "¿Qué pasa si este componente falla?"
- "¿Cómo manejarías un 10x de tráfico?"

**ML-Specific**:
- "¿Cómo evaluarías la calidad del sistema?"
- "¿Cómo detectarías model drift?"
- "¿Cómo harías A/B testing de modelos?"

**Trade-offs**:
- "¿Cuál es el bottleneck principal?"
- "¿Cómo reducirías costos sin perder calidad?"
- "¿Dónde agregarías observability?"

## 🧠 Mental Models

### STAR Framework (para decisiones)
- **S**ituation: Describe el contexto
- **T**ask: Qué necesitas lograr
- **A**ction: Qué solución propones
- **R**esult: Qué impacto esperas (latency, cost, quality)

### CAP Theorem (para trade-offs)
- **C**onsistency
- **A**vailability
- **P**artition tolerance

En sistemas distribuidos, solo puedes tener 2 de 3.

### Think Aloud
- Verbaliza tu proceso de pensamiento
- Explica tus assumptions
- Menciona alternativas que consideraste

## ✅ Interview Checklist

### Before Drawing
- [ ] Clarificaste todos los requisitos
- [ ] Entiendes scale, latency, cost targets
- [ ] Identificaste los casos de uso principales

### During Design
- [ ] Empezaste con high-level architecture
- [ ] Identificaste todos los componentes principales
- [ ] Definiste data flow claramente
- [ ] Hiciste deep dive en 2-3 componentes
- [ ] Discutiste trade-offs abiertamente

### Before Finishing
- [ ] Mencionaste monitoring/logging
- [ ] Discutiste failure scenarios
- [ ] Hablaste sobre scaling
- [ ] Consideraste cost optimization
- [ ] Mencionaste testing/validation

## 🚫 Common Mistakes

1. **Jumping to details too fast**: Empieza con high-level siempre
2. **Not asking questions**: Clarifica requisitos antes de diseñar
3. **Silent designing**: Think aloud, comunica tu proceso
4. **Ignoring non-functional requirements**: Scale, latency, cost son críticos
5. **One-size-fits-all**: No hay "perfect solution", solo trade-offs
6. **Over-engineering MVP**: Start simple, then scale
7. **Forgetting ML-specific considerations**: Model drift, feedback loops, evaluation

## 📝 Example Framework Application

**Question**: "Design a semantic search system for e-commerce products"

**Step 1 - Clarification** (5 min):
- Scale: 1M products, 10K QPS
- Latency: <100ms P95
- Quality: Top 10 results with >80% relevance
- Updates: Daily batch updates of product catalog

**Step 2 - High-level** (12 min):
```
User Query → API Gateway → Search Service
                              ↓
                    ┌─────────┴─────────┐
                    ↓                   ↓
              Cache (Redis)      Vector DB (Pinecone)
                                        ↓
                                 Product Catalog
                                        ↓
                                  Reranking Model
                                        ↓
                                Return Top 10
```

**Step 3 - Deep Dive** (25 min):
- Vector DB: HNSW index, batch updates, monitoring
- Reranking: Cross-encoder for top 100 → top 10
- Caching: Cache popular queries, 24h TTL

**Step 4 - Trade-offs** (8 min):
- Chose Pinecone (managed) vs self-hosted (Weaviate)
- Reranking adds 20ms but +15% relevance
- Daily updates vs real-time (cost vs freshness)

**Step 5 - Q&A** (5 min):
- How to handle cold start? → Popular items cache
- How to evaluate? → Click-through rate, conversion
- How to scale? → Horizontal scaling + sharding

---

## 🎯 Practice Template

Use this template to practice:

```markdown
## System Design: [Problem Name]

### Requirements
**Functional**:
-
-

**Non-functional**:
- Scale:
- Latency:
- Cost:

### High-level Architecture
[Draw diagram here]

### Components
1. **API Layer**:
2. **ML Service**:
3. **Database**:
4. **Caching**:
5. **Monitoring**:

### Data Flow
Request:
Response:

### Deep Dive
**Component 1**:
- Implementation:
- Optimization:

**Component 2**:
- Implementation:
- Optimization:

### Trade-offs
- Decision 1: X vs Y → Chose X because...
- Decision 2: ...

### Scaling
- Current:
- 10x scale:
- Bottlenecks:

### Monitoring
- Metrics:
- Alerts:
- Dashboards:
```

---

**Next**: [Case Study: Chat System](./chat-system.md)
