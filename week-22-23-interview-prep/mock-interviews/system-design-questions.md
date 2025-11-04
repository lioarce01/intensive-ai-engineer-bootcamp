# System Design Interview Questions

## 🎯 Overview

This document contains 25+ real system design questions asked at top AI/ML companies. Practice answering each in 45-60 minutes using the [system design framework](../system-design/framework.md).

## 📊 Difficulty Levels

- 🟢 **Easy**: Core concepts, straightforward architecture
- 🟡 **Medium**: Multiple components, some trade-offs
- 🔴 **Hard**: Complex trade-offs, scale challenges, deep technical depth

---

## 🔵 RAG & Search Systems

### 1. Design a RAG System for Legal Document Search 🟡

**Scenario**: Law firm with 100K legal documents wants semantic search + Q&A

**Key considerations**:
- Accuracy is critical (legal implications)
- Need citation/source tracking
- Multi-tenant (different clients' documents)
- GDPR compliance

**Follow-up questions**:
- How would you handle very long documents (100+ pages)?
- How would you ensure no cross-client data leakage?
- How would you handle updates to existing documents?
- What evaluation metrics would you use?

**Expected discussion**:
- Chunking strategy for long documents
- Hybrid search (semantic + keyword)
- Reranking for precision
- Citation tracking
- Multi-tenancy with namespaces
- Compliance (encryption, access control)

---

### 2. Design a Semantic Search Engine for E-commerce 🟢

**Scenario**: E-commerce site with 1M products, 10K QPS

**Key considerations**:
- Latency: <100ms P95
- Support for filters (price, category, brand)
- Handle typos and synonyms
- Product catalog updates daily

**Follow-up questions**:
- How would you handle the cold start problem?
- How would you evaluate search quality?
- How would you handle seasonal trends?
- How would you scale to 10M products?

**Expected discussion**:
- Vector DB selection (Pinecone, Weaviate)
- Hybrid search (semantic + filters)
- Caching popular queries
- Daily batch updates vs real-time
- Monitoring (click-through rate, conversion)

---

### 3. Design a Code Search Engine (like GitHub Search) 🔴

**Scenario**: Search across 10M+ repositories, support natural language + code queries

**Key considerations**:
- Different programming languages
- Support for both semantic and exact match
- Need to search code, comments, and documentation
- Handle large repositories efficiently

**Follow-up questions**:
- How would you handle syntax-specific searches (e.g., "find all Python functions that use async")?
- How would you rank results?
- How would you keep the index updated with new commits?
- How would you handle private vs public repositories?

**Expected discussion**:
- Code-specific embeddings (CodeBERT, GraphCodeBERT)
- AST parsing for structural search
- Hybrid approach (semantic + regex)
- Incremental indexing (per commit)
- Multi-tenancy and permissions

---

## 🤖 Conversational AI & Chatbots

### 4. Design a Customer Support Chatbot 🟡

**Scenario**: E-commerce company wants AI chatbot to handle 80% of support tickets

**Key considerations**:
- Multi-turn conversations with context
- Integration with order database, inventory, FAQ
- Need to escalate to human when necessary
- Support multiple languages

**Follow-up questions**:
- How would you decide when to escalate to a human?
- How would you handle multi-turn conversations?
- How would you ensure the bot doesn't make up information?
- How would you evaluate quality?

**Expected discussion**:
- RAG for FAQ + knowledge base
- Function calling for order lookup
- Conversation state management
- Confidence scoring for escalation
- Guardrails (no hallucinations)
- Multi-language support (translation layer)

---

### 5. Design a ChatGPT-like System 🔴

**Scenario**: Build conversational AI supporting 100K concurrent users, streaming responses

**Key considerations**:
- Multi-turn conversations
- Low latency (<2s first token)
- Conversation history management
- Content moderation

**Follow-up questions**:
- How would you handle very long conversations (100+ turns)?
- How would you optimize cost (token usage)?
- How would you prevent prompt injection attacks?
- How would you scale to 1M users?

**Expected discussion**:
- See detailed case study: [Chat System Design](../system-design/chat-system.md)
- vLLM for model serving
- Context truncation strategies
- Streaming with SSE
- Redis for conversation caching
- Moderation pipeline

---

### 6. Design a Code Generation API (like GitHub Copilot) 🔴

**Scenario**: Real-time code suggestions, 1000 QPS, <200ms latency

**Key considerations**:
- Real-time (low latency)
- Context-aware (surrounding code)
- Multi-language support
- Security (no generated secrets/credentials)

**Follow-up questions**:
- How would you optimize for latency?
- How would you personalize suggestions per user?
- How would you prevent generating malicious code?
- How would you handle rate limiting?

**Expected discussion**:
- Model selection (smaller models for speed)
- Context window management
- Speculative decoding for speed
- Caching common patterns
- Security scanning (secrets detection)
- A/B testing different models

---

## 📝 Content Generation & Moderation

### 7. Design a Content Moderation System 🟡

**Scenario**: Social media platform needs to flag harmful content in real-time

**Key considerations**:
- Real-time (<100ms)
- Handle text, images, and video
- Low false positive rate (< 1%)
- Handle multiple languages

**Follow-up questions**:
- How would you handle adversarial attacks (users trying to bypass)?
- How would you improve the system over time?
- How would you handle appeals (false positives)?
- What's the trade-off between precision and recall?

**Expected discussion**:
- Multi-layer approach (keywords → ML → human review)
- Different models for text/image/video
- Confidence thresholds
- Human-in-the-loop for uncertain cases
- Active learning from appeals
- Adversarial robustness

---

### 8. Design a News Article Summarization Service 🟢

**Scenario**: Summarize news articles for a mobile app, 1000 articles/day

**Key considerations**:
- Quality summaries (3-5 sentences)
- Fast processing (batch overnight)
- Handle different article lengths
- Support multiple languages

**Follow-up questions**:
- How would you evaluate summary quality?
- How would you handle very long articles (10K+ words)?
- How would you personalize summaries per user?
- How would you scale to 1M articles/day?

**Expected discussion**:
- Extractive vs abstractive summarization
- Model selection (T5, BART, GPT)
- Batch processing vs real-time
- Quality evaluation (ROUGE, human eval)
- Caching summaries
- Multi-language models

---

### 9. Design a Personalized Email Generator 🟡

**Scenario**: Generate personalized marketing emails for 10M users

**Key considerations**:
- Personalization (user history, preferences)
- Batch generation (overnight)
- A/B testing different versions
- Spam filter avoidance

**Follow-up questions**:
- How would you incorporate user data?
- How would you evaluate which emails perform better?
- How would you prevent generating offensive content?
- How would you scale to 100M users?

**Expected discussion**:
- Template + LLM for personalization
- Prompt engineering with user data
- Batch generation (parallel processing)
- A/B testing framework
- Quality gates (spam score, tone)
- Cost optimization (caching similar users)

---

## 🎯 Recommendation & Ranking Systems

### 10. Design a Content Recommendation Engine 🟡

**Scenario**: Video streaming platform, recommend next video to watch

**Key considerations**:
- Personalized (user history, preferences)
- Real-time (<100ms)
- Diverse recommendations (not just similar)
- Cold start problem (new users/videos)

**Follow-up questions**:
- How would you balance personalization vs diversity?
- How would you handle the cold start problem?
- How would you evaluate recommendation quality?
- How would you incorporate real-time signals (clicks, skips)?

**Expected discussion**:
- Collaborative filtering + content-based
- Embedding-based retrieval (fast candidate generation)
- Reranking with neural network
- Real-time features (recent clicks)
- A/B testing
- Cold start strategies (popular items, explore/exploit)

---

### 11. Design a Job Recommendation System (like LinkedIn) 🔴

**Scenario**: Match candidates with jobs, 1M candidates, 100K jobs

**Key considerations**:
- Two-sided marketplace (candidates and jobs)
- Skills matching, location, salary
- Real-time updates (new jobs posted)
- Fairness (avoid bias)

**Follow-up questions**:
- How would you ensure fairness in recommendations?
- How would you handle incomplete profiles?
- How would you optimize for both sides (candidate satisfaction + employer satisfaction)?
- How would you scale to 10M candidates?

**Expected discussion**:
- Two-tower model (candidate embedding + job embedding)
- Hybrid matching (semantic + filters)
- Real-time indexing (new jobs)
- Bias mitigation (demographic parity)
- Multi-objective optimization (click-through + apply rate)

---

## 🔬 Model Training & MLOps

### 12. Design a Model Training Pipeline 🟡

**Scenario**: Train and deploy LLMs for your company, weekly retraining

**Key considerations**:
- Data collection and curation
- Distributed training (multiple GPUs)
- Model evaluation and testing
- Deployment pipeline

**Follow-up questions**:
- How would you ensure data quality?
- How would you monitor model performance over time?
- How would you handle model versioning?
- How would you rollback a bad model?

**Expected discussion**:
- Data pipeline (collection → cleaning → versioning)
- Training infrastructure (PyTorch + DeepSpeed)
- Evaluation (automated + human)
- Model registry (MLflow)
- Gradual rollout with A/B testing
- Monitoring (drift detection)

---

### 13. Design an A/B Testing Platform for ML Models 🔴

**Scenario**: Test new models in production with minimal risk

**Key considerations**:
- Traffic splitting (control vs treatment)
- Metric collection and analysis
- Statistical significance testing
- Gradual rollout

**Follow-up questions**:
- How would you handle multi-variate testing (testing 5 models)?
- How would you detect if a model is significantly worse?
- How would you minimize user impact during testing?
- How long should an A/B test run?

**Expected discussion**:
- Traffic routing (hash-based, consistent)
- Metric instrumentation (latency, quality, engagement)
- Statistical testing (t-test, bootstrap)
- Early stopping (if model is clearly worse)
- Gradual rollout (10% → 50% → 100%)

---

### 14. Design a Feature Store for ML 🔴

**Scenario**: Centralized feature management for 100+ ML models

**Key considerations**:
- Online + offline features
- Low latency (<10ms online)
- Feature versioning and lineage
- Data consistency (train/serve)

**Follow-up questions**:
- How would you handle feature drift?
- How would you ensure no data leakage?
- How would you scale to 1000s of features?
- How would you handle real-time features?

**Expected discussion**:
- Architecture (Feast, Tecton)
- Online store (Redis) + offline store (S3/BigQuery)
- Point-in-time joins (avoid leakage)
- Feature versioning
- Monitoring (feature drift, nulls)
- Streaming features (Kafka → Feature store)

---

## 🌐 Multimodal & Advanced

### 15. Design an Image Search Engine 🟡

**Scenario**: Search for similar images in a database of 10M images

**Key considerations**:
- Visual similarity search
- Support for text queries ("red dress")
- Low latency (<200ms)
- Handle different image sizes

**Follow-up questions**:
- How would you handle text queries?
- How would you deduplicate similar images?
- How would you scale to 100M images?
- How would you handle copyright detection?

**Expected discussion**:
- Image embeddings (CLIP, ResNet)
- Vector DB for similarity search
- Text-to-image search with CLIP
- Perceptual hashing for deduplication
- Approximate nearest neighbors (HNSW)
- Caching popular queries

---

### 16. Design a Real-time Translation Service 🟡

**Scenario**: Translate chat messages in real-time, support 50 languages

**Key considerations**:
- Real-time (<200ms)
- High quality (BLEU > 30)
- Support 50+ languages
- Handle informal text (slang, typos)

**Follow-up questions**:
- How would you handle low-resource languages?
- How would you detect the source language?
- How would you handle offensive content?
- How would you scale to 10K QPS?

**Expected discussion**:
- Model selection (mBART, NLLB, GPT)
- Language detection (fastText)
- Caching common phrases
- Quality evaluation (BLEU, human eval)
- Moderation pipeline
- Cost optimization (smaller models for high-resource pairs)

---

### 17. Design a Voice Assistant (like Alexa) 🔴

**Scenario**: Voice-activated assistant for smart home control

**Key considerations**:
- Speech-to-text (real-time)
- Intent detection
- Multi-turn conversations
- Text-to-speech (natural voice)
- Low latency (<1s end-to-end)

**Follow-up questions**:
- How would you handle background noise?
- How would you personalize per user (voice recognition)?
- How would you handle offline mode?
- How would you integrate with smart home devices?

**Expected discussion**:
- Pipeline: STT → NLU → Action → TTS
- Edge vs cloud processing (latency vs accuracy)
- Wake word detection (local)
- Intent classification + slot filling
- Function calling for device control
- Voice biometrics for personalization

---

## 💰 Cost & Efficiency

### 18. Design a Cost-Optimized LLM API 🟡

**Scenario**: Serve LLM API at <$0.001 per request

**Key considerations**:
- Aggressive caching
- Model selection (size vs quality)
- Token optimization
- Infrastructure cost

**Follow-up questions**:
- How would you cache responses semantically (not just exact match)?
- How would you route queries to different models?
- How would you optimize prompt length?
- How would you scale without increasing cost?

**Expected discussion**:
- Semantic caching (embed query → find similar → return cached)
- Model routing (small model for simple, large for complex)
- Prompt compression (remove redundancy)
- Batching requests
- Spot instances for training
- Quantization (4-bit models)

---

### 19. Design a Batch Processing System for LLMs 🟢

**Scenario**: Process 1M documents overnight with LLMs

**Key considerations**:
- Cost-efficient (batch processing)
- Fault-tolerant (resume from failures)
- Progress tracking
- Quality assurance

**Follow-up questions**:
- How would you handle failures (retries)?
- How would you monitor progress?
- How would you optimize for cost?
- How would you ensure quality?

**Expected discussion**:
- Queue-based architecture (SQS, Kafka)
- Worker pool with auto-scaling
- Checkpointing (resume from failure)
- Rate limiting (avoid API throttling)
- Batch API usage (OpenAI batch)
- Sampling for quality checks

---

## 🏢 Enterprise & Multi-tenant

### 20. Design a Multi-tenant RAG System 🔴

**Scenario**: SaaS product serving 1000+ customers, each with their own documents

**Key considerations**:
- Data isolation (no cross-customer leakage)
- Per-customer customization
- Cost allocation
- Varying scale (some customers have 10K docs, others 1M)

**Follow-up questions**:
- How would you ensure no data leakage?
- How would you handle customers with very different scales?
- How would you allocate costs per customer?
- How would you allow custom models per customer?

**Expected discussion**:
- Namespaces in vector DB (per customer)
- Separate indexes for large customers
- Resource quotas and rate limiting
- Cost tracking (tokens, storage, compute)
- Custom embedding models (optional)
- Monitoring per customer

---

### 21. Design an AI-powered Compliance System 🔴

**Scenario**: Scan documents for regulatory compliance (GDPR, HIPAA)

**Key considerations**:
- High accuracy (compliance is critical)
- Explain decisions (why flagged?)
- Audit trail
- Handle 10K+ documents/day

**Follow-up questions**:
- How would you handle false positives?
- How would you keep up with changing regulations?
- How would you explain model decisions?
- How would you ensure auditability?

**Expected discussion**:
- Rule-based + ML hybrid
- High recall (flag all potential issues)
- Human review for flagged items
- Explainability (highlight specific clauses)
- Version control for rules
- Audit logs (immutable)

---

## 🎨 Creative Applications

### 22. Design a Code Review Assistant 🟡

**Scenario**: Automatically review pull requests and suggest improvements

**Key considerations**:
- Analyze code for bugs, style, performance
- Generate helpful comments
- Integrate with GitHub
- Handle multiple languages

**Follow-up questions**:
- How would you avoid annoying developers with too many comments?
- How would you learn from accepted/rejected suggestions?
- How would you handle false positives?
- How would you scale to 1000s of PRs/day?

**Expected discussion**:
- Static analysis + LLM
- Code embeddings for context
- Comment generation with examples
- Confidence scoring (only surface high-confidence)
- Feedback loop (learn from dismissals)
- Integration (GitHub Actions)

---

### 23. Design a Meeting Summarization System 🟡

**Scenario**: Transcribe meetings and generate summaries + action items

**Key considerations**:
- Speech-to-text (multiple speakers)
- Generate summary + action items
- Real-time or post-meeting
- Handle different accents/languages

**Follow-up questions**:
- How would you handle speaker diarization (who said what)?
- How would you extract action items?
- How would you handle multi-language meetings?
- How would you ensure privacy?

**Expected discussion**:
- STT (Whisper) + diarization (Pyannote)
- Summarization (extractive + abstractive)
- Action item extraction (NER + classification)
- Privacy (on-device processing option)
- Real-time vs post-processing

---

### 24. Design a Smart Document Editor (like Notion AI) 🟡

**Scenario**: AI-powered document editor with auto-completion, rewriting, etc.

**Key considerations**:
- Real-time suggestions (<200ms)
- Context-aware (surrounding text)
- Multiple features (complete, rewrite, summarize)
- Offline support

**Follow-up questions**:
- How would you handle latency?
- How would you personalize per user?
- How would you handle undo/redo?
- How would you scale to 1M users?

**Expected discussion**:
- Smaller models for speed
- Context window (surrounding paragraphs)
- Streaming suggestions
- Client-side caching
- User preference learning
- Conflict resolution (concurrent edits)

---

### 25. Design a SQL Query Generator from Natural Language 🔴

**Scenario**: Users ask questions in plain English, system generates SQL

**Key considerations**:
- Accurate SQL generation
- Handle complex queries (joins, subqueries)
- Database schema understanding
- Explain generated query

**Follow-up questions**:
- How would you handle ambiguous questions?
- How would you prevent SQL injection?
- How would you handle very large schemas?
- How would you verify query correctness?

**Expected discussion**:
- Text-to-SQL models (GPT-4, specialized models)
- Schema context (provide relevant tables)
- Few-shot prompting with examples
- Query validation (syntax + permissions)
- Explain plan visualization
- Sandbox execution first

---

## 📚 How to Practice

### Step 1: Understand the Question (5 min)
- Read carefully
- Identify core challenge
- Note key constraints

### Step 2: Solo Practice (45 min)
- Follow the [system design framework](../system-design/framework.md)
- Draw on paper or whiteboard
- Think out loud (practice verbalizing)

### Step 3: Self-Review (15 min)
- Did you cover all components?
- Did you discuss trade-offs?
- Did you mention monitoring?
- Compare with example solutions

### Step 4: Mock Interview (60 min)
- Find a partner (friend, colleague, online)
- Take turns being interviewer/interviewee
- Get feedback

### Step 5: Iterate
- Identify weak areas
- Study specific topics (e.g., vector DBs)
- Practice again

---

## 🎯 Evaluation Rubric

| Criteria | Poor | Good | Excellent |
|----------|------|------|-----------|
| **Clarification** | Jumps to solution | Asks some questions | Asks thorough questions, clarifies ambiguities |
| **Architecture** | Missing components | Has main components | Comprehensive, considers failure modes |
| **Trade-offs** | No discussion | Mentions some | Deep discussion, justifies choices |
| **Scale** | Ignores scale | Mentions scaling | Detailed scaling strategy |
| **Communication** | Unclear, jumps around | Clear, structured | Excellent, thinks aloud, engages interviewer |
| **Depth** | Surface-level | Good depth in 1-2 areas | Deep technical knowledge across multiple areas |

---

**Next**: [Coding Questions](./coding-questions.md)
**Back**: [Debugging Methodology](../debugging/methodology.md)
