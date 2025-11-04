# Week 19-20 Projects

## Main Project: Multimodal Safety App

A production-ready multimodal application that processes text, images, and audio with comprehensive safety guardrails.

### Features

- **Multimodal Understanding**
  - Image analysis with CLIP
  - Audio transcription with Whisper
  - Cross-modal search and reasoning

- **Safety Layer**
  - Content moderation (toxicity, hate speech)
  - PII detection and redaction
  - Bias monitoring
  - Input validation (prompt injection, jailbreak detection)

- **Production Ready**
  - FastAPI backend with async processing
  - Rate limiting and authentication
  - Comprehensive logging and monitoring
  - Error handling and graceful degradation

### Quick Start

```bash
cd multimodal-safety-app

# Install dependencies
pip install -r requirements.txt

# Download models
python src/download_models.py

# Run server
uvicorn src.main:app --reload

# Run tests
pytest tests/
```

### API Endpoints

- `POST /api/v1/analyze/image` - Analyze image with text query
- `POST /api/v1/analyze/audio` - Transcribe and analyze audio
- `POST /api/v1/search` - Cross-modal search
- `POST /api/v1/vqa` - Visual Question Answering
- `GET /api/v1/health` - Health check

### Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────┐
│      FastAPI Server         │
├─────────────────────────────┤
│  Input Validation Layer     │
│  - Prompt Injection Check   │
│  - Jailbreak Detection      │
│  - Rate Limiting            │
└──────┬──────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│  Multimodal Processing      │
├─────────────────────────────┤
│  • CLIP (Image + Text)      │
│  • Whisper (Audio)          │
│  • Cross-Modal Fusion       │
└──────┬──────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│    Safety Layer             │
├─────────────────────────────┤
│  • Toxicity Filter          │
│  • PII Detector             │
│  • Bias Monitor             │
│  • Hallucination Check      │
└──────┬──────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│  Logging & Monitoring       │
│  - Audit logs               │
│  - Performance metrics      │
│  - Safety alerts            │
└─────────────────────────────┘
```

### Project Structure

```
multimodal-safety-app/
├── src/
│   ├── main.py                 # FastAPI application
│   ├── models/
│   │   ├── clip_model.py       # CLIP wrapper
│   │   ├── whisper_model.py    # Whisper wrapper
│   │   └── fusion.py           # Multimodal fusion
│   ├── safety/
│   │   ├── toxicity.py         # Toxicity detection
│   │   ├── pii_detector.py     # PII protection
│   │   ├── bias_monitor.py     # Bias monitoring
│   │   └── input_validator.py  # Input validation
│   ├── api/
│   │   ├── routes.py           # API endpoints
│   │   └── schemas.py          # Pydantic models
│   └── utils/
│       ├── logging.py          # Logging setup
│       └── monitoring.py       # Metrics collection
├── tests/
│   ├── test_safety.py          # Safety layer tests
│   ├── test_models.py          # Model tests
│   └── test_api.py             # API tests
├── config/
│   ├── settings.py             # Configuration
│   └── logging.yaml            # Logging config
├── requirements.txt
└── README.md
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test suite
pytest tests/test_safety.py -v

# Run integration tests
pytest tests/integration/ -v
```

### Deployment

```bash
# Build Docker image
docker build -t multimodal-safety-app .

# Run container
docker run -p 8000:8000 multimodal-safety-app

# Docker Compose (with Redis, PostgreSQL)
docker-compose up
```

### Monitoring

The application includes:

- **Prometheus metrics** at `/metrics`
- **Health checks** at `/health`
- **Audit logs** in `logs/audit.log`
- **Performance tracking** via middleware

### Safety Configuration

Edit `config/settings.py` to adjust:

```python
# Toxicity thresholds
TOXICITY_THRESHOLD = 0.7

# PII detection sensitivity
PII_ENABLED = True
PII_ENTITIES = ["EMAIL", "PHONE", "SSN", "CREDIT_CARD"]

# Rate limiting
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_WINDOW = 3600  # 1 hour

# Bias monitoring
BIAS_MONITORING_ENABLED = True
FAIRNESS_THRESHOLD = 0.8
```

### Performance Targets

- API response time: < 2s (P95)
- Throughput: > 100 requests/minute
- Availability: > 99.9%
- Safety recall: > 95% for toxic content

### License

MIT

### Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## Additional Exercises

### Exercise 1: Visual Search Engine
Build an image search engine using CLIP embeddings.

### Exercise 2: Audio Classifier
Create an audio event classifier using Whisper features.

### Exercise 3: Bias Audit Tool
Implement a tool to audit model outputs for demographic bias.

### Exercise 4: Content Moderation Dashboard
Build a web dashboard for reviewing flagged content.

### Exercise 5: Multimodal Chatbot
Create a chatbot that can discuss images, audio, and text.

---

**See the main project in `multimodal-safety-app/` to get started!**
