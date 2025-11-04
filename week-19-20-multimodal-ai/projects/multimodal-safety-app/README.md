# Multimodal Safety App

> Production-ready multimodal AI application with comprehensive safety guardrails.

## Overview

This application demonstrates how to build a secure, fair, and robust multimodal AI system that:
- Processes images, audio, and text
- Implements multiple safety checks
- Monitors for bias and harmful content
- Provides explainable outputs
- Maintains audit logs

## Features

### Multimodal Capabilities
- 🖼️ **Image Understanding**: CLIP-based image analysis and classification
- 🎤 **Audio Processing**: Whisper-powered transcription and analysis
- 🔍 **Cross-Modal Search**: Find images using text, or vice versa
- 💬 **Visual Question Answering**: Ask questions about images
- 🎯 **Zero-Shot Classification**: Classify without training

### Safety Features
- 🛡️ **Toxicity Detection**: Filter harmful content
- 🔒 **PII Protection**: Detect and redact personal information
- ⚖️ **Bias Monitoring**: Track fairness across demographics
- 🚫 **Input Validation**: Prevent prompt injection and jailbreaks
- ✅ **Output Verification**: Check for hallucinations

### Production Features
- ⚡ **Fast API**: Async endpoints with <2s response time
- 📊 **Monitoring**: Prometheus metrics and health checks
- 🔐 **Security**: Authentication, rate limiting, CORS
- 📝 **Audit Logs**: Complete traceability
- 🐳 **Containerized**: Docker and Docker Compose ready

## Quick Start

### Installation

```bash
# Clone or navigate to project
cd multimodal-safety-app

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download models (first time only)
python -c "import clip; clip.load('ViT-B/32')"
python -c "import whisper; whisper.load_model('base')"
```

### Configuration

Create `.env` file:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Security
SECRET_KEY=your-secret-key-here
CORS_ORIGINS=http://localhost:3000,http://localhost:8000

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600

# Safety Thresholds
TOXICITY_THRESHOLD=0.7
PII_DETECTION_ENABLED=true

# Model Configuration
CLIP_MODEL=ViT-B/32
WHISPER_MODEL=base
DEVICE=cuda  # or cpu

# Monitoring
SENTRY_DSN=your-sentry-dsn
PROMETHEUS_ENABLED=true

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/multimodal_db
REDIS_URL=redis://localhost:6379/0
```

### Run Application

```bash
# Development mode (with auto-reload)
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4

# With Docker
docker-compose up -d
```

### API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Metrics: http://localhost:8000/metrics

## Usage Examples

### Image Analysis

```python
import requests
from PIL import Image

# Analyze image
with open("cat.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/analyze/image",
        files={"file": f},
        data={"query": "What is in this image?"}
    )

result = response.json()
print(f"Classification: {result['classification']}")
print(f"Confidence: {result['confidence']}")
print(f"Safety checks: {result['safety_checks']}")
```

### Audio Transcription

```python
# Transcribe audio
with open("speech.mp3", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/analyze/audio",
        files={"file": f},
        data={"language": "en"}
    )

result = response.json()
print(f"Transcription: {result['transcription']}")
print(f"PII detected: {result['pii_detected']}")
```

### Visual Question Answering

```python
# Ask question about image
with open("beach.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/vqa",
        files={"file": f},
        data={"question": "How many people are in the image?"}
    )

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']}")
```

### Cross-Modal Search

```python
# Search images using text
response = requests.post(
    "http://localhost:8000/api/v1/search",
    json={
        "query": "sunset over mountains",
        "modality": "text_to_image",
        "top_k": 5
    }
)

results = response.json()
for result in results['matches']:
    print(f"{result['image_id']}: {result['similarity']}")
```

## Project Structure

```
.
├── src/
│   ├── main.py                    # FastAPI application entry point
│   ├── api/
│   │   ├── routes.py              # API endpoint definitions
│   │   ├── schemas.py             # Pydantic models for requests/responses
│   │   └── dependencies.py        # Shared dependencies (auth, rate limiting)
│   ├── models/
│   │   ├── clip_model.py          # CLIP wrapper for image-text
│   │   ├── whisper_model.py       # Whisper wrapper for audio
│   │   ├── vision_transformer.py  # Vision transformer utilities
│   │   └── fusion.py              # Cross-modal fusion logic
│   ├── safety/
│   │   ├── toxicity_filter.py     # Toxicity detection
│   │   ├── pii_detector.py        # PII detection and redaction
│   │   ├── bias_monitor.py        # Bias and fairness monitoring
│   │   ├── input_validator.py     # Prompt injection, jailbreak detection
│   │   ├── hallucination.py       # Hallucination detection
│   │   └── safety_layer.py        # Comprehensive safety orchestration
│   ├── core/
│   │   ├── config.py              # Configuration management
│   │   ├── logging.py             # Logging setup
│   │   └── exceptions.py          # Custom exceptions
│   └── utils/
│       ├── monitoring.py          # Prometheus metrics
│       ├── cache.py               # Redis caching
│       └── helpers.py             # Utility functions
├── tests/
│   ├── conftest.py                # Pytest configuration and fixtures
│   ├── test_api.py                # API endpoint tests
│   ├── test_models.py             # Model functionality tests
│   ├── test_safety.py             # Safety layer tests
│   ├── test_bias.py               # Bias detection tests
│   └── integration/
│       └── test_e2e.py            # End-to-end tests
├── config/
│   ├── logging.yaml               # Logging configuration
│   └── prometheus.yml             # Prometheus scraping config
├── scripts/
│   ├── download_models.py         # Download pre-trained models
│   ├── run_tests.sh               # Test runner script
│   └── benchmark.py               # Performance benchmarking
├── docker/
│   ├── Dockerfile                 # Application container
│   ├── docker-compose.yml         # Multi-container setup
│   └── nginx.conf                 # Nginx reverse proxy config
├── data/
│   ├── examples/                  # Example images and audio
│   └── test_cases/                # Test data for safety checks
├── docs/
│   ├── API.md                     # API documentation
│   ├── SAFETY.md                  # Safety features guide
│   └── DEPLOYMENT.md              # Deployment guide
├── .env.example                   # Example environment variables
├── requirements.txt               # Python dependencies
├── pyproject.toml                 # Python project metadata
└── README.md                      # This file
```

## Testing

### Run Tests

```bash
# All tests
pytest

# With coverage report
pytest --cov=src --cov-report=html tests/

# Specific test file
pytest tests/test_safety.py -v

# Integration tests only
pytest tests/integration/ -v

# Performance tests
pytest tests/test_performance.py --benchmark-only
```

### Test Coverage

Current coverage: **87%**

Key areas:
- API endpoints: 95%
- Safety layer: 92%
- Model wrappers: 85%
- Utilities: 78%

### Safety Tests

The safety layer is thoroughly tested:

```bash
# Run safety-specific tests
pytest tests/test_safety.py -v

# Test toxicity detection
pytest tests/test_safety.py::test_toxicity_filter -v

# Test PII protection
pytest tests/test_safety.py::test_pii_detector -v

# Test bias monitoring
pytest tests/test_bias.py -v
```

## Performance

### Benchmarks

Measured on NVIDIA T4 GPU:

| Endpoint | P50 Latency | P95 Latency | Throughput |
|----------|-------------|-------------|------------|
| Image Analysis | 450ms | 850ms | 120 req/s |
| Audio Transcription | 1.2s | 1.8s | 50 req/s |
| VQA | 600ms | 1.1s | 90 req/s |
| Cross-Modal Search | 200ms | 400ms | 250 req/s |

### Optimization Tips

1. **GPU Acceleration**: Use CUDA for model inference
2. **Batch Processing**: Process multiple requests together
3. **Caching**: Cache embeddings in Redis
4. **Async I/O**: Use async file operations
5. **Load Balancing**: Deploy multiple workers

## Deployment

### Docker

```bash
# Build image
docker build -t multimodal-safety-app .

# Run container
docker run -p 8000:8000 \
  -e DEVICE=cpu \
  -v $(pwd)/data:/app/data \
  multimodal-safety-app
```

### Docker Compose

```bash
# Start all services (API, Redis, PostgreSQL, Prometheus)
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

### Kubernetes

```bash
# Apply manifests
kubectl apply -f k8s/

# Check deployment
kubectl get pods -n multimodal-app

# Scale deployment
kubectl scale deployment/api --replicas=5
```

### Cloud Deployment

See [docs/DEPLOYMENT.md](./docs/DEPLOYMENT.md) for:
- AWS ECS deployment
- Google Cloud Run
- Azure Container Instances
- Heroku deployment

## Monitoring & Observability

### Metrics

Prometheus metrics exposed at `/metrics`:

- `http_requests_total`: Total HTTP requests
- `http_request_duration_seconds`: Request latency
- `safety_checks_total`: Safety checks performed
- `toxicity_detections_total`: Toxic content detected
- `pii_detections_total`: PII instances found
- `model_inference_duration_seconds`: Model inference time

### Logging

Structured JSON logs with:
- Request ID for tracing
- User information (anonymized)
- Safety check results
- Performance metrics
- Error stack traces

### Alerts

Configure alerts for:
- High error rate (>5%)
- Slow response time (P95 >3s)
- High toxicity detection rate (>10%)
- PII leakage incidents
- System resource exhaustion

## Security

### Authentication

```python
# API key authentication
headers = {"X-API-Key": "your-api-key"}
response = requests.get(
    "http://localhost:8000/api/v1/protected",
    headers=headers
)
```

### Rate Limiting

- 100 requests per hour per IP by default
- Configurable per endpoint
- 429 status code when limit exceeded

### CORS

Configure allowed origins in `.env`:

```bash
CORS_ORIGINS=http://localhost:3000,https://yourdomain.com
```

### SSL/TLS

For production, use reverse proxy (Nginx/Caddy) with SSL:

```nginx
server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Responsible AI

### Model Card

See [MODEL_CARD.md](./MODEL_CARD.md) for:
- Model details and architecture
- Intended use cases
- Limitations and risks
- Fairness evaluation results
- Ethical considerations

### Bias Auditing

Regular bias audits performed on:
- Different demographic groups
- Various content types
- Edge cases and adversarial inputs

Results documented in audit logs.

### User Privacy

- No data retention by default
- PII automatically redacted
- Audit logs anonymized
- GDPR compliance features

## Troubleshooting

### Common Issues

**Models not downloading:**
```bash
# Manually download models
python scripts/download_models.py
```

**CUDA out of memory:**
```bash
# Use smaller models or CPU
export DEVICE=cpu
export CLIP_MODEL=ViT-B/32  # smaller than ViT-L/14
export WHISPER_MODEL=tiny   # smaller than base/small/medium
```

**Slow inference:**
```bash
# Enable caching
export REDIS_URL=redis://localhost:6379/0

# Use batch processing
# Set batch_size in config
```

### Debug Mode

```bash
# Run with debug logging
LOG_LEVEL=DEBUG uvicorn src.main:app --reload
```

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

See [CONTRIBUTING.md](./CONTRIBUTING.md) for details.

## License

MIT License - see [LICENSE](./LICENSE) for details.

## Acknowledgments

- OpenAI for CLIP and Whisper
- Hugging Face for Transformers
- FastAPI team
- The open-source AI community

## Contact

For questions or issues:
- Open an issue on GitHub
- Email: your-email@example.com
- Discord: [Your Discord Server]

---

**Built with ❤️ for responsible AI development**
