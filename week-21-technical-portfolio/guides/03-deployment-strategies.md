# Deployment Strategies for AI Applications

> Comprehensive guide to deploying AI/ML applications to production with best practices for scalability, reliability, and cost-effectiveness.

## Table of Contents
- [Deployment Platforms](#deployment-platforms)
- [Containerization](#containerization)
- [CI/CD Pipelines](#cicd-pipelines)
- [Monitoring & Logging](#monitoring--logging)
- [Scaling Strategies](#scaling-strategies)
- [Cost Optimization](#cost-optimization)

## Deployment Platforms

### Platform Comparison Matrix

| Platform | Best For | Pros | Cons | Free Tier |
|----------|----------|------|------|-----------|
| **HuggingFace Spaces** | ML demos, model hosting | Free GPU, ML-focused, easy | Limited customization | Yes (CPU/GPU) |
| **Railway** | Full-stack apps, APIs | Simple, generous free tier | US-only datacenters | $5/month credit |
| **Render** | Web services, Docker | Auto-deploy, free SSL | Cold starts on free tier | Yes |
| **Fly.io** | Global edge deployment | Multi-region, fast | More complex setup | Yes (limited) |
| **AWS Lambda** | Serverless functions | Pay-per-use, scalable | Cold starts, complexity | Yes (generous) |
| **Google Cloud Run** | Containerized apps | Auto-scaling, pay-per-use | GCP learning curve | Yes |
| **Vercel/Netlify** | Static sites, APIs | Instant deploys, CDN | Not for long-running | Yes |
| **Modal** | GPU workloads | Easy GPU access, serverless | Newer platform | Yes (limited) |

## Containerization

### Docker Basics

**Dockerfile for FastAPI + PyTorch**:
```dockerfile
# Use official Python runtime
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Multi-stage Build (Optimized)**:
```dockerfile
# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Copy wheels from builder
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .

# Install from wheels (faster)
RUN pip install --no-cache /wheels/*

# Copy application
COPY . .

# Non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**GPU Support**:
```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch with CUDA
RUN pip install --no-cache-dir \
    torch==2.0.0+cu118 \
    -f https://download.pytorch.org/whl/torch_stable.html

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python3", "app.py"]
```

### Docker Compose

**docker-compose.yml**:
```yaml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/mydb
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    volumes:
      - ./app:/app/app  # Mount for development
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=mydb
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - api
    restart: unless-stopped

volumes:
  postgres_data:
```

**Production docker-compose**:
```yaml
version: '3.8'

services:
  api:
    image: your-registry/your-app:latest
    deploy:
      replicas: 3
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
      resources:
        limits:
          cpus: '1'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 1G
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  # ... other services
```

## CI/CD Pipelines

### GitHub Actions - Complete Pipeline

**.github/workflows/deploy.yml**:
```yaml
name: Deploy to Production

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov ruff black mypy

      - name: Lint with ruff
        run: ruff check src/

      - name: Format check with black
        run: black --check src/

      - name: Type check with mypy
        run: mypy src/

      - name: Run tests
        run: pytest tests/ --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'

    permissions:
      contents: read
      packages: write

    steps:
      - uses: actions/checkout@v3

      - name: Log in to Container Registry
        uses: docker/login-action@v2
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v4
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=semver,pattern={{version}}
            type=sha

      - name: Build and push Docker image
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'

    steps:
      - name: Deploy to production
        uses: appleboy/ssh-action@master
        with:
          host: ${{ secrets.PROD_HOST }}
          username: ${{ secrets.PROD_USER }}
          key: ${{ secrets.PROD_SSH_KEY }}
          script: |
            cd /app
            docker pull ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:main
            docker-compose down
            docker-compose up -d
            docker system prune -f

      - name: Health check
        run: |
          sleep 10
          curl -f https://your-app.com/health || exit 1

      - name: Notify on success
        if: success()
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          text: 'Deployment successful! 🚀'
          webhook_url: ${{ secrets.SLACK_WEBHOOK }}

      - name: Notify on failure
        if: failure()
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          text: 'Deployment failed! ❌'
          webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

### Platform-Specific Deployments

#### Railway

**railway.json**:
```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE",
    "dockerfilePath": "Dockerfile"
  },
  "deploy": {
    "startCommand": "uvicorn app.main:app --host 0.0.0.0 --port $PORT",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

Deploy:
```bash
# Install Railway CLI
npm i -g @railway/cli

# Login
railway login

# Link to project
railway link

# Deploy
railway up
```

#### Render

**render.yaml**:
```yaml
services:
  - type: web
    name: my-ai-api
    env: docker
    plan: starter
    healthCheckPath: /health
    envVars:
      - key: PYTHON_VERSION
        value: 3.11
      - key: DATABASE_URL
        fromDatabase:
          name: my-db
          property: connectionString
    autoDeploy: true

databases:
  - name: my-db
    plan: starter
    databaseName: mydb
    user: myuser

  - name: my-redis
    plan: starter
    type: redis
```

#### Fly.io

**fly.toml**:
```toml
app = "my-ai-app"
primary_region = "iad"

[build]
  dockerfile = "Dockerfile"

[env]
  PORT = "8000"

[http_service]
  internal_port = 8000
  force_https = true
  auto_stop_machines = true
  auto_start_machines = true
  min_machines_running = 0

[[vm]]
  cpu_kind = "shared"
  cpus = 1
  memory_mb = 1024

[[services]]
  protocol = "tcp"
  internal_port = 8000

  [[services.ports]]
    port = 80
    handlers = ["http"]

  [[services.ports]]
    port = 443
    handlers = ["tls", "http"]

  [services.concurrency]
    type = "connections"
    hard_limit = 25
    soft_limit = 20

  [[services.tcp_checks]]
    interval = "15s"
    timeout = "2s"
    grace_period = "5s"
```

Deploy:
```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Login
flyctl auth login

# Launch app
flyctl launch

# Deploy
flyctl deploy

# Scale
flyctl scale count 3
flyctl scale vm shared-cpu-2x --memory 2048
```

## Monitoring & Logging

### Application Monitoring

**Prometheus + Grafana**:

```python
# app/monitoring.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Define metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint']
)

MODEL_INFERENCE_TIME = Histogram(
    'model_inference_duration_seconds',
    'Model inference time',
    ['model_name']
)

ACTIVE_CONNECTIONS = Gauge(
    'active_connections',
    'Number of active connections'
)

def track_metrics(func):
    """Decorator to track request metrics"""
    async def wrapper(*args, **kwargs):
        start_time = time.time()

        try:
            result = await func(*args, **kwargs)
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status=200
            ).inc()
            return result

        except Exception as e:
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status=500
            ).inc()
            raise

        finally:
            duration = time.time() - start_time
            REQUEST_LATENCY.labels(
                method=request.method,
                endpoint=request.url.path
            ).observe(duration)

    return wrapper
```

**FastAPI Integration**:
```python
from fastapi import FastAPI
from prometheus_client import make_asgi_app

app = FastAPI()

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

@app.middleware("http")
async def add_metrics(request: Request, call_next):
    ACTIVE_CONNECTIONS.inc()
    try:
        response = await call_next(request)
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        return response
    finally:
        ACTIVE_CONNECTIONS.dec()
```

### Structured Logging

```python
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
        }

        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_data)

# Configure logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Usage
logger.info("Processing request", extra={
    'user_id': user_id,
    'model': 'gpt-3.5',
    'tokens': 150
})
```

### Error Tracking with Sentry

```python
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

sentry_sdk.init(
    dsn="your-sentry-dsn",
    integrations=[FastApiIntegration()],
    traces_sample_rate=1.0,
    environment="production"
)

# Automatic error tracking
@app.get("/predict")
async def predict(text: str):
    try:
        result = model.predict(text)
        return {"result": result}
    except Exception as e:
        # Automatically captured by Sentry
        raise
```

## Scaling Strategies

### Horizontal Scaling

**Load Balancer (Nginx)**:

```nginx
upstream app_servers {
    least_conn;  # Load balancing method
    server app1:8000 weight=1;
    server app2:8000 weight=1;
    server app3:8000 weight=2;  # More powerful server
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://app_servers;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;

        # Buffering
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
    }

    # Health check endpoint
    location /health {
        access_log off;
        proxy_pass http://app_servers/health;
    }
}
```

### Vertical Scaling

**Resource Limits**:

```yaml
# docker-compose.yml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

### Auto-scaling

**Kubernetes HPA** (Horizontal Pod Autoscaler):

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ai-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ai-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Caching Strategy

**Redis Caching**:

```python
import redis
import json
import hashlib
from functools import wraps

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_result(ttl=3600):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key from function name and args
            cache_key = f"{func.__name__}:{hashlib.md5(str(args).encode()).hexdigest()}"

            # Try to get from cache
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)

            # Compute result
            result = await func(*args, **kwargs)

            # Store in cache
            redis_client.setex(
                cache_key,
                ttl,
                json.dumps(result)
            )

            return result
        return wrapper
    return decorator

@cache_result(ttl=3600)
async def expensive_model_inference(text: str):
    return model.predict(text)
```

## Cost Optimization

### Cold Start Optimization

**Keep-Alive Ping**:

```python
# scheduled_ping.py
import requests
import schedule
import time

def ping_app():
    try:
        requests.get("https://your-app.com/health", timeout=5)
        print("Ping successful")
    except Exception as e:
        print(f"Ping failed: {e}")

# Ping every 5 minutes
schedule.every(5).minutes.do(ping_app)

while True:
    schedule.run_pending()
    time.sleep(60)
```

### Model Optimization

**Quantization**:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained("model-name")

# Dynamic quantization (CPU)
model_quantized = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Static quantization (more involved but better)
# See: https://pytorch.org/docs/stable/quantization.html

# Save quantized model
torch.save(model_quantized.state_dict(), "model_quantized.pth")
```

**Model Pruning**:

```python
import torch.nn.utils.prune as prune

# Prune 30% of connections
prune.l1_unstructured(model.layer, name="weight", amount=0.3)

# Make pruning permanent
prune.remove(model.layer, 'weight')
```

### Batch Processing

```python
from fastapi import BackgroundTasks
import asyncio

batch_queue = []
batch_size = 32

async def process_batch():
    global batch_queue
    while True:
        if len(batch_queue) >= batch_size:
            # Process batch
            batch = batch_queue[:batch_size]
            results = model.predict_batch(batch)

            # Store results
            for item, result in zip(batch, results):
                item['future'].set_result(result)

            batch_queue = batch_queue[batch_size:]

        await asyncio.sleep(0.1)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(process_batch())

@app.post("/predict")
async def predict(text: str):
    future = asyncio.Future()
    batch_queue.append({'text': text, 'future': future})
    result = await future
    return {"result": result}
```

## Deployment Checklist

### Pre-Deployment

- [ ] All tests passing (unit, integration, e2e)
- [ ] Security scan completed (no vulnerabilities)
- [ ] Performance benchmarks met
- [ ] Load testing completed
- [ ] Documentation updated
- [ ] Environment variables configured
- [ ] Database migrations ready
- [ ] Backup strategy in place
- [ ] Rollback plan documented

### Deployment

- [ ] Build Docker image
- [ ] Push to registry
- [ ] Deploy to staging first
- [ ] Run smoke tests
- [ ] Monitor logs for errors
- [ ] Check health endpoints
- [ ] Verify metrics
- [ ] Test key user flows

### Post-Deployment

- [ ] Monitor error rates
- [ ] Check performance metrics
- [ ] Review logs
- [ ] Verify all features working
- [ ] Monitor resource usage
- [ ] Update status page
- [ ] Notify stakeholders
- [ ] Document any issues

## Troubleshooting Common Issues

### Issue: High latency

**Solutions**:
- Add caching layer (Redis)
- Optimize model (quantization, pruning)
- Use batch processing
- Add load balancer
- Scale horizontally

### Issue: Out of memory

**Solutions**:
- Reduce batch size
- Clear GPU cache after inference
- Use gradient checkpointing
- Switch to smaller model
- Add more memory

### Issue: Cold starts

**Solutions**:
- Keep-alive pings
- Pre-warm instances
- Use serverless with provisioned concurrency
- Cache model in-memory
- Reduce image size

### Issue: Failed deployments

**Solutions**:
- Check logs
- Verify environment variables
- Test locally with production settings
- Verify Docker image builds
- Check resource limits

---

**Next**: [Writing Technical Documentation](04-technical-documentation.md)
