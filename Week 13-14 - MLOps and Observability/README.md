# Week 13-14: MLOps & Observability

A production-grade MLOps pipeline with comprehensive observability, monitoring, and experiment tracking.

## Overview

This project demonstrates a complete MLOps stack featuring:

- **FastAPI REST API** for ML model serving
- **MLflow** for experiment tracking and model registry
- **Docker** for containerization and deployment
- **Prometheus** for metrics collection
- **Grafana** for visualization
- **Jaeger** for distributed tracing
- **OpenTelemetry** for observability instrumentation

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client Requests                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   FastAPI API   │
                    │   (Port 8000)   │
                    └────────┬────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
     ┌────▼────┐      ┌──────▼──────┐    ┌─────▼─────┐
     │ MLflow  │      │ Prometheus  │    │  Jaeger   │
     │  5000   │      │    9090     │    │   16686   │
     └─────────┘      └──────┬──────┘    └───────────┘
                             │
                      ┌──────▼──────┐
                      │   Grafana   │
                      │    3000     │
                      └─────────────┘
```

## Features

### 1. ML API Endpoints

- `POST /train` - Train ML models with automatic experiment tracking
- `POST /predict` - Make predictions with deployed models
- `GET /health` - Health check endpoint
- `GET /model/info` - Get current model information
- `GET /metrics/summary` - Application metrics summary
- `POST /model/load/{run_id}` - Load specific model version

### 2. Observability Stack

#### Metrics (Prometheus + Grafana)
- Request latency and throughput
- Model inference time
- Training duration
- Error rates
- Resource utilization

#### Tracing (Jaeger + OpenTelemetry)
- Distributed request tracing
- Service dependency mapping
- Performance bottleneck identification
- End-to-end request flow visualization

#### Logging (Structured JSON)
- Structured logging with JSON format
- Request/response logging
- Error tracking
- Audit trails

### 3. Experiment Tracking (MLflow)

- Model versioning
- Parameter and metric logging
- Artifact storage
- Model registry
- Experiment comparison

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.10+
- Make (optional)

### Installation

1. **Clone and navigate to the project:**
```bash
cd "Week 13-14 - MLOps and Observability"
```

2. **Start all services:**
```bash
make up
# or
docker-compose up -d --build
```

3. **Wait for services to be ready (30 seconds):**
```bash
# Check service status
docker-compose ps
```

### Access the Services

- **API Documentation**: http://localhost:8000/docs
- **MLflow UI**: http://localhost:5000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **Jaeger UI**: http://localhost:16686

## Usage Examples

### 1. Train a Model

```python
import requests

# Training data
payload = {
    "data": [
        [5.1, 3.5, 1.4, 0.2],
        [4.9, 3.0, 1.4, 0.2],
        [6.2, 2.9, 4.3, 1.3],
        [5.9, 3.0, 4.2, 1.5]
    ],
    "target": [0, 0, 1, 1],
    "algorithm": "random_forest",
    "hyperparameters": {
        "n_estimators": 100,
        "max_depth": 10
    }
}

response = requests.post("http://localhost:8000/train", json=payload)
result = response.json()
print(f"Run ID: {result['run_id']}")
print(f"Accuracy: {result['metrics']['accuracy']:.4f}")
```

### 2. Make Predictions

```python
# Prediction request
payload = {
    "features": [
        [5.1, 3.5, 1.4, 0.2],
        [6.2, 2.9, 4.3, 1.3]
    ]
}

response = requests.post("http://localhost:8000/predict", json=payload)
result = response.json()
print(f"Predictions: {result['predictions']}")
```

### 3. Run Example Script

```bash
python examples/train_example.py
```

## Development

### Local Development

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run tests:**
```bash
make test
# or
pytest tests/ -v --cov=api
```

3. **Lint code:**
```bash
make lint
```

4. **Format code:**
```bash
make format
```

### Project Structure

```
.
├── api/                      # API application
│   ├── main.py              # FastAPI application
│   ├── models.py            # Pydantic models
│   ├── ml_service.py        # ML service with MLflow
│   └── observability.py     # Logging and metrics
├── monitoring/              # Monitoring configuration
│   ├── prometheus.yml       # Prometheus config
│   └── grafana/            # Grafana dashboards
├── examples/               # Example scripts
│   └── train_example.py    # Training example
├── tests/                  # Test suite
│   └── test_api.py        # API tests
├── scripts/               # Utility scripts
│   ├── start.sh          # Start services
│   └── stop.sh           # Stop services
├── docker-compose.yml     # Service orchestration
├── Dockerfile            # API container image
├── requirements.txt      # Python dependencies
└── Makefile             # Common commands
```

## API Documentation

### Train Model
```http
POST /train
Content-Type: application/json

{
  "data": [[...], [...]],
  "target": [0, 1],
  "algorithm": "random_forest",
  "hyperparameters": {
    "n_estimators": 100,
    "max_depth": 10
  }
}
```

**Supported Algorithms:**
- `random_forest` - Random Forest Classifier
- `gradient_boosting` - Gradient Boosting Classifier
- `logistic_regression` - Logistic Regression

### Predict
```http
POST /predict
Content-Type: application/json

{
  "features": [[5.1, 3.5, 1.4, 0.2]]
}
```

### Get Metrics
```http
GET /metrics/summary
```

Returns application performance metrics including prediction count, training count, and average times.

## Monitoring & Observability

### Prometheus Metrics

Access Prometheus at http://localhost:9090

Key metrics:
- `http_request_duration_seconds` - Request latency
- `http_requests_total` - Total requests
- `prediction_time_seconds` - Inference time
- `training_time_seconds` - Training duration

### Grafana Dashboards

Access Grafana at http://localhost:3000 (admin/admin)

Pre-configured data source:
- Prometheus (default)

Create dashboards to visualize:
- API performance
- Model metrics
- System resources
- Error rates

### Jaeger Tracing

Access Jaeger at http://localhost:16686

View distributed traces to understand:
- Request flow through services
- Performance bottlenecks
- Service dependencies
- Error propagation

### MLflow Tracking

Access MLflow at http://localhost:5000

Features:
- Experiment comparison
- Model versioning
- Parameter tracking
- Metric visualization
- Artifact management

## CI/CD Pipeline

GitHub Actions workflow included for:

1. **Testing** - Run unit tests on multiple Python versions
2. **Linting** - Code quality checks
3. **Building** - Docker image build
4. **Integration Tests** - End-to-end testing

Workflow file: `.github/workflows/ci.yml`

## Troubleshooting

### Services not starting

```bash
# Check logs
docker-compose logs -f

# Restart services
docker-compose restart
```

### Port conflicts

If ports are already in use, modify `docker-compose.yml`:
```yaml
ports:
  - "8001:8000"  # Change host port
```

### MLflow connection issues

Ensure the MLflow service is running:
```bash
docker-compose ps mlflow
curl http://localhost:5000/health
```

## Performance Optimization

### Model Inference

- Use batch predictions for multiple samples
- Cache frequently used models
- Consider model quantization for faster inference

### API Performance

- Increase Uvicorn workers: `--workers 4`
- Use async endpoints for I/O operations
- Enable response compression

### Monitoring

- Adjust Prometheus scrape intervals
- Configure metric retention policies
- Set up alerting rules

## Best Practices

1. **Experiment Tracking**
   - Always log experiments to MLflow
   - Track all hyperparameters
   - Version your datasets

2. **Model Versioning**
   - Tag models with semantic versions
   - Document model changes
   - Test before production deployment

3. **Monitoring**
   - Set up alerts for anomalies
   - Monitor model drift
   - Track prediction latency

4. **Security**
   - Use environment variables for secrets
   - Enable authentication in production
   - Secure service-to-service communication

## Advanced Topics

### Custom Algorithms

Add new algorithms in `api/ml_service.py`:

```python
from sklearn.svm import SVC

def _get_algorithm(self, algorithm: str, hyperparameters: Optional[Dict] = None):
    algorithms = {
        "random_forest": RandomForestClassifier,
        "svm": SVC,  # Add new algorithm
        # ...
    }
```

### Custom Metrics

Add metrics in `api/observability.py`:

```python
def record_custom_metric(self, metric_name: str, value: float):
    """Record custom application metric."""
    if metric_name not in self.metrics:
        self.metrics[metric_name] = []
    self.metrics[metric_name].append(value)
```

### Model Serving at Scale

For production scale:
1. Use Kubernetes for orchestration
2. Implement horizontal pod autoscaling
3. Use Redis for model caching
4. Add load balancing
5. Implement circuit breakers

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [OpenTelemetry Python](https://opentelemetry.io/docs/instrumentation/python/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `make test`
5. Format code: `make format`
6. Submit a pull request

## License

MIT License

## Support

For issues and questions:
- Check the [API Documentation](http://localhost:8000/docs)
- Review service logs: `docker-compose logs -f`
- Open an issue in the repository

---

**Happy MLOps!** 🚀🤖📊
