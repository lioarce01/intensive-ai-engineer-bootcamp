#!/bin/bash

# Start all services using docker-compose

echo "🚀 Starting MLOps services..."

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed. Please install it first."
    exit 1
fi

# Build and start services
docker-compose up -d --build

echo "⏳ Waiting for services to be ready..."
sleep 10

# Check if services are running
echo ""
echo "📊 Service Status:"
docker-compose ps

echo ""
echo "✅ Services started successfully!"
echo ""
echo "🌐 Available endpoints:"
echo "  API:        http://localhost:8000"
echo "  API Docs:   http://localhost:8000/docs"
echo "  MLflow:     http://localhost:5000"
echo "  Prometheus: http://localhost:9090"
echo "  Grafana:    http://localhost:3000 (admin/admin)"
echo "  Jaeger:     http://localhost:16686"
echo ""
echo "📝 To view logs: docker-compose logs -f"
echo "🛑 To stop:      docker-compose down"
