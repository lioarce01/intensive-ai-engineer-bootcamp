#!/bin/bash

# Stop all services

echo "🛑 Stopping MLOps services..."

docker-compose down

echo "✅ All services stopped."
echo ""
echo "💾 To remove all data volumes, run: docker-compose down -v"
