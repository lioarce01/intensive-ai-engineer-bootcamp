.PHONY: help build up down logs test clean install lint format

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make build      - Build Docker images"
	@echo "  make up         - Start all services"
	@echo "  make down       - Stop all services"
	@echo "  make logs       - View logs"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run linting"
	@echo "  make format     - Format code"
	@echo "  make clean      - Clean up volumes and containers"

install:
	pip install -r requirements.txt
	pip install pytest pytest-cov pytest-asyncio flake8 black isort

build:
	docker-compose build

up:
	@chmod +x scripts/start.sh
	@./scripts/start.sh

down:
	docker-compose down

logs:
	docker-compose logs -f

test:
	pytest tests/ -v --cov=api --cov-report=html

lint:
	flake8 api/ --count --statistics
	black --check api/
	isort --check-only api/

format:
	black api/
	isort api/

clean:
	docker-compose down -v
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
