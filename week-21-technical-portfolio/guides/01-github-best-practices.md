# GitHub Best Practices for AI Engineers

> A comprehensive guide to building a professional GitHub presence that showcases your AI engineering skills.

## Table of Contents
- [Repository Setup](#repository-setup)
- [Code Organization](#code-organization)
- [Commit Best Practices](#commit-best-practices)
- [Documentation](#documentation)
- [CI/CD](#cicd)
- [Security](#security)

## Repository Setup

### Initial Configuration

```bash
# Initialize with proper .gitignore
git init
curl -o .gitignore https://raw.githubusercontent.com/github/gitignore/main/Python.gitignore

# Add custom ignores for AI/ML
cat >> .gitignore << EOL

# ML specific
*.pkl
*.h5
*.onnx
*.pt
*.pth
models/
data/
logs/
wandb/
.neptune/

# Environment
.env
.env.local
secrets/

# IDEs
.vscode/
.idea/
*.swp
EOL

# Create essential files
touch README.md LICENSE requirements.txt setup.py
```

### Repository Structure

```
project-name/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── workflows/
│       ├── test.yml
│       └── lint.yml
├── src/
│   └── project_name/
│       ├── __init__.py
│       ├── main.py
│       ├── models/
│       ├── utils/
│       └── config.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   └── test_*.py
├── docs/
│   ├── architecture.md
│   ├── api.md
│   └── deployment.md
├── examples/
│   └── basic_usage.py
├── scripts/
│   ├── train.py
│   └── evaluate.py
├── .gitignore
├── .pre-commit-config.yaml
├── Dockerfile
├── docker-compose.yml
├── LICENSE
├── README.md
├── requirements.txt
├── setup.py
└── pyproject.toml
```

## Code Organization

### Python Package Structure

**setup.py**:
```python
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="your-project-name",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A short description",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/your-project",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "ruff>=0.1",
            "mypy>=1.0",
        ],
    },
)
```

**pyproject.toml**:
```toml
[build-system]
requires = ["setuptools>=65.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "your-project-name"
version = "0.1.0"
description = "A short description"
readme = "README.md"
requires-python = ">=3.9"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
dependencies = [
    "torch>=2.0.0",
    "transformers>=4.30.0",
    "fastapi>=0.100.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "black>=23.0",
    "ruff>=0.1",
    "mypy>=1.0",
]

[tool.black]
line-length = 100
target-version = ['py39', 'py310', 'py311']
include = '\.pyi?$'

[tool.ruff]
line-length = 100
select = ["E", "F", "I", "N", "W"]
ignore = ["E501"]

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = "--cov=src --cov-report=html --cov-report=term-missing"
```

### Module Organization

**src/project_name/__init__.py**:
```python
"""
Your Project Name
================

A brief description of what your project does.

Example usage:
    >>> from project_name import Model
    >>> model = Model()
    >>> result = model.predict("input text")
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .models import Model
from .utils import preprocess, postprocess

__all__ = [
    "Model",
    "preprocess",
    "postprocess",
]
```

**Type Hints & Docstrings**:
```python
from typing import List, Dict, Optional, Union
import torch

def process_text(
    text: str,
    max_length: int = 512,
    truncation: bool = True,
    padding: str = "max_length",
) -> Dict[str, torch.Tensor]:
    """
    Process input text for model consumption.

    Args:
        text: Input text to process
        max_length: Maximum sequence length
        truncation: Whether to truncate sequences
        padding: Padding strategy ('max_length', 'longest', etc.)

    Returns:
        Dictionary containing tokenized inputs:
            - input_ids: Token IDs (torch.Tensor)
            - attention_mask: Attention mask (torch.Tensor)

    Raises:
        ValueError: If text is empty or max_length is invalid

    Example:
        >>> tokens = process_text("Hello world", max_length=128)
        >>> tokens['input_ids'].shape
        torch.Size([1, 128])
    """
    if not text:
        raise ValueError("Input text cannot be empty")

    # Implementation here
    pass
```

## Commit Best Practices

### Commit Message Format

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `perf`: Performance improvements

**Examples**:
```bash
# Feature
git commit -m "feat(model): add LoRA fine-tuning support"

# Bug fix
git commit -m "fix(api): handle timeout errors gracefully"

# Documentation
git commit -m "docs: add deployment guide for AWS"

# Refactoring with body
git commit -m "refactor(training): improve data loading pipeline

- Implement DataLoader with prefetching
- Add caching for preprocessed data
- Reduce memory footprint by 40%"

# Breaking change
git commit -m "feat(api)!: change response format to match OpenAI

BREAKING CHANGE: API responses now use 'choices' instead of 'results'"
```

### Git Workflow

```bash
# Create feature branch
git checkout -b feature/add-rag-system

# Make changes
git add src/rag/
git commit -m "feat(rag): implement hybrid search with reranking"

# Keep branch updated
git fetch origin
git rebase origin/main

# Push and create PR
git push -u origin feature/add-rag-system
```

### Pre-commit Hooks

**.pre-commit-config.yaml**:
```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=10000']
      - id: check-json
      - id: check-toml
      - id: detect-private-key

  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.0.280
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.4.1
    hooks:
      - id: mypy
        additional_dependencies: [types-requests]
```

Install:
```bash
pip install pre-commit
pre-commit install
```

## Documentation

### README.md Template

```markdown
# Project Name

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> One-line description of what your project does

[Demo](https://huggingface.co/spaces/user/project) | [Documentation](https://docs.example.com) | [Paper](https://arxiv.org/abs/...)

![Demo GIF](assets/demo.gif)

## ✨ Features

- 🚀 Fast inference (<100ms)
- 🎯 High accuracy (95%+)
- 🔧 Easy to customize
- 📦 Production-ready
- 🐳 Docker support

## 🚀 Quick Start

```bash
# Install
pip install your-project-name

# Run
from your_project import Model

model = Model()
result = model.predict("your input")
```

## 📦 Installation

### From PyPI
```bash
pip install your-project-name
```

### From Source
```bash
git clone https://github.com/user/project.git
cd project
pip install -e .
```

### Docker
```bash
docker pull user/project:latest
docker run -p 8000:8000 user/project:latest
```

## 💻 Usage

### Basic Example
```python
# Your example here
```

### Advanced Usage
```python
# Advanced example here
```

### API Reference
[Full API documentation](docs/api.md)

## 🏗️ Architecture

![Architecture Diagram](docs/architecture.png)

Brief explanation of system design...

## 📊 Performance

| Metric | Value |
|--------|-------|
| Latency (P50) | 45ms |
| Latency (P95) | 120ms |
| Accuracy | 95.3% |
| Throughput | 1000 req/s |

## 🛠️ Tech Stack

- **ML**: PyTorch, Transformers, PEFT
- **API**: FastAPI, Uvicorn
- **Data**: Pandas, NumPy
- **Deployment**: Docker, AWS Lambda
- **Monitoring**: Prometheus, Grafana

## 🗺️ Roadmap

- [x] Basic implementation
- [x] API endpoint
- [ ] Batch processing
- [ ] Multi-GPU support
- [ ] WebSocket streaming

## 🤝 Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

```bash
# Fork and clone
git clone https://github.com/your-username/project.git

# Create branch
git checkout -b feature/amazing-feature

# Make changes and test
pytest tests/

# Submit PR
```

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- Built with [Transformers](https://huggingface.co/transformers)
- Inspired by [similar-project](https://github.com/...)

## 📬 Contact

Your Name - [@twitter](https://twitter.com/handle) - email@example.com

Project Link: [https://github.com/user/project](https://github.com/user/project)
```

### CONTRIBUTING.md

```markdown
# Contributing to Project Name

Thank you for your interest in contributing!

## Development Setup

1. Fork and clone the repository
2. Install dependencies: `pip install -e ".[dev]"`
3. Install pre-commit hooks: `pre-commit install`
4. Create a branch: `git checkout -b feature/your-feature`

## Code Standards

- Follow PEP 8 style guide
- Use type hints
- Write docstrings for all public functions
- Add tests for new features
- Ensure all tests pass: `pytest`
- Format code with `black`
- Lint with `ruff`

## Pull Request Process

1. Update README.md with details of changes
2. Update documentation if needed
3. Add tests for new functionality
4. Ensure CI passes
5. Request review from maintainers

## Code Review Guidelines

- Be respectful and constructive
- Focus on code quality and design
- Suggest improvements, don't demand
- Approve when satisfied

## Questions?

Open an issue or reach out to maintainers.
```

## CI/CD

### GitHub Actions - Testing

**.github/workflows/test.yml**:
```yaml
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11']

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"

    - name: Lint with ruff
      run: |
        ruff check src/ tests/

    - name: Check formatting with black
      run: |
        black --check src/ tests/

    - name: Type check with mypy
      run: |
        mypy src/

    - name: Test with pytest
      run: |
        pytest tests/ --cov=src --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### GitHub Actions - Deployment

**.github/workflows/deploy.yml**:
```yaml
name: Deploy

on:
  push:
    tags:
      - 'v*'

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install build twine

    - name: Build package
      run: |
        python -m build

    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_TOKEN }}
      run: |
        twine upload dist/*

    - name: Build Docker image
      run: |
        docker build -t user/project:${{ github.ref_name }} .
        docker tag user/project:${{ github.ref_name }} user/project:latest

    - name: Push to Docker Hub
      env:
        DOCKER_USERNAME: ${{ secrets.DOCKER_USERNAME }}
        DOCKER_PASSWORD: ${{ secrets.DOCKER_PASSWORD }}
      run: |
        echo $DOCKER_PASSWORD | docker login -u $DOCKER_USERNAME --password-stdin
        docker push user/project:${{ github.ref_name }}
        docker push user/project:latest
```

## Security

### Environment Variables

Never commit secrets! Use `.env` files:

**.env.example**:
```bash
# API Keys
OPENAI_API_KEY=your_key_here
HUGGINGFACE_TOKEN=your_token_here

# Database
DATABASE_URL=postgresql://user:pass@localhost/db

# AWS
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret

# Application
DEBUG=False
LOG_LEVEL=INFO
```

**Loading in Python**:
```python
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")
```

### Security Scanning

**.github/workflows/security.yml**:
```yaml
name: Security Scan

on: [push, pull_request]

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Run Bandit
      run: |
        pip install bandit
        bandit -r src/

    - name: Run Safety
      run: |
        pip install safety
        safety check

    - name: Scan for secrets
      uses: trufflesecurity/trufflehog@main
      with:
        path: ./
```

## Branch Protection

Configure on GitHub:

1. **Settings → Branches → Add rule**
2. **Branch name pattern**: `main`
3. **Enable**:
   - Require pull request reviews (1-2 reviewers)
   - Require status checks to pass
   - Require conversation resolution
   - Include administrators

## GitHub Profile Optimization

### Profile README

Create `username/username/README.md`:

```markdown
# Hi there, I'm [Your Name] 👋

## 🚀 AI Engineer | LLM Specialist

I build production-ready AI systems with a focus on LLMs, RAG, and agent architectures.

### 🔭 Currently working on
- [Project 1](link) - Brief description
- [Project 2](link) - Brief description

### 🌱 Recently learned
- RLHF and alignment techniques
- Production-grade RAG systems
- Inference optimization with vLLM

### 💼 Tech Stack
![Python](https://img.shields.io/badge/-Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/-Transformers-yellow)
![FastAPI](https://img.shields.io/badge/-FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/-Docker-2496ED?style=flat&logo=docker&logoColor=white)

### 📊 GitHub Stats
![Your GitHub Stats](https://github-readme-stats.vercel.app/api?username=yourusername&show_icons=true&theme=radical)

### 📫 How to reach me
- Twitter: [@yourhandle](https://twitter.com/yourhandle)
- LinkedIn: [Your Name](https://linkedin.com/in/yourname)
- Email: your.email@example.com
```

---

**Next**: [Creating Effective Demos](02-creating-demos.md)
