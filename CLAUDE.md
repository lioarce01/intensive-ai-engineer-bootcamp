# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI Engineer bootcamp repository focused on a 6-month intensive program covering LLMs and AI agents. The repository contains implementation of foundational NLP concepts, transformer architectures, and fine-tuning techniques with both code modules and interactive notebooks.

## Repository Structure

The repository currently contains:
- `README.md` - Comprehensive 24-week bootcamp curriculum covering AI engineering topics
- `.claude/agents/` - Contains specialized agent configuration files for different AI/ML domains:
  - `ai-agent-builder.md` - Agent building expertise
  - `fine-tuning-specialist.md` - Model fine-tuning guidance
  - `mlops-engineer.md` - MLOps and deployment expertise
  - `nlp-transformer-expert.md` - Transformer architecture expertise
  - `python-pro.md` - Advanced Python development
  - `rag-architect.md` - Retrieval-Augmented Generation expertise

### Week Projects

#### Week 3-4: NLP Basics (`week-03-04-nlp-basics/`)
- **01-tokenization/**: Tokenization fundamentals with Python implementations and notebooks
- **02-word-embeddings/**: Word embeddings tutorial and comprehensive implementations
- **03-semantic-search/**: Full-featured semantic search engine with:
  - Core modules: `embedding_engine.py`, `search_engine.py`, `document_processor.py`
  - FAISS-based indexing for vector similarity search
  - FastAPI web application with Docker support
  - Comprehensive documentation (README.md, PROJECT_STRUCTURE.md, FAISS_IMPLEMENTATION.md)
  - Test suite and demo applications

#### Week 5-6: Transformer Architecture (`week-05-06-transformer-architecture/`)
Complete transformer encoder built from scratch:
- `attention_mechanism.py` - Multi-head attention implementation
- `positional_encoding.py` - Position embeddings for sequence data
- `feedforward.py` - Feedforward neural network layers
- `normalization_residuals.py` - Layer normalization and residual connections
- `transformer_encoder.py` - Complete encoder architecture
- `example_usage.py` - Practical examples and demonstrations

#### Week 7-8: Fine-tuning (`week-07-08-fine-tuning/`)
Parameter-efficient fine-tuning with LoRA:
- `lora_implementation.py` - Low-Rank Adaptation module
- `01_lora_concepts_theory.ipynb` - Theoretical foundations
- `02_lora_implementation.ipynb` - Step-by-step implementation
- `03_lora_with_real_models.ipynb` - Integration with pre-trained models
- `04_email_classification_dataset.ipynb` - Practical dataset preparation
- `05_complete_training_pipeline.ipynb` - End-to-end training workflow
- `06_final_comparison_and_summary.ipynb` - Results analysis and comparison

## Development Context

This repository serves as a learning and project workspace for an AI engineering bootcamp. The curriculum covers:

**Weeks 1-8**: Foundational topics (math, NLP, transformers, fine-tuning)
**Weeks 9-14**: Advanced applications (agents, RAG systems)  
**Weeks 15-20**: Infrastructure and deployment (training, inference, MLOps)
**Weeks 21-24**: Portfolio development and specialization

## Key Technologies In Use

Currently implemented technologies:
- **ML/AI**: PyTorch, Hugging Face Transformers, sentence-transformers
- **Vector DBs**: FAISS (Facebook AI Similarity Search)
- **Web Framework**: FastAPI (semantic search web application)
- **Deployment**: Docker, docker-compose
- **Data Processing**: NumPy, pandas (for dataset handling)
- **Fine-tuning**: LoRA (Low-Rank Adaptation) for parameter-efficient training

Future technologies (based on curriculum):
- LangChain (AI agents)
- Chroma (additional vector DB option)
- vLLM (inference optimization)
- MLflow, Gradio, Hugging Face Spaces (MLOps)

## Project Development Notes

- Repository structure follows week-by-week curriculum progression
- Each week folder contains both production code (.py modules) and educational notebooks (.ipynb)
- Semantic search project includes comprehensive documentation and deployment configurations
- Requirements files are project-specific (e.g., `requirements_semantic_search.txt`, `requirements_web.txt`)
- Transformer implementation is modular and built from scratch for educational purposes

## Running Projects

### Semantic Search Application
```bash
cd week-03-04-nlp-basics/03-semantic-search/webapp
pip install -r requirements_web.txt
python start_web_interface.py
```
Or using Docker:
```bash
docker-compose up
```

### Transformer Encoder Example
```bash
cd week-05-06-transformer-architecture
python example_usage.py
```

### Fine-tuning Notebooks
Navigate to `week-07-08-fine-tuning/` and run notebooks sequentially (01 through 06)

## Current State

The repository has completed the foundational phase (Weeks 3-8) with working implementations:
- NLP fundamentals with production-ready semantic search engine
- Complete transformer encoder architecture from scratch
- LoRA-based fine-tuning pipeline with comprehensive tutorials

Next phases will focus on:
- Weeks 9-14: AI agents and RAG systems (LangChain integration)
- Weeks 15-20: MLOps, training infrastructure, and inference optimization
- Weeks 21-24: Portfolio projects and specialization