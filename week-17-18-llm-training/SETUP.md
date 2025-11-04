# Week 17-18 Setup Guide

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

## Quick Start

### 1. Dataset Curation

```bash
# Collect and process data
cd dataset-curation
python data_collection.py
python data_cleaning.py
python tokenization.py
```

### 2. Train Mini-LM

```bash
cd mini-lm-project

# Train with default config (50M model)
python train.py --config small_fast

# Train larger model
python train.py --config medium --model mini-100m
```

### 3. Generate Text

```bash
# Interactive generation
python inference.py \
  --checkpoint checkpoints/best_model.pt \
  --tokenizer tokenizers/custom_tokenizer.json

# Single prompt
python inference.py \
  --checkpoint checkpoints/best_model.pt \
  --tokenizer tokenizers/custom_tokenizer.json \
  --prompt "Once upon a time" \
  --max-tokens 100
```

## Directory Structure

```
week-17-18-llm-training/
├── dataset-curation/           # Data collection and preprocessing
│   ├── data_collection.py
│   ├── data_cleaning.py
│   ├── tokenization.py
│   └── quality_filters.py
├── training-metrics/           # Evaluation metrics
│   ├── perplexity.py
│   ├── bleu_rouge.py
│   ├── custom_metrics.py
│   └── evaluation.py
├── efficient-training/         # Training optimizations
│   ├── gradient_accumulation.py
│   ├── mixed_precision.py
│   ├── gradient_checkpointing.py
│   └── distributed_training.py
├── mini-lm-project/           # Main training project
│   ├── model.py
│   ├── config.py
│   ├── train.py
│   └── inference.py
└── notebooks/                 # Educational notebooks
    └── (coming soon)
```

## Training Configurations

### Small Fast (50M params)
- Training time: ~2 hours on T4
- Memory: ~8GB
- Target perplexity: <40

```bash
python train.py --config small_fast
```

### Medium (100M params)
- Training time: ~4 hours on T4
- Memory: ~12GB
- Target perplexity: <30

```bash
python train.py --config medium
```

### Large (250M params)
- Training time: ~8 hours on T4
- Memory: ~16GB (requires gradient checkpointing)
- Target perplexity: <25

```bash
python train.py --config large
```

## Common Issues

### Out of Memory
- Reduce batch size
- Enable gradient checkpointing
- Use mixed precision training
- Increase gradient accumulation steps

### Slow Training
- Enable mixed precision (FP16)
- Use larger batch size
- Check data loading (increase num_workers)

### High Perplexity
- Train longer
- Increase model size
- Improve data quality
- Adjust learning rate

## Next Steps

1. Complete the dataset curation pipeline
2. Train your first Mini-LM
3. Evaluate on various benchmarks
4. Deploy as an API (see Week 13-14)
5. Experiment with different architectures
