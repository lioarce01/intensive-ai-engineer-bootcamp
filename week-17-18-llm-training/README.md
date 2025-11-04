# Week 17-18: Entrenamiento de LLMs desde Cero

> **Objetivo**: Entrenar un modelo de lenguaje desde cero, comprendiendo dataset curation, métricas de evaluación y técnicas de entrenamiento eficiente.

## 🎯 Objetivos de Aprendizaje

1. **Dataset Curation**: Aprender a recolectar, limpiar y preparar datos de calidad para entrenar LLMs
2. **Métricas de Evaluación**: Dominar perplexity, BLEU, ROUGE y métricas modernas
3. **Efficient Training**: Implementar técnicas de entrenamiento eficiente (gradient accumulation, mixed precision, etc.)
4. **Training from Scratch**: Entrenar un mini Language Model funcional

## 📚 Estructura del Módulo

```
week-17-18-llm-training/
├── dataset-curation/          # Scripts para curación de datasets
│   ├── data_collection.py     # Web scraping y recolección
│   ├── data_cleaning.py       # Limpieza y preprocesamiento
│   ├── tokenization.py        # Tokenización y preparación
│   └── quality_filters.py     # Filtros de calidad
├── training-metrics/          # Implementación de métricas
│   ├── perplexity.py         # Cálculo de perplexity
│   ├── bleu_rouge.py         # BLEU y ROUGE scores
│   ├── custom_metrics.py     # Métricas personalizadas
│   └── evaluation.py         # Pipeline de evaluación
├── efficient-training/        # Técnicas de entrenamiento eficiente
│   ├── gradient_accumulation.py
│   ├── mixed_precision.py
│   ├── gradient_checkpointing.py
│   └── distributed_training.py
├── mini-lm-project/          # Proyecto principal
│   ├── model.py              # Arquitectura del modelo
│   ├── train.py              # Script de entrenamiento
│   ├── config.py             # Configuración
│   └── inference.py          # Inferencia
└── notebooks/                # Notebooks educativos
    ├── 01_dataset_exploration.ipynb
    ├── 02_metrics_deep_dive.ipynb
    └── 03_training_mini_lm.ipynb
```

## 🚀 Proyecto Principal: Mini Language Model

### Especificaciones del Modelo
- **Arquitectura**: Transformer decoder-only (estilo GPT)
- **Tamaño**: ~50M parámetros
- **Dataset**: Subset de Wikipedia + código público
- **Training Time**: ~2-4 horas en GPU (T4/V100)
- **Target Perplexity**: <30 en validation set

### Componentes Clave

#### 1. Dataset Curation
```python
# Ejemplo de pipeline de curación
pipeline = DatasetPipeline([
    WebScraper(sources=['wikipedia', 'github']),
    QualityFilter(min_length=100, max_length=1024),
    DeduplicationFilter(),
    TokenCounter(target_tokens=100_000_000),
    Tokenizer(vocab_size=32_000)
])
```

#### 2. Métricas de Evaluación
- **Perplexity**: Medida de incertidumbre del modelo
- **BLEU/ROUGE**: Para generación de texto
- **Custom Metrics**: Coherencia, diversidad, toxicidad

#### 3. Entrenamiento Eficiente
- Gradient Accumulation (simular batch size grande)
- Mixed Precision Training (FP16)
- Gradient Checkpointing (ahorrar memoria)
- Distributed Data Parallel (múltiples GPUs)

## 📋 Tasks del Proyecto

### Task 1: Dataset Curation (Días 1-3)
- [ ] Recolectar 100M tokens de texto de calidad
- [ ] Implementar filtros de calidad
- [ ] Crear pipeline de preprocesamiento
- [ ] Validar distribución de datos

### Task 2: Metrics Implementation (Días 4-5)
- [ ] Implementar perplexity desde cero
- [ ] Integrar BLEU y ROUGE
- [ ] Crear dashboard de métricas
- [ ] Validar contra implementaciones estándar

### Task 3: Training Setup (Días 6-8)
- [ ] Definir arquitectura del modelo
- [ ] Implementar efficient training techniques
- [ ] Configurar experiment tracking (MLflow/W&B)
- [ ] Setup validation pipeline

### Task 4: Model Training (Días 9-12)
- [ ] Entrenar modelo base
- [ ] Monitorear métricas en tiempo real
- [ ] Ajustar hiperparámetros
- [ ] Validar convergencia

### Task 5: Evaluation & Deployment (Días 13-14)
- [ ] Evaluación rigurosa del modelo
- [ ] Comparar con baselines
- [ ] Documentar resultados
- [ ] Deploy API de inferencia

## 🛠️ Setup

```bash
# Instalar dependencias
pip install -r requirements.txt

# Descargar dataset de ejemplo
python dataset-curation/download_data.py

# Entrenar modelo
python mini-lm-project/train.py --config configs/mini_lm.yaml

# Evaluar modelo
python mini-lm-project/evaluate.py --checkpoint checkpoints/best_model.pt
```

## 📊 Métricas de Éxito

| Métrica | Target | Notas |
|---------|--------|-------|
| **Perplexity (val)** | <30 | Medida principal de calidad |
| **Training Time** | <4 hours | En GPU T4/V100 |
| **Model Size** | ~50M params | Balancear tamaño vs performance |
| **Throughput** | >100 tokens/s | Velocidad de generación |
| **BLEU Score** | >20 | En task de generación |

## 🔍 Conceptos Clave

### Perplexity
```python
# Perplexity = exp(cross_entropy_loss)
perplexity = torch.exp(loss)
```
- **Interpretación**: Cuántas opciones "promedio" el modelo considera por token
- **Bueno**: <20 (excelente), 20-50 (aceptable), >50 (necesita mejoras)

### Dataset Quality
- **Diversidad**: Múltiples dominios y estilos
- **Limpieza**: Sin duplicados, errores o contenido tóxico
- **Balance**: Representación equilibrada de temas
- **Tamaño**: Suficiente para generalización (100M+ tokens)

### Efficient Training
- **Gradient Accumulation**: Simular batch size de 1024 con GPU pequeña
- **Mixed Precision**: 2x speedup con AMP (Automatic Mixed Precision)
- **Gradient Checkpointing**: 40% menos memoria a cambio de 20% más tiempo

## 📚 Recursos

### Papers Fundamentales
- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - Arquitectura y training
- [Scaling Laws](https://arxiv.org/abs/2001.08361) - Relación entre tamaño y performance
- [Chinchilla Paper](https://arxiv.org/abs/2203.15556) - Optimal training compute

### Implementaciones de Referencia
- [nanoGPT](https://github.com/karpathy/nanoGPT) - Minimal GPT implementation
- [TinyLlama](https://github.com/jzhang38/TinyLlama) - 1.1B parameter model training
- [minGPT](https://github.com/karpathy/minGPT) - Educational GPT

### Herramientas
- **Datasets**: Hugging Face Datasets, Common Crawl
- **Training**: PyTorch, DeepSpeed, Accelerate
- **Monitoring**: Weights & Biases, TensorBoard, MLflow
- **Evaluation**: lm-evaluation-harness

## 🎓 Entregables

1. **Mini LM Funcional** (~50M params)
   - Código de training reproducible
   - Checkpoints del modelo
   - Métricas de evaluación

2. **Dataset Curado** (100M tokens)
   - Pipeline de curación documentado
   - Estadísticas y visualizaciones
   - Validación de calidad

3. **Reporte de Training**
   - Curvas de learning
   - Análisis de métricas
   - Comparación con baselines
   - Lecciones aprendidas

4. **API de Inferencia**
   - Endpoint FastAPI
   - Generación de texto
   - Métricas de latencia

## 💡 Tips Prácticos

1. **Start Small**: Prueba con 10M tokens antes de escalar
2. **Monitor Always**: Usa W&B o TensorBoard desde el día 1
3. **Validate Early**: Revisa outputs cada 1000 steps
4. **Save Frequently**: Checkpoints cada hora durante training
5. **Document Everything**: El training es caro, documenta los experimentos

## 🚧 Troubleshooting Común

### Perplexity no baja
- Verificar learning rate (típicamente 3e-4 para Adam)
- Revisar calidad de datos
- Aumentar model capacity
- Entrenar por más tiempo

### Out of Memory
- Reducir batch size
- Activar gradient checkpointing
- Usar gradient accumulation
- Reducir sequence length

### Training inestable
- Usar gradient clipping (max_norm=1.0)
- Reducir learning rate
- Aumentar warmup steps
- Revisar data preprocessing

## 🎯 Siguientes Pasos

Después de completar este módulo:
- **Week 19-20**: Multimodal AI y Responsible AI
- **Advanced**: Implementar RLHF para alineamiento
- **Production**: Deployment en Hugging Face Spaces
- **Research**: Contribuir a proyectos open source de LLMs

---

**Tiempo estimado**: 14 días
**Dificultad**: ⭐⭐⭐⭐⭐ (Avanzado)
**Prerequisitos**: Weeks 5-8 (Transformers y Fine-tuning)
