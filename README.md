# Itinerario Intensivo de 6 Meses para Convertirse en AI Engineer (Enfoque Pareto 20/80)

**Objetivo:** Convertirte en AI Engineer especializado en LLMs, agentes inteligentes y sistemas de entrenamiento y despliegue, enfocando en el 20% del conocimiento que habilita el 80% de impacto práctico.

**Duración:** 6 meses — 8 horas semanales (fines de semana)

**Formato:** Estudio teórico enfocado + ejercicios prácticos desde el inicio, con proyectos de aplicación real y portfolio público.

---

## Mes 1: Fundamentos clave para entender Deep Learning y NLP

### Semana 1-2: Álgebra, cálculo y probabilidad esenciales

* **Temas:** Derivadas parciales, gradiente, vectores/matrices, entropía, probabilidad condicional
* **Práctica:** Softmax, cross-entropy, backpropagation manual
* **Recursos:**

  * [Essence of Linear Algebra (3Blue1Brown)](https://www.youtube.com/watch?v=fNk_zzaMoSs)
  * [CS229 Notes](https://cs229.stanford.edu/notes2022fall/)

### Semana 3-4: NLP práctico y embeddings

* **Temas:** Tokenización, word2vec, GloVe, TF-IDF, visualización de espacios semánticos
* **Práctica:** Entrenar word2vec, similitud semántica
* **Proyecto:** Buscador semántico básico
* **Recursos:**

  * [The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/)

---

## Mes 2: Transformers y Fundamentos de LLMs

### Semana 5-6: Arquitectura Transformer explicada

* **Temas:** Multi-head attention, encoder/decoder, positional encoding
* **Práctica:** Implementación mínima de Transformer encoder
* **Recursos:**

  * [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
  * [Karpathy GPT](https://github.com/karpathy/ng-video-lecture)

### Semana 7-8: Fine-tuning práctico con LoRA/QLoRA

* **Temas:** Pretraining, LoRA, QLoRA, PEFT con HF
* **Práctica:** Fine-tune Mistral o LLaMA en una tarea de negocio
* **Proyecto:** Clasificador de tickets o correos reales
* **Recursos:**

  * [Hugging Face PEFT](https://github.com/huggingface/peft)

---

## Mes 3: RAG, agentes y herramientas inteligentes

### Semana 9-10: Agentes + Herramientas externas

* **Temas:** LangChain, Tools, memoria, planificadores
* **Práctica:** Agente que consulta APIs + funciones custom
* **Proyecto:** Agente que responde preguntas de documentación técnica

### Semana 11-12: Retrieval-Augmented Generation (RAG)

* **Temas:** Chunking, embeddings, FAISS/Chroma, RAG pipelines
* **Práctica:** RAG local + indexado desde PDFs
* **Proyecto:** Asistente sobre documentación empresarial

---

## Mes 4: Entrenamiento desde cero y eficiencia

### Semana 13-14: Códigos base de entrenamiento + optimización

* **Temas:** Backprop, inicialización, regularización, SGD, Adam
* **Práctica:** Entrenar desde cero sobre texto/tablas
* **Proyecto:** Entrenar un clasificador con PyTorch puro

### Semana 15-16: Entrenamiento de LLMs + evaluación

* **Temas:** Dataset curation, tokenización, métricas (perplexity, BLEU)
* **Práctica:** Entrenar transformer pequeño con corpus reducido
* **Proyecto:** Mini GPT entrenado sobre tus textos

---

## Mes 5: Inferencia optimizada y MLOps mínimo viable

### Semana 17-18: Infraestructura de inferencia eficiente

* **Temas:** vLLM, Text Generation Inference, quantization, distillation
* **Práctica:** Servir modelo como API REST eficiente
* **Proyecto:** Microservicio de generación de texto

### Semana 19-20: MLOps aplicado (lo esencial)

* **Temas:** MLflow, Docker, FastAPI, versionado de modelos
* **Práctica:** Microservicio reproducible y versionado
* **Proyecto:** API deployada vía Docker + MLflow tracking

---

## Mes 6: Portafolio sólido + preparación profesional

### Semana 21: Portafolio técnico y publicación

* GitHub limpio con README detallados
* Proyectos públicos en Hugging Face / Gradio
* Blog técnico documentando uno de tus proyectos

### Semana 22-23: Entrevistas y preguntas clave

* Arquitecturas modernas (Transformers, LLMs, RAG)
* Deployment, tradeoffs, escalabilidad
* Evaluación de modelos, debugging en producción
* Mock interviews

### Semana 24: Aplicación laboral y especialización

* Aplicar a roles técnicos con portafolio
* Identificar contribuciones open source o problemas reales
* Elegir área de especialización: agentes, infra, eficiencia, alignment

---

> ✅ **Recordatorio final:** Este itinerario está optimizado para dominar lo esencial que permite construir, escalar y desplegar soluciones reales con LLMs y agentes. Todo lo demás se vuelve mucho más accesible con este núcleo de conocimientos.
