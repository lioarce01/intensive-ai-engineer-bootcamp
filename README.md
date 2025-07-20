# Itinerario Intensivo de 6 Meses para Convertirse en AI Engineer Especializado en LLMs

**Objetivo:** Prepararte como AI Engineer especializado en LLMs, agentes inteligentes y entrenamiento de modelos, listo para trabajar en empresas grandes.

**Duración:** 6 meses (8 horas semanales los fines de semana).

**Formato:** Estudio teórico + ejercicios prácticos desde el inicio, con proyectos aplicados por módulo y cierre con preparación profesional.

---

## Mes 1: Fundamentos matemáticos + NLP moderno

### Semana 1-2: Repaso de álgebra, cálculo y probabilidad (solo lo esencial)

* **Temas:** Vectores, matrices, derivadas parciales, gradiente, probabilidad condicional, entropía
* **Práctica:** Implementar desde cero softmax, cross-entropy, funciones de activación
* **Recursos:**

  * [Essence of Linear Algebra (3Blue1Brown)](https://www.youtube.com/watch?v=fNk_zzaMoSs)
  * [Stanford CS229 notes](https://cs229.stanford.edu/)

### Semana 3-4: NLP clásico y base de embeddings

* **Temas:** Tokenización, TF-IDF, word2vec, GloVe, FastText
* **Práctica:** Explorar y visualizar embeddings (TSNE), entrenar word2vec en corpus propio
* **Proyecto:** Explorador de similitud semántica entre textos cortos
* **Recursos:**

  * [The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/)
  * [Hugging Face Datasets](https://huggingface.co/docs/datasets)

---

## Mes 2: Transformers y LLMs

### Semana 5-6: Arquitectura Transformer en profundidad

* **Temas:** Multi-head attention, Positional Encoding, Encoder vs Decoder
* **Práctica:** Implementación simple de Transformer encoder desde cero
* **Recursos:**

  * [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
  * [Jay Alammar Visual Guides](https://jalammar.github.io/)

### Semana 7-8: LLMs y fine-tuning básico

* **Temas:** Pretraining, fine-tuning, LoRA, QLoRA, PEFT
* **Práctica:** Fine-tuning de LLaMA 2 o Mistral usando Hugging Face + PEFT
* **Proyecto:** Clasificador de texto fine-tuneado para tareas de negocio
* **Recursos:**

  * [Hugging Face PEFT](https://github.com/huggingface/peft)
  * [LoRA Explained](https://lightning.ai/pages/community/tutorial/lora-from-scratch/)

---

## Mes 3: Agentes y RAG

### Semana 9-10: Fundamentos de agentes y herramientas

* **Temas:** LangChain, OpenAI Tools, planificadores, memoria, herramientas custom
* **Práctica:** Crear agentes multi-step con funciones custom (Python)
* **Proyecto:** Asistente automatizado que usa una API externa para responder
* **Recursos:**

  * [LangChain Docs](https://docs.langchain.com/)
  * [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)

### Semana 11-12: Retrieval-Augmented Generation (RAG)

* **Temas:** Chunking, embeddings, vectordb (FAISS, Chroma), pipelines RAG
* **Práctica:** Indexar documentos y crear RAG sobre base local
* **Proyecto:** RAG para documentación técnica de empresa ficticia
* **Recursos:**

  * [HuggingFace RAG Tutorial](https://huggingface.co/blog/rag)
  * [LlamaIndex](https://docs.llamaindex.ai/)

---

## Mes 4: Deep Learning avanzado y entrenamiento de modelos

### Semana 13-14: Entrenamiento desde cero

* **Temas:** Código desde cero para backprop, optimizadores (Adam, SGD), regularización
* **Práctica:** Entrenar MLP sobre texto/tablas
* **Proyecto:** Clasificador desde cero sin usar frameworks
* **Recursos:**

  * [CS231n (Stanford)](https://cs231n.github.io/)

### Semana 15-16: Entrenamiento de LLMs + evaluación

* **Temas:** Dataset curation, tokenización, data collators, evaluación (BLEU, ROUGE, perplexity)
* **Práctica:** Entrenar pequeño modelo autoregresivo con Transformers
* **Proyecto:** Mini GPT entrenado con corpus propio

---

## Mes 5: Despliegue e infraestructura

### Semana 17-18: Infraestructura de inferencia

* **Temas:** vLLM, TGI (Text Generation Inference), optimización en inferencia
* **Práctica:** Servir un modelo usando vLLM con interfaz REST
* **Recursos:**

  * [vLLM GitHub](https://github.com/vllm-project/vllm)
  * [TGI GitHub](https://github.com/huggingface/text-generation-inference)

### Semana 19-20: MLOps aplicado a LLMs

* **Temas:** Tracking (MLflow), versionado, Docker, deployment con FastAPI
* **Proyecto:** Microservicio AI desplegado con FastAPI + Docker
* **Recursos:**

  * [Full Stack LLM Course (Free)](https://fullstackdeeplearning.com/llm-bootcamp/)

---

## Mes 6: Portafolio y entrevistas

### Semana 21: Armado de portafolio

* **Tareas:**

  * Repos GitHub ordenados y con README explicativos
  * Publicar proyectos en Hugging Face o Gradio
  * Crear blog técnico (Medium, Hashnode) explicando un proyecto

### Semana 22-23: Preparación para entrevistas técnicas

* **Temas:**

  * Preguntas comunes de AI Engineer (arquitecturas, tradeoffs, escalabilidad)
  * Problemas en ML (overfitting, drift, bias, evaluación)
  * Repaso de conceptos clave (attention, embeddings, RLHF, etc)
* **Práctica:** Mock interviews y repaso de notebooks

### Semana 24: Aplicaciones y roadmap futuro

* **Tareas:**

  * Aplicar a posiciones con buen ajuste
  * Identificar posibles contribuciones open source
  * Definir área de especialización (e.g. agentes, RAG, MLOps)

---

> 🔍 **Consejo final:** Mantenete actualizado con papers, conferencias (como NeurIPS, ICLR, ICML), y participá en comunidades (Discord, Hugging Face, Twitter). Documentar tu aprendizaje te hará destacar.
