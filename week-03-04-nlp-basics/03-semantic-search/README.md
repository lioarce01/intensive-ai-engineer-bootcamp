# 03-Semantic Search: Complete Pipeline

Building upon Week 3-4 tokenization and word embeddings to create a functional semantic search engine.

## 📁 File Structure & Purpose

### Core Components (`01-core/`)

**`document_processor.py`**
- **Purpose**: Document parsing, text cleaning, and intelligent chunking
- **Key Features**: Multiple chunking strategies (fixed-size, sentence-boundary, semantic)
- **Integration**: Uses existing TextPreprocessor from word embeddings module
- **Output**: Structured document chunks with metadata

**`embedding_engine.py`**
- **Purpose**: Generate embeddings using multiple approaches for comparison
- **Models**: Word2Vec averaging (from your trained model) + Sentence Transformers
- **Key Classes**: `EnhancedEmbeddingEngine`, `Word2VecAveragingModel`, `SentenceTransformerModel`
- **Output**: Dual embeddings (100D Word2Vec + 384D Sentence Transformer)

**`search_engine.py`**
- **Purpose**: FAISS-powered semantic search with multiple embedding models
- **Features**: Vector similarity search, batch processing, performance metrics
- **Integration**: Works with embedding engine and document store

**`tfidf_search.py`**
- **Purpose**: Classical TF-IDF search for baseline comparison
- **Features**: Advanced preprocessing, stemming, n-grams, explainable results
- **Output**: Keyword-based search with matched terms highlighting

**`hybrid_ranker.py`**
- **Purpose**: Combines semantic and TF-IDF search results
- **Strategies**: Linear fusion, RRF, CombSum, CombMNZ
- **Educational**: Shows evolution from classical to modern search

### Storage Layer (`02-indexing/`)

**`document_store.py`**
- **Purpose**: Persistent storage for documents, chunks, and embeddings
- **Features**: JSON-based document metadata, pickle-based embedding cache
- **Structure**: Organized storage with document/chunk hierarchy
- **Efficiency**: Caching system to avoid re-computing embeddings

**`vector_store.py`**
- **Purpose**: FAISS vector indices for efficient similarity search
- **Features**: Multiple embedding model support, persistent storage
- **Performance**: Sub-millisecond search times on 57 document chunks

### Web Application (`webapp/`)

**`app.py`**
- **Purpose**: Production FastAPI application with full ML pipeline
- **Features**: RESTful API, search endpoints, health monitoring
- **Integration**: Complete semantic search system integration

**`demo_app.py`**
- **Purpose**: Lightweight demo version with mock data
- **Usage**: Quick start without ML dependencies
- **Educational**: Shows API structure and interface design

**`static/`**
- **Purpose**: Modern responsive web interface
- **Files**: `index.html`, `style.css`, `script.js`
- **Features**: Method comparison, real-time search, educational explanations

**`start_web_interface.py`**
- **Purpose**: Production startup script with dependency checking
- **Features**: Automatic component initialization and health checks

### Testing & Demo

**`test_document_processing.py`**
- **Purpose**: Comprehensive demo of the entire pipeline
- **Features**: Document processing, embedding generation, search demonstrations
- **Educational**: Shows progression from Word2Vec to modern transformers

**`search_integration_demo.py`**
- **Purpose**: Complete FAISS + TF-IDF + Hybrid search demonstration
- **Features**: Performance analysis, quality comparison, fusion strategies
- **Output**: Comprehensive system testing and benchmarking

**`simple_search_demo.py`**
- **Purpose**: Quick verification that components are working
- **Output**: Status check of all major components
- **Usage**: Fast smoke test of the system

### Data

**`sample_documents/`**
- 5 technical documents (AI, ML, NLP, Deep Learning, Python)
- Pre-processed and chunked for testing
- 80+ embedded chunks in storage

**`demo_storage/`**
- Document metadata (JSON)
- Cached embeddings (pickle files)
- FAISS indices for vector search
- TF-IDF models and vocabularies

## 🚀 Current Status

✅ **Complete & Working:**
- Document processing pipeline
- Dual embedding generation (Word2Vec + Transformers)
- Persistent storage system
- 80+ document chunks indexed
- Educational comparison framework

⏳ **Next Steps (Option A):**
- FAISS vector search implementation
- FastAPI web interface
- TF-IDF vs embedding comparison
- Complete web deployment

## 🧪 Quick Test

### Core System Test
```bash
cd 03-semantic-search
python simple_search_demo.py
```

### Complete Search Integration Demo  
```bash
python search_integration_demo.py
```

### Web Interface
```bash
cd webapp
python demo_app.py  # Demo mode (no ML dependencies)
# OR
python start_web_interface.py  # Full implementation
```

**Expected Output:**
- SentenceTransformer model loaded
- Embedding engine with multiple models initialized
- FAISS indices loaded successfully  
- Document store ready with 57 chunks
- Web interface available at http://localhost:8000

## 📚 Educational Value

This implementation demonstrates:
- **Evolution**: From Word2Vec (Week 2) to modern transformers
- **Architecture**: Clean separation of processing, embedding, and storage
- **Comparison**: Side-by-side classical vs modern approaches
- **Scalability**: Foundation ready for FAISS and web deployment

---

**Built for AI Engineering Bootcamp Week 3-4**  
*Next: FAISS + FastAPI web interface*