# Semantic Search Web Interface

A complete FastAPI-based web interface for the educational semantic search engine, showcasing the differences between classical and modern search approaches.

## 🌟 Features

### Core Search Capabilities
- **Semantic Search**: Vector similarity search using neural embeddings
- **TF-IDF Search**: Classical term frequency-based search
- **Hybrid Search**: Combined approach with adjustable weights
- **Comparison Mode**: Side-by-side evaluation of all methods

### Educational Interface
- **Interactive Search Options**: Adjustable parameters and weights
- **Real-time Results**: Fast, responsive search with performance metrics
- **Method Explanations**: Educational information about each approach
- **Visual Comparisons**: Clear presentation of search method differences

### Production Features
- **RESTful API**: Complete FastAPI backend with OpenAPI documentation
- **Health Monitoring**: Service health checks and performance metrics
- **Error Handling**: Robust error handling and user feedback
- **Responsive Design**: Works on desktop and mobile devices

## 🚀 Quick Start

### 1. Setup and Installation

```bash
# Install web interface dependencies
pip install -r requirements_web.txt

# Initialize the search system (if not done already)
python search_integration_demo.py

# Start the web interface
python start_web_interface.py
```

### 2. Access the Interface

- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/api/docs
- **Alternative Docs**: http://localhost:8000/api/redoc

### 3. Optional: Custom Configuration

```bash
# Custom host and port
python start_web_interface.py --host 127.0.0.1 --port 8080

# Development mode (auto-reload)
python start_web_interface.py --dev

# Check setup without starting server
python start_web_interface.py --check-only
```

## 📁 Project Structure

```
semantic-search/
├── app.py                     # FastAPI application
├── models.py                  # Pydantic request/response models
├── start_web_interface.py     # Startup script
├── requirements_web.txt       # Web interface dependencies
│
├── static/                    # Frontend assets
│   ├── index.html            # Main web interface
│   ├── style.css             # Responsive styles
│   └── script.js             # Interactive functionality
│
├── 01-core/                   # Search engine components
├── 02-indexing/               # Storage and indexing
└── demo_storage/              # Generated indices and data
```

## 🔧 API Endpoints

### Search Endpoints

#### POST `/api/search`
Main search endpoint supporting all search types.

```json
{
  "query": "machine learning algorithms",
  "search_type": "hybrid",
  "top_k": 10,
  "semantic_weight": 0.6,
  "tfidf_weight": 0.4
}
```

#### POST `/api/compare`
Compare all search methods side by side.

```json
{
  "query": "neural networks deep learning",
  "top_k": 5
}
```

### Information Endpoints

#### GET `/api/health`
Service health check with component status.

#### GET `/api/metrics`
Performance metrics and search statistics.

#### GET `/api/documents`
List of available documents in the search index.

#### GET `/api/models`
Available embedding models and their specifications.

## 🎨 Web Interface Guide

### Search Options

1. **Search Method Selection**:
   - **Semantic**: Uses neural embeddings for context understanding
   - **TF-IDF**: Classical keyword-based search
   - **Hybrid**: Combines both approaches with adjustable weights

2. **Advanced Parameters**:
   - **Results Count**: Number of results to return (1-20)
   - **Hybrid Weights**: Balance between semantic and TF-IDF scoring
   - **Comparison Mode**: Enable side-by-side method comparison

### Results Display

- **Single Method Results**: Detailed results with relevance scores and content snippets
- **Comparison Results**: Side-by-side comparison showing method differences
- **Performance Metrics**: Search time, result count, and method-specific information
- **Highlighted Terms**: Query terms highlighted in search results

### Statistics Panel

Real-time statistics including:
- Total searches performed
- Average search time
- Document count
- Service health status

## 🧠 Educational Value

### Understanding Search Methods

**Semantic Search**:
- ✅ Understands context and meaning
- ✅ Handles synonyms and related concepts
- ❌ May miss exact term matches
- ❌ Computationally intensive

**TF-IDF Search**:
- ✅ Fast and efficient
- ✅ Great for exact term matches
- ✅ Interpretable results
- ❌ Doesn't understand context

**Hybrid Search**:
- ✅ Balances precision and recall
- ✅ Adaptable weighting
- ✅ Robust performance
- ❌ More complex to tune

### Example Queries

Try these queries to see the differences:

1. **Conceptual**: "machine learning algorithms"
2. **Specific**: "neural network backpropagation"
3. **Synonymous**: "AI artificial intelligence"
4. **Technical**: "gradient descent optimization"
5. **Broad**: "data science techniques"

## 🛠️ Development

### Running in Development Mode

```bash
# Auto-reload on file changes
python start_web_interface.py --dev

# Or directly with uvicorn
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Adding New Features

1. **Backend Changes**: Modify `app.py` for new endpoints
2. **Models**: Update `models.py` for new request/response schemas
3. **Frontend**: Edit files in `static/` directory
4. **Styling**: Customize `static/style.css`
5. **Functionality**: Extend `static/script.js`

### Testing the API

Use the interactive documentation at `/api/docs` or test with curl:

```bash
# Health check
curl http://localhost:8000/api/health

# Search request
curl -X POST "http://localhost:8000/api/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "machine learning", "search_type": "semantic", "top_k": 5}'
```

## 📊 Performance Considerations

### Optimization Tips

1. **Caching**: Results and embeddings are cached for better performance
2. **Batch Processing**: Multiple queries can be processed efficiently
3. **Index Management**: FAISS indices are loaded once and reused
4. **Async Operations**: FastAPI handles concurrent requests well

### Scaling Options

1. **Horizontal Scaling**: Deploy multiple instances behind a load balancer
2. **Database Backend**: Replace file-based storage with a database
3. **Container Deployment**: Use Docker for consistent deployments
4. **Cloud Integration**: Deploy to cloud platforms with auto-scaling

## 🔒 Security Considerations

- **Input Validation**: All inputs are validated using Pydantic models
- **Rate Limiting**: Consider adding rate limiting for production use
- **CORS Configuration**: Currently allows all origins (adjust for production)
- **Error Handling**: Errors are handled gracefully without exposing internals

## 🐛 Troubleshooting

### Common Issues

**Service Not Starting**:
- Check if all dependencies are installed: `pip install -r requirements_web.txt`
- Verify search indices are built: `python search_integration_demo.py`
- Check port availability: Try a different port with `--port 8080`

**Search Errors**:
- Ensure document store is populated
- Check if embeddings are generated
- Verify FAISS indices are built

**Performance Issues**:
- Reduce batch sizes in configuration
- Consider using CPU-only FAISS for lighter deployments
- Monitor memory usage with large document collections

### Getting Help

1. Check the logs for detailed error messages
2. Use `--check-only` flag to verify setup
3. Review the API documentation at `/api/docs`
4. Inspect network requests in browser developer tools

## 📚 Educational Context

This web interface is designed for the AI Engineering Bootcamp (Week 3-4) and serves as:

1. **Practical Implementation**: Real-world application of search concepts
2. **Comparative Analysis**: Direct comparison of classical vs. modern approaches
3. **Interactive Learning**: Hands-on experimentation with different methods
4. **Portfolio Piece**: Deployable demonstration of ML engineering skills

The interface is intentionally educational, providing explanations and insights into how different search methods work and when to use each approach.

## 🚀 Next Steps

- **Enhanced UI**: Add more advanced search options and filters
- **Analytics Dashboard**: Detailed search analytics and user behavior tracking
- **Document Management**: Interface for adding and managing documents
- **Advanced Features**: Query expansion, result clustering, personalization
- **Mobile App**: Native mobile application using the API
- **Integration**: Connect with external data sources and APIs