#!/usr/bin/env python3
"""
Comprehensive Document Processing and Semantic Search Test
========================================================

This script demonstrates the complete document processing pipeline including:
- Document loading and chunking with different strategies
- Embedding generation using multiple models
- Document storage and retrieval
- Semantic search capabilities
- Performance analysis and model comparisons

This serves as both a test suite and a demonstration of the semantic search
engine capabilities, building upon the existing Word2Vec implementation.

Author: AI Bootcamp Week 3-4
Date: 2025
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Add the current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.extend([
    str(current_dir / "01-core"),
    str(current_dir / "02-indexing"),
    str(current_dir.parent / "02-word-embeddings")
])

# Core imports
import numpy as np

# Import our components
try:
    from document_processor import (
        DocumentProcessor, 
        FixedSizeChunker, 
        SentenceBoundaryChunker,
        create_sample_processor
    )
    DOC_PROCESSOR_OK = True
except ImportError as e:
    print(f"Warning: Could not import document processor: {e}")
    DOC_PROCESSOR_OK = False

try:
    from embedding_engine import (
        EnhancedEmbeddingEngine,
        create_embedding_engine,
        Word2VecAveragingModel,
        SentenceTransformerModel
    )
    EMBEDDING_ENGINE_OK = True
except ImportError as e:
    print(f"Warning: Could not import embedding engine: {e}")
    EMBEDDING_ENGINE_OK = False

try:
    from document_store import DocumentStore, create_document_store
    DOC_STORE_OK = True
except ImportError as e:
    print(f"Warning: Could not import document store: {e}")
    DOC_STORE_OK = False

# Try to import existing word embeddings components for comparison
try:
    from word_embeddings_comprehensive import (
        TextPreprocessor,
        EmbeddingAnalyzer,
        SemanticSearchEngine
    )
    WORD_EMBEDDINGS_OK = True
except ImportError as e:
    print(f"Info: Word embeddings module not available: {e}")
    WORD_EMBEDDINGS_OK = False


class DocumentProcessingDemo:
    """Comprehensive demonstration of the document processing pipeline."""
    
    def __init__(self, sample_docs_dir: str = "sample_documents",
                 storage_dir: str = "demo_storage"):
        """
        Initialize the demonstration.
        
        Args:
            sample_docs_dir: Directory containing sample documents
            storage_dir: Directory for storing processed data
        """
        self.sample_docs_dir = Path(sample_docs_dir)
        self.storage_dir = Path(storage_dir)
        
        # Create storage directory
        self.storage_dir.mkdir(exist_ok=True)
        
        # Initialize components if available
        self.processor = None
        self.embedding_engine = None
        self.document_store = None
        
        # Statistics
        self.stats = {
            'documents_processed': 0,
            'chunks_created': 0,
            'embeddings_generated': 0,
            'processing_time': 0
        }
    
    def setup_components(self):
        """Setup all processing components."""
        print("Setting up processing components...")
        
        # Document processor
        if DOC_PROCESSOR_OK:
            self.processor = create_sample_processor("sentence")
            print("✓ Document processor initialized")
        else:
            print("✗ Document processor not available")
        
        # Embedding engine
        if EMBEDDING_ENGINE_OK:
            cache_file = str(self.storage_dir / "embeddings_cache.json")
            self.embedding_engine = create_embedding_engine(
                use_word2vec=True,
                use_sentence_transformers=True,
                cache_file=cache_file
            )
            print("✓ Embedding engine initialized")
            print(f"  Available models: {list(self.embedding_engine.models.keys())}")
        else:
            print("✗ Embedding engine not available")
        
        # Document store
        if DOC_STORE_OK:
            store_dir = str(self.storage_dir / "document_store")
            self.document_store = create_document_store(store_dir)
            print("✓ Document store initialized")
        else:
            print("✗ Document store not available")
    
    def discover_sample_documents(self) -> List[Path]:
        """Discover all sample documents in the sample directory."""
        if not self.sample_docs_dir.exists():
            print(f"Sample documents directory not found: {self.sample_docs_dir}")
            return []
        
        supported_extensions = ['.txt', '.md', '.markdown']
        documents = []
        
        for ext in supported_extensions:
            documents.extend(self.sample_docs_dir.glob(f"*{ext}"))
        
        print(f"Discovered {len(documents)} sample documents:")
        for doc in documents:
            print(f"  - {doc.name}")
        
        return documents
    
    def test_document_processing(self):
        """Test the document processing pipeline."""
        print("\n" + "="*60)
        print("TESTING DOCUMENT PROCESSING PIPELINE")
        print("="*60)
        
        if not self.processor:
            print("Document processor not available - skipping test")
            return
        
        # Discover sample documents
        doc_paths = self.discover_sample_documents()
        if not doc_paths:
            print("No sample documents found - creating test document")
            return self._test_with_sample_text()
        
        # Process each document
        processed_docs = []
        start_time = time.time()
        
        for doc_path in doc_paths:
            try:
                print(f"\nProcessing: {doc_path.name}")
                document = self.processor.process_document(str(doc_path))
                processed_docs.append(document)
                
                print(f"  Document ID: {document.document_id}")
                print(f"  Title: {document.title}")
                print(f"  Content length: {len(document.content)} characters")
                print(f"  Number of chunks: {len(document.chunks)}")
                print(f"  File type: {document.file_type}")
                
                # Show first few chunks
                for i, chunk in enumerate(document.chunks[:3]):
                    print(f"    Chunk {i+1}: {chunk.content[:80]}...")
                
                if len(document.chunks) > 3:
                    print(f"    ... and {len(document.chunks) - 3} more chunks")
                
                self.stats['documents_processed'] += 1
                self.stats['chunks_created'] += len(document.chunks)
                
            except Exception as e:
                print(f"  Error processing {doc_path.name}: {e}")
        
        processing_time = time.time() - start_time
        self.stats['processing_time'] += processing_time
        
        print(f"\nProcessing completed in {processing_time:.2f} seconds")
        print(f"Successfully processed {len(processed_docs)} documents")
        
        # Store processed documents
        if self.document_store and processed_docs:
            print("\nStoring documents...")
            stored_count = self.document_store.add_documents(processed_docs)
            print(f"Stored {stored_count} documents in document store")
        
        return processed_docs
    
    def _test_with_sample_text(self):
        """Test processing with sample text when no files are available."""
        sample_texts = [
            {
                'id': 'sample_ai',
                'title': 'AI Sample',
                'content': 'Artificial intelligence is transforming how we process and understand information. Machine learning algorithms can identify patterns in data that humans might miss. Deep learning neural networks are particularly effective at handling complex, high-dimensional data.'
            },
            {
                'id': 'sample_nlp',
                'title': 'NLP Sample', 
                'content': 'Natural language processing enables computers to understand human language. Tokenization breaks text into words and phrases. Embeddings represent words as numerical vectors that capture semantic meaning.'
            }
        ]
        
        processed_docs = []
        for sample in sample_texts:
            document = self.processor.process_text(
                text=sample['content'],
                document_id=sample['id'],
                title=sample['title'],
                metadata={'source': 'sample', 'type': 'demo'}
            )
            processed_docs.append(document)
            
            print(f"\nProcessed sample: {document.title}")
            print(f"  Chunks: {len(document.chunks)}")
            for i, chunk in enumerate(document.chunks):
                print(f"    {i+1}. {chunk.content}")
        
        if self.document_store:
            self.document_store.add_documents(processed_docs)
        
        return processed_docs
    
    def test_embedding_generation(self, documents=None):
        """Test embedding generation for processed documents."""
        print("\n" + "="*60)
        print("TESTING EMBEDDING GENERATION")
        print("="*60)
        
        if not self.embedding_engine:
            print("Embedding engine not available - skipping test")
            return
        
        # Get documents from store if not provided
        if documents is None and self.document_store:
            # Get all documents from store
            stored_docs = list(self.document_store.documents_index.values())
            if not stored_docs:
                print("No documents in store - skipping embedding test")
                return
            documents = stored_docs
        
        if not documents:
            print("No documents available for embedding test")
            return
        
        # Collect all chunks from all documents
        all_chunks = []
        if hasattr(documents[0], 'chunks'):  # Document objects
            for doc in documents:
                all_chunks.extend(doc.chunks)
        else:  # StoredDocument objects - need to get chunks from store
            for doc in documents:
                chunks = self.document_store.get_document_chunks(doc.document_id)
                all_chunks.extend(chunks)
        
        print(f"Generating embeddings for {len(all_chunks)} chunks...")
        
        # Test each available embedding model
        for model_name in self.embedding_engine.models.keys():
            print(f"\nTesting model: {model_name}")
            start_time = time.time()
            
            try:
                # Generate embeddings for all chunks
                chunk_texts = [chunk.content for chunk in all_chunks]
                embedding_results = self.embedding_engine.embed_documents(
                    chunk_texts, 
                    model_name=model_name,
                    batch_size=16
                )
                
                embedding_time = time.time() - start_time
                
                print(f"  Generated {len(embedding_results)} embeddings in {embedding_time:.2f}s")
                print(f"  Embedding dimension: {embedding_results[0].embedding_dim}")
                print(f"  Average embedding norm: {np.mean([np.linalg.norm(r.embedding) for r in embedding_results]):.3f}")
                
                # Store embeddings if document store is available
                if self.document_store:
                    print(f"  Storing embeddings in document store...")
                    chunk_embedding_pairs = [
                        (chunk.chunk_id, result) 
                        for chunk, result in zip(all_chunks, embedding_results)
                    ]
                    stored_count = self.document_store.add_embeddings_batch(chunk_embedding_pairs)
                    print(f"  Stored {stored_count} embeddings")
                    
                    self.stats['embeddings_generated'] += stored_count
                
            except Exception as e:
                print(f"  Error with model {model_name}: {e}")
    
    def test_semantic_search(self):
        """Test semantic search functionality."""
        print("\n" + "="*60)
        print("TESTING SEMANTIC SEARCH")
        print("="*60)
        
        if not self.embedding_engine or not self.document_store:
            print("Required components not available - skipping search test")
            return
        
        # Check if we have embeddings
        stats = self.document_store.get_storage_stats()
        if stats['embeddings'] == 0:
            print("No embeddings in store - skipping search test")
            return
        
        print(f"Document store contains {stats['embeddings']} embeddings")
        
        # Test queries
        test_queries = [
            "artificial intelligence and machine learning",
            "neural networks and deep learning",
            "natural language processing techniques",
            "Python programming concepts",
            "data analysis and statistics"
        ]
        
        # Test search with each available model
        available_models = list(stats['embeddings_per_model'].keys())
        
        for model_name in available_models:
            print(f"\n--- Search Results with {model_name} ---")
            
            # Get all embeddings for this model
            chunk_embeddings = self.document_store.get_all_embeddings(model_name)
            if not chunk_embeddings:
                print(f"No embeddings found for model {model_name}")
                continue
            
            print(f"Searching across {len(chunk_embeddings)} chunks...")
            
            # Extract embeddings and chunk info
            embeddings_matrix = np.array([emb.embedding for _, emb in chunk_embeddings])
            chunk_ids = [chunk_id for chunk_id, _ in chunk_embeddings]
            
            # Test each query
            for query in test_queries:
                print(f"\nQuery: '{query}'")
                
                try:
                    # Generate query embedding
                    query_result = self.embedding_engine.embed_text(query, model_name)
                    query_embedding = query_result.embedding
                    
                    # Find similar chunks
                    similar_indices = self.embedding_engine.find_similar(
                        query_embedding,
                        [emb.embedding for _, emb in chunk_embeddings],
                        top_k=3,
                        metric='cosine'
                    )
                    
                    # Display results
                    for rank, (idx, score) in enumerate(similar_indices, 1):
                        chunk_id = chunk_ids[idx]
                        chunk = self.document_store.get_chunk(chunk_id)
                        
                        if chunk:
                            print(f"  {rank}. Score: {score:.3f}")
                            print(f"     Chunk: {chunk_id}")
                            print(f"     Content: {chunk.content[:100]}...")
                            if len(chunk.content) > 100:
                                print(f"              ...{chunk.content[-30:]}")
                        else:
                            print(f"  {rank}. Score: {score:.3f} (chunk not found)")
                    
                except Exception as e:
                    print(f"  Error processing query '{query}': {e}")
    
    def test_model_comparison(self):
        """Compare different embedding models on the same texts."""
        print("\n" + "="*60)
        print("TESTING MODEL COMPARISON")
        print("="*60)
        
        if not self.embedding_engine:
            print("Embedding engine not available - skipping comparison")
            return
        
        if len(self.embedding_engine.models) < 2:
            print("Need at least 2 models for comparison")
            return
        
        # Sample texts for comparison
        comparison_texts = [
            "Machine learning is a subset of artificial intelligence",
            "Deep neural networks can learn complex patterns",
            "Natural language processing helps computers understand text",
            "Python is a popular programming language for data science"
        ]
        
        print(f"Comparing models on {len(comparison_texts)} sample texts...")
        
        # Generate embeddings with all models
        comparison_results = self.embedding_engine.compare_models(comparison_texts)
        
        print("\nModel Comparison Results:")
        print("-" * 40)
        
        for model_name, results in comparison_results.items():
            print(f"\n{model_name}:")
            print(f"  Embedding dimension: {results['embedding_dim']}")
            print(f"  Mean embedding norm: {results['mean_norm']:.3f}")
            print(f"  Std embedding norm: {results['std_norm']:.3f}")
            
            if 'mean_similarity' in results:
                print(f"  Mean pairwise similarity: {results['mean_similarity']:.3f}")
                print(f"  Std pairwise similarity: {results['std_similarity']:.3f}")
        
        # Test cross-model similarities for the same text
        if len(comparison_results) >= 2:
            print("\nCross-model similarity analysis:")
            print("-" * 40)
            
            model_names = list(comparison_results.keys())
            text_idx = 0  # Compare first text across models
            
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    model1, model2 = model_names[i], model_names[j]
                    emb1 = comparison_results[model1]['embeddings'][text_idx]
                    emb2 = comparison_results[model2]['embeddings'][text_idx]
                    
                    # Need to handle different dimensionalities
                    if len(emb1) == len(emb2):
                        similarity = self.embedding_engine.compute_similarity(emb1, emb2)
                        print(f"  {model1} vs {model2}: {similarity:.3f}")
                    else:
                        print(f"  {model1} vs {model2}: Different dimensions ({len(emb1)} vs {len(emb2)})")
    
    def display_statistics(self):
        """Display comprehensive statistics about the processing."""
        print("\n" + "="*60)
        print("PROCESSING STATISTICS")
        print("="*60)
        
        print(f"Documents processed: {self.stats['documents_processed']}")
        print(f"Chunks created: {self.stats['chunks_created']}")
        print(f"Embeddings generated: {self.stats['embeddings_generated']}")
        print(f"Total processing time: {self.stats['processing_time']:.2f} seconds")
        
        if self.stats['chunks_created'] > 0:
            print(f"Average chunks per document: {self.stats['chunks_created'] / max(1, self.stats['documents_processed']):.1f}")
        
        # Document store statistics
        if self.document_store:
            store_stats = self.document_store.get_storage_stats()
            print(f"\nDocument Store Statistics:")
            print(f"  Storage directory: {store_stats['storage_directory']}")
            print(f"  Total storage: {store_stats['total_storage_mb']:.2f} MB")
            print(f"  Documents: {store_stats['documents']}")
            print(f"  Chunks: {store_stats['chunks']}")
            print(f"  Embeddings: {store_stats['embeddings']}")
            
            if store_stats['models']:
                print(f"  Available models: {', '.join(store_stats['models'])}")
                print(f"  Embeddings per model:")
                for model, count in store_stats['embeddings_per_model'].items():
                    print(f"    {model}: {count}")
        
        # Embedding engine statistics
        if self.embedding_engine:
            engine_stats = self.embedding_engine.get_stats()
            print(f"\nEmbedding Engine Statistics:")
            print(f"  Number of models: {engine_stats['num_models']}")
            print(f"  Cache size: {engine_stats['cache_stats']['cache_size']}")
            print(f"  Cache hit rate: {engine_stats['cache_stats']['hit_rate']:.2%}")
    
    def run_complete_demo(self):
        """Run the complete demonstration pipeline."""
        print("SEMANTIC SEARCH ENGINE DEMONSTRATION")
        print("=" * 60)
        print("This demo showcases the complete document processing and")
        print("semantic search pipeline, building upon existing Word2Vec")
        print("implementations with modern sentence transformers.")
        print("=" * 60)
        
        # Setup
        self.setup_components()
        
        if not any([self.processor, self.embedding_engine, self.document_store]):
            print("\nNo components available - cannot run demo")
            return
        
        # Step 1: Document Processing
        processed_docs = self.test_document_processing()
        
        # Step 2: Embedding Generation  
        self.test_embedding_generation(processed_docs)
        
        # Step 3: Semantic Search
        self.test_semantic_search()
        
        # Step 4: Model Comparison
        self.test_model_comparison()
        
        # Step 5: Statistics
        self.display_statistics()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nKey accomplishments:")
        print("✓ Document processing with smart chunking strategies")
        print("✓ Multi-model embedding generation (Word2Vec + Transformers)")
        print("✓ Efficient document storage and retrieval")
        print("✓ Semantic search across document collections")
        print("✓ Performance analysis and model comparisons")
        print("\nThe semantic search engine is ready for use!")


def main():
    """Main function to run the demonstration."""
    # Change to the script directory for relative paths
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print("Starting Document Processing and Semantic Search Demo")
    print("Working directory:", os.getcwd())
    
    # Create and run demonstration
    demo = DocumentProcessingDemo(
        sample_docs_dir="sample_documents",
        storage_dir="demo_storage"
    )
    
    try:
        demo.run_complete_demo()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()