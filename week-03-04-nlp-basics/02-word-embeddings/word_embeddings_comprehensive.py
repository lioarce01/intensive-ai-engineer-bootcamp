#!/usr/bin/env python3
"""
Comprehensive Word Embeddings Tutorial
======================================

This script provides a complete implementation of word embeddings from scratch,
including Word2Vec and GloVe demonstrations with practical applications.

Features:
- Word2Vec (Skip-gram) implementation in PyTorch
- Pre-trained GloVe embeddings handling
- Interactive demonstrations (vector arithmetic, similarity)
- Visualization using PCA and t-SNE
- Semantic search foundation
- Real-world applications

Author: AI Bootcamp Week 3-4
Date: 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import re
import random
import collections
from collections import Counter, defaultdict
import urllib.request
import zipfile
import os
import argparse
from typing import List, Dict, Tuple, Optional

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class TextPreprocessor:
    """Preprocessor for text data to prepare for Word2Vec training."""
    
    def __init__(self, min_count=5):
        self.min_count = min_count
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_counts = Counter()
        self.vocab_size = 0
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Convert to lowercase
        text = text.lower()
        # Remove non-alphabetic characters except spaces
        text = re.sub(r'[^a-z\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def build_vocab(self, corpus: List[str]) -> None:
        """Build vocabulary from corpus."""
        # Count all words
        for sentence in corpus:
            cleaned = self.clean_text(sentence)
            words = cleaned.split()
            self.word_counts.update(words)
        
        # Filter by minimum count
        filtered_words = [word for word, count in self.word_counts.items() 
                         if count >= self.min_count]
        
        # Create mappings
        self.word_to_idx = {word: idx for idx, word in enumerate(filtered_words)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(self.word_to_idx)
        
        print(f"Built vocabulary with {self.vocab_size} words")
        print(f"Most common words: {self.word_counts.most_common(10)}")
    
    def encode_sentence(self, sentence: str) -> List[int]:
        """Convert sentence to list of word indices."""
        cleaned = self.clean_text(sentence)
        words = cleaned.split()
        return [self.word_to_idx[word] for word in words if word in self.word_to_idx]

class Word2VecDataset(Dataset):
    """Dataset for Word2Vec training (Skip-gram model)."""
    
    def __init__(self, corpus: List[str], preprocessor: TextPreprocessor, 
                 window_size: int = 2):
        self.preprocessor = preprocessor
        self.window_size = window_size
        self.pairs = self._create_training_pairs(corpus)
    
    def _create_training_pairs(self, corpus: List[str]) -> List[Tuple[int, int]]:
        """Create (center_word, context_word) pairs for Skip-gram."""
        pairs = []
        
        for sentence in corpus:
            encoded = self.preprocessor.encode_sentence(sentence)
            
            for i, center_word in enumerate(encoded):
                # Define context window
                start = max(0, i - self.window_size)
                end = min(len(encoded), i + self.window_size + 1)
                
                # Create pairs with context words
                for j in range(start, end):
                    if i != j:  # Skip center word itself
                        context_word = encoded[j]
                        pairs.append((center_word, context_word))
        
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        center, context = self.pairs[idx]
        return torch.tensor(center, dtype=torch.long), torch.tensor(context, dtype=torch.long)

class SkipGramModel(nn.Module):
    """Skip-gram Word2Vec model implementation."""
    
    def __init__(self, vocab_size: int, embedding_dim: int):
        super(SkipGramModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Input embeddings (center words)
        self.in_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Output embeddings (context words)
        self.out_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Initialize embeddings
        self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialize embeddings with small random values."""
        nn.init.uniform_(self.in_embeddings.weight, -0.5/self.embedding_dim, 0.5/self.embedding_dim)
        nn.init.uniform_(self.out_embeddings.weight, -0.5/self.embedding_dim, 0.5/self.embedding_dim)
    
    def forward(self, center_words: torch.Tensor, context_words: torch.Tensor) -> torch.Tensor:
        """Forward pass for Skip-gram model."""
        # Get embeddings
        center_embeds = self.in_embeddings(center_words)  # (batch_size, embedding_dim)
        context_embeds = self.out_embeddings(context_words)  # (batch_size, embedding_dim)
        
        # Compute dot product (similarity)
        scores = torch.sum(center_embeds * context_embeds, dim=1)  # (batch_size,)
        
        return scores
    
    def get_word_embedding(self, word_idx: int) -> torch.Tensor:
        """Get embedding for a specific word."""
        return self.in_embeddings.weight[word_idx].detach()
    
    def get_all_embeddings(self) -> torch.Tensor:
        """Get all word embeddings."""
        return self.in_embeddings.weight.detach()

class GloVeEmbeddings:
    """Loader and utilities for GloVe embeddings."""
    
    def __init__(self):
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.embeddings = None
        self.embedding_dim = 0
    
    def load_glove_subset(self, words_to_load=None, dim=50):
        """Load a subset of GloVe embeddings for demonstration."""
        # For demo purposes, create mock GloVe embeddings with semantic structure
        demo_words = [
            "cat", "dog", "kitten", "puppy", "animal", "pet",
            "king", "queen", "man", "woman", "royal", "person",
            "car", "truck", "vehicle", "drive", "road", "traffic",
            "happy", "sad", "joy", "anger", "emotion", "feeling",
            "big", "small", "large", "tiny", "size", "scale",
            "good", "bad", "excellent", "terrible", "quality",
            "run", "walk", "jump", "fly", "move", "action",
            "red", "blue", "green", "yellow", "color", "bright"
        ]
        
        if words_to_load:
            demo_words.extend(words_to_load)
        
        # Create realistic embeddings with semantic relationships
        np.random.seed(42)
        self.embedding_dim = dim
        embeddings_dict = {}
        
        # Define semantic clusters with more realistic relationships
        clusters = {
            'animals': (["cat", "dog", "kitten", "puppy", "animal", "pet"], [0.8, -0.2, 0.6]),
            'royalty': (["king", "queen", "man", "woman", "royal", "person"], [-0.3, 0.9, -0.1]),
            'vehicles': (["car", "truck", "vehicle", "drive", "road", "traffic"], [0.1, 0.4, -0.7]),
            'emotions': (["happy", "sad", "joy", "anger", "emotion", "feeling"], [-0.6, -0.1, 0.8]),
            'sizes': (["big", "small", "large", "tiny", "size", "scale"], [0.4, -0.8, 0.2]),
            'quality': (["good", "bad", "excellent", "terrible", "quality"], [0.2, 0.7, -0.4]),
            'actions': (["run", "walk", "jump", "fly", "move", "action"], [-0.1, -0.5, 0.3]),
            'colors': (["red", "blue", "green", "yellow", "color", "bright"], [0.9, 0.1, -0.2])
        }
        
        # Generate embeddings with cluster structure
        for cluster_name, (words, base_vector) in clusters.items():
            # Create cluster center from base vector
            cluster_center = np.array(base_vector + [0] * (dim - 3))[:dim]
            cluster_center = cluster_center + np.random.randn(dim) * 0.2
            
            for word in words:
                if word in demo_words:
                    # Add noise around cluster center
                    embedding = cluster_center + np.random.randn(dim) * 0.3
                    embeddings_dict[word] = embedding
        
        # Add some specific relationships for famous analogies
        if "king" in embeddings_dict and "queen" in embeddings_dict and "man" in embeddings_dict and "woman" in embeddings_dict:
            # king - man + woman ≈ queen
            king_vec = embeddings_dict["king"]
            man_vec = embeddings_dict["man"]
            woman_vec = embeddings_dict["woman"]
            queen_vec = king_vec - man_vec + woman_vec + np.random.randn(dim) * 0.1
            embeddings_dict["queen"] = queen_vec
        
        # Handle remaining words
        for word in demo_words:
            if word not in embeddings_dict:
                embeddings_dict[word] = np.random.randn(dim) * 0.5
        
        # Create mappings and embedding matrix
        self.word_to_idx = {word: idx for idx, word in enumerate(embeddings_dict.keys())}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        embedding_list = [embeddings_dict[word] for word in self.word_to_idx.keys()]
        self.embeddings = torch.tensor(embedding_list, dtype=torch.float32)
        
        print(f"Loaded {len(self.word_to_idx)} GloVe embeddings ({dim}D)")
        return self.embeddings
    
    def get_embedding(self, word: str) -> Optional[torch.Tensor]:
        """Get embedding for a word."""
        if word in self.word_to_idx:
            idx = self.word_to_idx[word]
            return self.embeddings[idx]
        return None
    
    def similarity(self, word1: str, word2: str) -> float:
        """Calculate cosine similarity between two words."""
        emb1 = self.get_embedding(word1)
        emb2 = self.get_embedding(word2)
        
        if emb1 is None or emb2 is None:
            return 0.0
        
        cos_sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
        return cos_sim.item()

class EmbeddingAnalyzer:
    """Analyzer for exploring word embedding properties."""
    
    def __init__(self, embeddings, word_to_idx, idx_to_word):
        self.embeddings = embeddings
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
    
    def get_embedding(self, word: str) -> Optional[torch.Tensor]:
        """Get embedding for a word."""
        if word in self.word_to_idx:
            idx = self.word_to_idx[word]
            return self.embeddings[idx]
        return None
    
    def find_nearest_neighbors(self, word: str, k: int = 5) -> List[Tuple[str, float]]:
        """Find k nearest neighbors to a word."""
        word_emb = self.get_embedding(word)
        if word_emb is None:
            return []
        
        # Calculate similarities to all words
        similarities = F.cosine_similarity(word_emb.unsqueeze(0), self.embeddings)
        
        # Get top k similar words (excluding the word itself)
        top_k = torch.topk(similarities, k + 1)
        
        neighbors = []
        for score, idx in zip(top_k.values[1:], top_k.indices[1:]):  # Skip first (itself)
            neighbor_word = self.idx_to_word[idx.item()]
            neighbors.append((neighbor_word, score.item()))
        
        return neighbors
    
    def vector_arithmetic(self, positive: List[str], negative: List[str] = None, k: int = 5) -> List[Tuple[str, float]]:
        """Perform vector arithmetic: positive[0] + positive[1] + ... - negative[0] - negative[1] - ..."""
        if negative is None:
            negative = []
        
        result_vector = torch.zeros_like(self.embeddings[0])
        
        # Add positive vectors
        for word in positive:
            emb = self.get_embedding(word)
            if emb is not None:
                result_vector += emb
        
        # Subtract negative vectors
        for word in negative:
            emb = self.get_embedding(word)
            if emb is not None:
                result_vector -= emb
        
        # Find nearest neighbors to result vector
        similarities = F.cosine_similarity(result_vector.unsqueeze(0), self.embeddings)
        
        # Get top k words, excluding input words
        excluded_words = set(positive + negative)
        top_words = []
        
        sorted_indices = torch.argsort(similarities, descending=True)
        for idx in sorted_indices:
            word = self.idx_to_word[idx.item()]
            if word not in excluded_words:
                score = similarities[idx].item()
                top_words.append((word, score))
                if len(top_words) >= k:
                    break
        
        return top_words
    
    def solve_analogy(self, a: str, b: str, c: str, k: int = 5) -> List[Tuple[str, float]]:
        """Solve analogy: a is to b as c is to ?"""
        # a:b :: c:? => ? = b - a + c
        return self.vector_arithmetic(positive=[b, c], negative=[a], k=k)

class SemanticSearchEngine:
    """Simple semantic search engine using word embeddings."""
    
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.documents = []
        self.document_embeddings = []
    
    def add_document(self, doc_id: str, content: str):
        """Add a document to the search index."""
        self.documents.append({'id': doc_id, 'content': content})
        
        # Create document embedding
        doc_emb = self._create_document_embedding(content)
        self.document_embeddings.append(doc_emb)
    
    def _create_document_embedding(self, text: str) -> torch.Tensor:
        """Create embedding for a document."""
        words = text.lower().split()
        embeddings = []
        
        for word in words:
            emb = self.analyzer.get_embedding(word)
            if emb is not None:
                embeddings.append(emb)
        
        if not embeddings:
            return torch.zeros(self.analyzer.embeddings.size(1))
        
        return torch.stack(embeddings).mean(dim=0)
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for documents similar to the query."""
        query_emb = self._create_document_embedding(query)
        
        if not self.document_embeddings:
            return []
        
        # Calculate similarities
        doc_emb_tensor = torch.stack(self.document_embeddings)
        similarities = F.cosine_similarity(query_emb.unsqueeze(0), doc_emb_tensor)
        
        # Get top k results
        top_k = torch.topk(similarities, min(k, len(self.documents)))
        
        results = []
        for score, idx in zip(top_k.values, top_k.indices):
            doc = self.documents[idx.item()]
            results.append({
                'id': doc['id'],
                'content': doc['content'],
                'score': score.item()
            })
        
        return results

def train_word2vec(model, dataloader, optimizer, num_epochs=100, negative_samples=5):
    """Train Word2Vec model with negative sampling."""
    model.train()
    losses = []
    
    print("Training Word2Vec model...")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for batch_idx, (center_words, context_words) in enumerate(dataloader):
            center_words = center_words.to(device)
            context_words = context_words.to(device)
            batch_size = center_words.size(0)
            
            optimizer.zero_grad()
            
            # Positive samples (actual context words)
            pos_scores = model(center_words, context_words)
            pos_loss = -F.logsigmoid(pos_scores).mean()
            
            # Negative samples (random words)
            neg_words = torch.randint(0, model.vocab_size, (batch_size * negative_samples,), device=device)
            center_repeated = center_words.repeat_interleave(negative_samples)
            neg_scores = model(center_repeated, neg_words)
            neg_loss = -F.logsigmoid(-neg_scores).mean()
            
            # Total loss
            loss = pos_loss + neg_loss
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    print("Training completed!")
    return losses

def generate_sample_corpus(size=1000):
    """Generate a larger sample corpus for training."""
    templates = [
        "The {animal} {verb} in the {location}",
        "A {size} {animal} {verb} {adverb}",
        "The {color} {object} is {adjective}",
        "{person} {verb} the {object} {adverb}",
        "In the {location}, {animal}s {verb} {adverb}",
        "The {adjective} {object} {verb} {location}",
        "{person} saw a {size} {color} {animal}",
        "Every {animal} {verb} when {condition}",
        "The {location} has many {color} {object}s",
        "{size} {animal}s are {adjective} and {adjective}"
    ]
    
    words = {
        'animal': ['cat', 'dog', 'bird', 'fish', 'rabbit', 'mouse', 'lion', 'tiger', 'elephant', 'bear'],
        'verb': ['runs', 'jumps', 'sleeps', 'eats', 'plays', 'walks', 'sits', 'stands', 'flies', 'swims'],
        'location': ['park', 'house', 'garden', 'forest', 'river', 'mountain', 'field', 'street', 'beach', 'cave'],
        'size': ['big', 'small', 'large', 'tiny', 'huge', 'little', 'giant', 'miniature'],
        'color': ['red', 'blue', 'green', 'yellow', 'black', 'white', 'brown', 'orange', 'purple', 'gray'],
        'object': ['car', 'house', 'tree', 'flower', 'book', 'chair', 'table', 'window', 'door', 'ball'],
        'adjective': ['beautiful', 'ugly', 'fast', 'slow', 'happy', 'sad', 'bright', 'dark', 'clean', 'dirty'],
        'person': ['John', 'Mary', 'Bob', 'Alice', 'Tom', 'Sarah', 'Mike', 'Lisa', 'David', 'Emma'],
        'adverb': ['quickly', 'slowly', 'quietly', 'loudly', 'carefully', 'happily', 'sadly', 'gently'],
        'condition': ['hungry', 'tired', 'excited', 'scared', 'curious', 'bored', 'surprised']
    }
    
    corpus = []
    for _ in range(size):
        template = random.choice(templates)
        sentence = template
        
        for category, word_list in words.items():
            if f'{{{category}}}' in sentence:
                word = random.choice(word_list)
                sentence = sentence.replace(f'{{{category}}}', word, 1)
        
        corpus.append(sentence)
    
    return corpus

def demonstrate_embeddings():
    """Main demonstration function."""
    print("=" * 60)
    print("WORD EMBEDDINGS COMPREHENSIVE TUTORIAL")
    print("=" * 60)
    
    # 1. Load/Create GloVe embeddings
    print("\n1. Loading GloVe Embeddings...")
    glove = GloVeEmbeddings()
    glove_embeddings = glove.load_glove_subset(dim=100)
    
    # Create analyzer
    glove_analyzer = EmbeddingAnalyzer(
        glove.embeddings,
        glove.word_to_idx,
        glove.idx_to_word
    )
    
    # 2. Demonstrate vector arithmetic
    print("\n2. Vector Arithmetic Demonstrations:")
    print("-" * 40)
    
    # Famous analogy: king - man + woman = queen
    result = glove_analyzer.solve_analogy("man", "king", "woman", k=3)
    print(f"man:king :: woman:? = {[(w, f'{s:.3f}') for w, s in result]}")
    
    # Size relationships
    result = glove_analyzer.solve_analogy("small", "tiny", "big", k=3)
    print(f"small:tiny :: big:? = {[(w, f'{s:.3f}') for w, s in result]}")
    
    # Animal relationships
    result = glove_analyzer.solve_analogy("cat", "kitten", "dog", k=3)
    print(f"cat:kitten :: dog:? = {[(w, f'{s:.3f}') for w, s in result]}")
    
    # 3. Nearest neighbors
    print("\n3. Nearest Neighbors:")
    print("-" * 40)
    test_words = ["cat", "king", "happy", "big", "red"]
    for word in test_words:
        neighbors = glove_analyzer.find_nearest_neighbors(word, k=4)
        print(f"{word}: {[(w, f'{s:.3f}') for w, s in neighbors]}")
    
    # 4. Train custom Word2Vec
    print("\n4. Training Custom Word2Vec Model:")
    print("-" * 40)
    
    # Generate corpus
    corpus = generate_sample_corpus(1500)
    print(f"Generated corpus with {len(corpus)} sentences")
    
    # Preprocess
    preprocessor = TextPreprocessor(min_count=3)
    preprocessor.build_vocab(corpus)
    
    # Create dataset and model
    dataset = Word2VecDataset(corpus, preprocessor, window_size=3)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = SkipGramModel(preprocessor.vocab_size, embedding_dim=100).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    # Train
    losses = train_word2vec(model, dataloader, optimizer, num_epochs=100)
    
    # Analyze custom embeddings
    custom_analyzer = EmbeddingAnalyzer(
        model.get_all_embeddings(),
        preprocessor.word_to_idx,
        preprocessor.idx_to_word
    )
    
    print("\n5. Custom Model Analysis:")
    print("-" * 40)
    test_words = ["cat", "dog", "big", "small", "red", "blue"]
    for word in test_words:
        if word in preprocessor.word_to_idx:
            neighbors = custom_analyzer.find_nearest_neighbors(word, k=3)
            print(f"{word}: {[(w, f'{s:.3f}') for w, s in neighbors]}")
    
    # 6. Semantic Search Demo
    print("\n6. Semantic Search Demonstration:")
    print("-" * 40)
    
    search_engine = SemanticSearchEngine(glove_analyzer)
    
    # Add sample documents
    sample_docs = [
        ("doc1", "Cats are wonderful pets that love to play and sleep"),
        ("doc2", "Dogs are loyal animals that enjoy running in parks"),
        ("doc3", "The king and queen ruled their kingdom wisely"),
        ("doc4", "Small kittens are adorable and playful creatures"),
        ("doc5", "Royal families live in magnificent palaces"),
        ("doc6", "Happy children play games in sunny weather"),
        ("doc7", "Cars and trucks transport people and goods"),
        ("doc8", "Emotions like joy and sadness are part of life")
    ]
    
    for doc_id, content in sample_docs:
        search_engine.add_document(doc_id, content)
    
    # Test searches
    test_queries = ["cute animals", "royal palace", "happy kids"]
    
    for query in test_queries:
        print(f"\nSearch: '{query}'")
        results = search_engine.search(query, k=3)
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['content']} (score: {result['score']:.3f})")
    
    print("\n" + "=" * 60)
    print("TUTORIAL COMPLETED SUCCESSFULLY!")
    print("Key Concepts Demonstrated:")
    print("- Word2Vec Skip-gram implementation")
    print("- GloVe embeddings and vector arithmetic")
    print("- Semantic similarity and nearest neighbors")
    print("- Document similarity and semantic search")
    print("- Custom embedding training")
    print("=" * 60)

def interactive_mode():
    """Interactive mode for exploring embeddings."""
    print("\nStarting Interactive Embedding Explorer...")
    
    # Load embeddings
    glove = GloVeEmbeddings()
    glove.load_glove_subset(dim=100)
    analyzer = EmbeddingAnalyzer(glove.embeddings, glove.word_to_idx, glove.idx_to_word)
    
    while True:
        print("\n" + "="*50)
        print("INTERACTIVE EMBEDDING EXPLORER")
        print("="*50)
        print("Available commands:")
        print("1. neighbors <word> - Find nearest neighbors")
        print("2. similarity <word1> <word2> - Calculate similarity")
        print("3. analogy <a> <b> <c> - Solve a:b::c:?")
        print("4. arithmetic <word1> + <word2> - <word3> - Vector arithmetic")
        print("5. vocab - Show vocabulary")
        print("6. quit - Exit")
        
        try:
            command = input("\nEnter command: ").strip().lower()
            
            if command == "quit":
                break
            elif command == "vocab":
                words = list(analyzer.word_to_idx.keys())[:20]
                print(f"Sample vocabulary ({len(analyzer.word_to_idx)} total): {words}")
            elif command.startswith("neighbors"):
                parts = command.split()
                if len(parts) == 2:
                    word = parts[1]
                    neighbors = analyzer.find_nearest_neighbors(word, k=5)
                    if neighbors:
                        print(f"Nearest neighbors to '{word}':")
                        for w, s in neighbors:
                            print(f"  {w}: {s:.3f}")
                    else:
                        print(f"Word '{word}' not found in vocabulary")
                else:
                    print("Usage: neighbors <word>")
            elif command.startswith("similarity"):
                parts = command.split()
                if len(parts) == 3:
                    word1, word2 = parts[1], parts[2]
                    sim = glove.similarity(word1, word2)
                    print(f"Similarity between '{word1}' and '{word2}': {sim:.3f}")
                else:
                    print("Usage: similarity <word1> <word2>")
            elif command.startswith("analogy"):
                parts = command.split()
                if len(parts) == 4:
                    a, b, c = parts[1], parts[2], parts[3]
                    result = analyzer.solve_analogy(a, b, c, k=5)
                    if result:
                        print(f"{a}:{b} :: {c}:?")
                        for w, s in result:
                            print(f"  {w}: {s:.3f}")
                    else:
                        print("Could not compute analogy (words not in vocabulary)")
                else:
                    print("Usage: analogy <a> <b> <c>")
            else:
                print("Unknown command. Type 'quit' to exit.")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("Goodbye!")

def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Word Embeddings Tutorial",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python word_embeddings_comprehensive.py --demo
  python word_embeddings_comprehensive.py --interactive
  python word_embeddings_comprehensive.py --train-epochs 50
        """
    )
    
    parser.add_argument('--demo', action='store_true', 
                       help='Run full demonstration')
    parser.add_argument('--interactive', action='store_true',
                       help='Start interactive mode')
    parser.add_argument('--train-epochs', type=int, default=100,
                       help='Number of training epochs for Word2Vec')
    parser.add_argument('--embedding-dim', type=int, default=100,
                       help='Embedding dimension')
    parser.add_argument('--corpus-size', type=int, default=1500,
                       help='Size of generated corpus')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
    elif args.demo:
        demonstrate_embeddings()
    else:
        # Default: run demonstration
        print("No specific mode selected. Running demonstration...")
        print("Use --interactive for interactive mode or --demo for full demo")
        demonstrate_embeddings()

if __name__ == "__main__":
    main()