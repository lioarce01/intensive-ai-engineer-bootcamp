#!/usr/bin/env python3
"""
Interactive Tokenization Tutorial: BPE, WordPiece, and SentencePiece
====================================================================

A simplified version that works on all platforms including Windows.
Demonstrates the three main tokenization algorithms used in modern NLP.
"""

import sys
import torch
from transformers import GPT2Tokenizer, BertTokenizer, T5Tokenizer
from collections import Counter
import re

class TokenizationDemo:
    """Interactive demonstration of different tokenization algorithms."""
    
    def __init__(self):
        print("Loading tokenizers...")
        
        # Load pre-trained tokenizers
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
        
        # Add padding token to GPT-2 if missing
        if self.gpt2_tokenizer.pad_token is None:
            self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token
        
        self.tokenizers = {
            "GPT-2 (BPE)": self.gpt2_tokenizer,
            "BERT (WordPiece)": self.bert_tokenizer, 
            "T5 (SentencePiece)": self.t5_tokenizer
        }
        
        print("All tokenizers loaded successfully!")
        print(f"PyTorch version: {torch.__version__}")
    
    def compare_tokenizers(self, text):
        """Compare how different tokenizers handle the same text."""
        print(f"\n[ANALYSIS] Text: '{text}'")
        print("=" * 60)
        
        results = {}
        
        for name, tokenizer in self.tokenizers.items():
            tokens = tokenizer.tokenize(text)
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            decoded = tokenizer.decode(token_ids)
            
            results[name] = {
                'tokens': tokens,
                'count': len(tokens),
                'ids': token_ids,
                'decoded': decoded,
                'perfect': text.strip().lower() == decoded.strip().lower()
            }
            
            print(f"\n{name}:")
            print(f"  Tokens ({len(tokens)}): {tokens[:8]}")
            if len(tokens) > 8:
                print(f"    ... and {len(tokens)-8} more")
            print(f"  Token IDs: {token_ids[:8]}")
            if len(token_ids) > 8:
                print(f"    ... and {len(token_ids)-8} more")
            print(f"  Perfect reconstruction: {results[name]['perfect']}")
        
        # Summary
        print(f"\n[SUMMARY] Token counts:")
        for name in self.tokenizers.keys():
            algorithm = name.split('(')[1].replace(')', '')
            print(f"  {algorithm}: {results[name]['count']} tokens")
        
        return results
    
    def analyze_algorithm_details(self, text, algorithm_name):
        """Detailed analysis of specific algorithm."""
        print(f"\n[DETAILED ANALYSIS] {algorithm_name}: '{text}'")
        print("-" * 50)
        
        if algorithm_name == "BPE":
            tokenizer = self.gpt2_tokenizer
            tokens = tokenizer.tokenize(text)
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            
            print("BPE Token breakdown:")
            for i, token in enumerate(tokens):
                # Handle BPE space encoding (G with combining character)
                display_token = token.replace('Ġ', '_SPACE_')
                token_id = token_ids[i] if i < len(token_ids) else 'N/A'
                print(f"  {i:2d}: '{display_token}' -> ID {token_id}")
        
        elif algorithm_name == "WordPiece":
            tokenizer = self.bert_tokenizer
            tokens = tokenizer.tokenize(text)
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            
            print("WordPiece Token breakdown:")
            for i, token in enumerate(tokens):
                if token.startswith('##'):
                    feature = "Subword continuation (##)"
                elif token in ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]']:
                    feature = "Special token"
                else:
                    feature = "Word beginning"
                
                token_id = token_ids[i] if i < len(token_ids) else 'N/A'
                print(f"  {i:2d}: '{token}' -> {feature} (ID {token_id})")
        
        elif algorithm_name == "SentencePiece":
            tokenizer = self.t5_tokenizer
            tokens = tokenizer.tokenize(text)
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            
            print("SentencePiece Token breakdown:")
            for i, token in enumerate(tokens):
                if token.startswith('▁'):
                    feature = "Word boundary marker"
                elif token in ['<pad>', '</s>', '<unk>']:
                    feature = "Special token"
                else:
                    feature = "Subword continuation"
                
                token_id = token_ids[i] if i < len(token_ids) else 'N/A'
                print(f"  {i:2d}: '{token}' -> {feature} (ID {token_id})")
    
    def test_oov_handling(self, text):
        """Test how different tokenizers handle out-of-vocabulary words."""
        print(f"\n[OOV TEST] Text: '{text}'")
        print("=" * 50)
        
        for name, tokenizer in self.tokenizers.items():
            tokens = tokenizer.tokenize(text)
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            decoded = tokenizer.decode(token_ids)
            
            # Count UNK tokens
            unk_count = 0
            if name == "BERT (WordPiece)":
                unk_count = tokens.count('[UNK]')
            elif name == "T5 (SentencePiece)":
                unk_count = tokens.count('<unk>')
            
            perfect = text.strip().lower() == decoded.strip().lower()
            
            print(f"\n{name}:")
            print(f"  Tokens: {tokens}")
            print(f"  UNK count: {unk_count}")
            print(f"  Decoded: '{decoded}'")
            print(f"  Perfect reconstruction: {perfect}")
    
    def demonstrate_batch_processing(self):
        """Demonstrate batch processing with padding."""
        print("\n[BATCH PROCESSING] Demonstration")
        print("=" * 40)
        
        batch_texts = [
            "Short text.",
            "Medium length sentence here.",
            "This is a much longer sentence that will demonstrate how padding works with different sequence lengths in batch processing scenarios."
        ]
        
        print("Original texts:")
        for i, text in enumerate(batch_texts):
            display_text = text[:50] + "..." if len(text) > 50 else text
            print(f"  {i+1}: {display_text}")
        
        # Use BERT for demonstration
        tokenizer = self.bert_tokenizer
        
        # Batch encode with padding
        batch_encoding = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=50,
            return_tensors="pt"
        )
        
        print(f"\nBatch encoding results:")
        print(f"  Input shape: {batch_encoding['input_ids'].shape}")
        print(f"  Attention mask shape: {batch_encoding['attention_mask'].shape}")
        
        for i in range(len(batch_texts)):
            attention_sum = batch_encoding['attention_mask'][i].sum().item()
            total_length = batch_encoding['input_ids'].shape[1]
            print(f"  Text {i+1}: {attention_sum} actual tokens, {total_length} total length")
    
    def explore_vocabularies(self):
        """Explore the vocabularies of different tokenizers."""
        print("\n[VOCABULARY EXPLORATION]")
        print("=" * 40)
        
        for name, tokenizer in self.tokenizers.items():
            vocab_size = len(tokenizer)
            print(f"\n{name}:")
            print(f"  Total vocabulary size: {vocab_size:,}")
            
            if hasattr(tokenizer, 'get_vocab'):
                vocab = tokenizer.get_vocab()
                
                if name == "GPT-2 (BPE)":
                    # Find space-prefixed tokens
                    space_tokens = [t for t in list(vocab.keys())[:500] if 'Ġ' in t][:3]
                    display_tokens = [t.replace('Ġ', '_SPACE_') for t in space_tokens]
                    print(f"  Space-prefixed tokens (sample): {display_tokens}")
                
                elif name == "BERT (WordPiece)":
                    # Find special and subword tokens
                    special_tokens = [t for t in vocab.keys() if t.startswith('[') and t.endswith(']')]
                    subword_tokens = [t for t in list(vocab.keys())[:500] if t.startswith('##')][:3]
                    print(f"  Special tokens: {special_tokens}")
                    print(f"  Subword tokens (sample): {subword_tokens}")
                
                elif name == "T5 (SentencePiece)":
                    # Find special and boundary tokens
                    special_tokens = [t for t in vocab.keys() if t.startswith('<') and t.endswith('>')]
                    boundary_tokens = [t for t in list(vocab.keys())[:200] if t.startswith('▁')][:3]
                    print(f"  Special tokens: {special_tokens}")
                    print(f"  Boundary tokens (sample): {boundary_tokens}")


class SimpleBPE:
    """Simplified BPE implementation for educational purposes."""
    
    def __init__(self):
        self.word_freqs = {}
        self.vocab = set()
        self.merges = []
    
    def get_word_frequencies(self, texts):
        """Count word frequencies in the corpus."""
        word_freq = Counter()
        for text in texts:
            words = text.lower().split()
            for word in words:
                # Remove punctuation for simplicity
                word = re.sub(r'[^a-zA-Z]', '', word)
                if word:
                    word_freq[word] += 1
        return word_freq
    
    def get_pairs(self, word_freqs):
        """Get all adjacent character pairs with their frequencies."""
        pairs = Counter()
        for word, freq in word_freqs.items():
            chars = list(word)
            for i in range(len(chars) - 1):
                pairs[(chars[i], chars[i + 1])] += freq
        return pairs
    
    def merge_vocab(self, pair, word_freqs):
        """Merge the most frequent pair in the vocabulary."""
        new_word_freqs = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        for word, freq in word_freqs.items():
            spaced_word = ' '.join(word)
            new_word = spaced_word.replace(bigram, replacement)
            new_word = new_word.replace(' ', '')
            new_word_freqs[new_word] = freq
        
        return new_word_freqs
    
    def train(self, texts, num_merges=10):
        """Train the BPE tokenizer."""
        print(f"\n[BPE TRAINING] Starting with {num_merges} merges")
        print("=" * 50)
        
        # Get word frequencies
        self.word_freqs = self.get_word_frequencies(texts)
        print(f"Training corpus: {len(texts)} texts")
        print(f"Unique words: {len(self.word_freqs)}")
        top_words = dict(list(self.word_freqs.most_common(3)))
        print(f"Most frequent words: {top_words}")
        
        # Initialize vocabulary with characters
        self.vocab = set()
        for word in self.word_freqs:
            for char in word:
                self.vocab.add(char)
        
        print(f"Initial vocabulary size: {len(self.vocab)}")
        
        # Perform merges
        current_word_freqs = self.word_freqs.copy()
        
        for i in range(num_merges):
            pairs = self.get_pairs(current_word_freqs)
            if not pairs:
                break
            
            # Get most frequent pair
            best_pair = pairs.most_common(1)[0][0]
            
            # Merge the pair
            current_word_freqs = self.merge_vocab(best_pair, current_word_freqs)
            
            # Add merged token to vocabulary
            new_token = ''.join(best_pair)
            self.vocab.add(new_token)
            self.merges.append(best_pair)
            
            pair_freq = pairs[best_pair]
            print(f"  Merge {i+1}: {best_pair} -> '{new_token}' (frequency: {pair_freq})")
        
        print(f"\nFinal vocabulary size: {len(self.vocab)}")
    
    def tokenize(self, word):
        """Tokenize a word using learned merges."""
        tokens = list(word.lower())
        
        # Apply merges in order
        for pair in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    tokens = tokens[:i] + [''.join(pair)] + tokens[i + 2:]
                else:
                    i += 1
        
        return tokens


def run_full_demo():
    """Run the complete tokenization tutorial."""
    print("INTERACTIVE TOKENIZATION TUTORIAL")
    print("=" * 50)
    print("Learn BPE, WordPiece, and SentencePiece algorithms!")
    print("This tutorial shows how modern tokenizers work in practice.")
    print()
    
    # Initialize demo
    try:
        demo = TokenizationDemo()
    except Exception as e:
        print(f"Error loading tokenizers: {e}")
        print("Make sure you have transformers library installed:")
        print("pip install torch transformers")
        return
    
    # Test cases
    basic_tests = [
        "Hello, world!",
        "COVID-19 pandemic", 
        "transformer architecture"
    ]
    
    oov_tests = [
        "supercalifragilisticexpialidocious",
        "GPT-3.5-turbo",
        "tokenization"
    ]
    
    # Part 1: Basic Comparisons
    print("\n" + "="*60)
    print("PART 1: Basic Tokenizer Comparison")
    print("="*60)
    
    for test_case in basic_tests:
        demo.compare_tokenizers(test_case)
    
    # Part 2: Algorithm Details
    print("\n" + "="*60)
    print("PART 2: Algorithm-Specific Analysis")
    print("="*60)
    
    analysis_text = "transformer"
    demo.analyze_algorithm_details(analysis_text, "BPE")
    demo.analyze_algorithm_details(analysis_text, "WordPiece") 
    demo.analyze_algorithm_details(analysis_text, "SentencePiece")
    
    # Part 3: OOV Handling
    print("\n" + "="*60)
    print("PART 3: Out-of-Vocabulary Handling")
    print("="*60)
    
    for oov_case in oov_tests:
        demo.test_oov_handling(oov_case)
    
    # Part 4: Batch Processing
    print("\n" + "="*60)
    print("PART 4: Batch Processing Demo")
    print("="*60)
    
    demo.demonstrate_batch_processing()
    
    # Part 5: Vocabulary Exploration
    print("\n" + "="*60)
    print("PART 5: Vocabulary Exploration") 
    print("="*60)
    
    demo.explore_vocabularies()
    
    # Part 6: Build BPE from Scratch
    print("\n" + "="*60)
    print("PART 6: Building BPE from Scratch")
    print("="*60)
    
    simple_bpe = SimpleBPE()
    training_texts = [
        "the quick brown fox jumps over the lazy dog",
        "machine learning is transforming the world",
        "tokenization is an important preprocessing step",
        "transformer models use attention mechanisms"
    ]
    
    simple_bpe.train(training_texts, num_merges=8)
    
    # Test our simple BPE
    print("\nTesting our Simple BPE:")
    test_words = ["the", "quick", "transformer", "machine"]
    for word in test_words:
        tokens = simple_bpe.tokenize(word)
        print(f"  '{word}' -> {tokens}")
    
    # Summary
    print("\n" + "="*60)
    print("TUTORIAL SUMMARY")
    print("="*60)
    
    summary = """
KEY TAKEAWAYS:

BPE (Byte-Pair Encoding) - GPT Style:
+ Excellent OOV handling (falls back to characters)
+ Deterministic and reversible
+ Good for generative tasks
- Can create very long sequences for rare words

WordPiece - BERT Style:  
+ Good balance of vocabulary size vs sequence length
+ Handles morphology reasonably well
+ Good for understanding tasks
- Uses [UNK] for truly unknown words (information loss)

SentencePiece - T5 Style:
+ Language-agnostic (handles any Unicode text)
+ No pre-tokenization step needed
+ Great for multilingual models
+ Fully reversible
- May need larger vocabulary for same coverage

PRACTICAL TIPS:
• Always use the same tokenizer for training and inference
• Pay attention to special tokens and their meanings
• Handle padding and truncation appropriately
• Test on edge cases (URLs, code, mixed languages)
• Consider the trade-off between vocab size and sequence length
• Use batch processing for efficiency
• Validate your tokenization pipeline thoroughly

NEXT STEPS:
• Experiment with domain-specific text
• Try training tokenizers on custom data
• Explore other algorithms (Unigram, etc.)
• Learn about tokenizer alignment and evaluation
"""
    
    print(summary)
    print("\nCongratulations! You've completed the tokenization tutorial!")


def interactive_mode():
    """Run in interactive mode where users can test their own text."""
    print("INTERACTIVE TOKENIZATION MODE")
    print("=" * 40)
    print("Enter your own text to analyze with different tokenizers!")
    print("Type 'quit' to exit\n")
    
    try:
        demo = TokenizationDemo()
    except Exception as e:
        print(f"Error loading tokenizers: {e}")
        return
    
    while True:
        try:
            user_input = input("\nEnter text to analyze: ").strip()
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            if user_input:
                demo.compare_tokenizers(user_input)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    try:
        if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
            interactive_mode()
        else:
            run_full_demo()
    except KeyboardInterrupt:
        print("\nDemo interrupted. Goodbye!")
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have the required packages:")
        print("pip install torch transformers")