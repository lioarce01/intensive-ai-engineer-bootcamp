# Week 5-6: Transformer Architecture - Multi-Head Attention

This directory contains a comprehensive, educational implementation of the multi-head attention mechanism from scratch in PyTorch.

## Files Overview

### `attention_mechanism.py`
The main implementation file containing:
- **MultiHeadAttention class**: Complete implementation with extensive docstrings
- **Educational annotations**: Detailed comments explaining the mathematics
- **Demonstration script**: Interactive example showing attention patterns
- **Utility functions**: Padding mask creation and attention visualization

### `test_attention.py`
Comprehensive test suite including:
- **Unit tests**: Dimension validation, normalization checks, masking verification
- **Cross-attention demo**: Shows how attention works across different sequences
- **Causal masking demo**: Demonstrates autoregressive attention patterns
- **Head diversity analysis**: Examines how different heads learn different patterns

## Key Concepts Explained

### 1. **Scaled Dot-Product Attention**
```
Attention(Q,K,V) = softmax(QK^T/√d_k)V
```

**Why divide by √d_k?**
- Prevents softmax saturation as dimensions increase
- Keeps dot product variance approximately 1
- Maintains stable gradients during training

### 2. **Multi-Head Attention Benefits**
- **Multiple perspectives**: Different heads capture different relationship types
- **Parallel processing**: All heads computed simultaneously
- **Representation richness**: Combined heads create richer representations

### 3. **Query, Key, Value Intuition**
- **Query**: "What am I looking for?"
- **Key**: "What do I represent?"
- **Value**: "What information do I carry?"

### 4. **Attention Types Demonstrated**
- **Self-attention**: Query = Key = Value (within same sequence)
- **Cross-attention**: Different sequences for Query vs Key/Value
- **Causal attention**: Prevents looking at future positions

## Mathematical Verification

The implementation includes several mathematical checks:
- ✓ Attention weights sum to 1.0 (softmax normalization)
- ✓ Dimension preservation throughout the network
- ✓ Proper masking behavior (masked positions get ~0 attention)
- ✓ Head diversity analysis (different heads learn different patterns)

## Usage Examples

### Basic Self-Attention
```python
from attention_mechanism import MultiHeadAttention

# Initialize attention mechanism
attention = MultiHeadAttention(d_model=512, num_heads=8)

# Apply self-attention
input_embeddings = torch.randn(batch_size, seq_len, d_model)
output, attention_weights = attention(
    query=input_embeddings,
    key=input_embeddings, 
    value=input_embeddings
)
```

### Cross-Attention (e.g., Machine Translation)
```python
# Encoder-decoder attention
encoder_output = torch.randn(batch_size, src_len, d_model)
decoder_states = torch.randn(batch_size, tgt_len, d_model)

output, cross_attention = attention(
    query=decoder_states,     # Target language
    key=encoder_output,       # Source language
    value=encoder_output      # Source language
)
```

### Causal Masking (Language Modeling)
```python
# Create causal mask (lower triangular)
seq_len = 10
causal_mask = torch.tril(torch.ones(seq_len, seq_len))

output, attention_weights = attention(
    query=embeddings, key=embeddings, value=embeddings,
    mask=causal_mask
)
```

## Performance Characteristics

- **Time Complexity**: O(n²d) where n=sequence length, d=model dimension
- **Space Complexity**: O(n²) for attention weight storage
- **Parallelization**: Excellent GPU efficiency
- **Memory considerations**: Quadratic scaling limits very long sequences

## Educational Insights

### Attention Pattern Visualization
The implementation includes visualization tools to see:
- Which tokens attend to which other tokens
- How different heads focus on different relationships
- The effect of masking on attention patterns

### Common Patterns Observed
- **Syntactic heads**: Focus on grammatical relationships
- **Semantic heads**: Capture meaning-based similarities  
- **Positional heads**: Learn distance-based patterns
- **Content heads**: Attend based on token content

## Next Steps

This attention implementation is the foundation for:
1. **Positional Encodings**: Adding position information
2. **Layer Normalization**: Stabilizing training
3. **Feed-Forward Networks**: Adding non-linearity
4. **Complete Transformer Blocks**: Combining all components

## Key Learning Outcomes

After studying this implementation, you should understand:
- How attention creates context-aware representations
- Why scaling by √d_k is crucial for training stability
- How multiple heads capture different relationship types
- The difference between self-attention and cross-attention
- How masking enables different attention patterns
- Performance trade-offs of the attention mechanism

## Running the Code

```bash
# Run the main educational demonstration
python attention_mechanism.py

# Run comprehensive tests and additional demos
python test_attention.py
```

Both scripts include extensive output explaining what's happening at each step, making them excellent educational resources for understanding transformer attention mechanisms.