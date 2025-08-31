"""
Multi-Head Attention Mechanism Implementation from Scratch

This module implements the multi-head attention mechanism as described in 
"Attention Is All You Need" (Vaswani et al., 2017) with extensive educational
annotations to help understand the mathematical concepts and implementation details.

Educational Focus:
- Understanding the core concepts of queries, keys, and values
- Mathematics behind scaled dot-product attention
- Why multiple attention heads are beneficial
- How attention creates context-aware representations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism implementation with educational annotations.
    
    The attention mechanism allows each position in a sequence to attend to all
    positions in the input sequence. This creates context-aware representations
    where each token's representation is influenced by all other tokens.
    
    Key Concepts:
    - Queries (Q): "What am I looking for?" - representations of positions that need context
    - Keys (K): "What do I contain?" - representations used to compute attention scores
    - Values (V): "What information do I carry?" - actual content to be aggregated
    
    The attention formula: Attention(Q,K,V) = softmax(QK^T/√d_k)V
    
    Args:
        d_model (int): Dimension of the model (input/output dimension)
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability for attention weights
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        
        # Validate inputs
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads}). "
            f"This ensures each head has integer dimension d_k = d_model / num_heads."
        )
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # WHY d_k = d_model / num_heads?
        # We want to keep the total parameter count and computational complexity
        # similar to single-head attention while gaining the benefits of multiple
        # perspectives. Each head operates on a smaller dimension (d_k) but we
        # have multiple heads that are later concatenated.
        
        # Linear transformations for Q, K, V
        # These project the input into query, key, and value spaces
        self.W_q = nn.Linear(d_model, d_model, bias=False)  # Query projection
        self.W_k = nn.Linear(d_model, d_model, bias=False)  # Key projection  
        self.W_v = nn.Linear(d_model, d_model, bias=False)  # Value projection
        
        # Final output projection
        # After concatenating all heads, we need another linear transformation
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights using Xavier/Glorot initialization
        # This helps with training stability by keeping gradients well-scaled
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize weights using Xavier uniform initialization.
        
        This initialization helps maintain the variance of activations and gradients
        throughout the network, which is crucial for training stability.
        """
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)
    
    def scaled_dot_product_attention(
        self, 
        Q: torch.Tensor, 
        K: torch.Tensor, 
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.
        
        The core attention mechanism that computes how much each position should
        attend to every other position in the sequence.
        
        Mathematical Formula:
        Attention(Q,K,V) = softmax(QK^T/√d_k)V
        
        Args:
            Q: Query tensor of shape (batch_size, num_heads, seq_len, d_k)
            K: Key tensor of shape (batch_size, num_heads, seq_len, d_k) 
            V: Value tensor of shape (batch_size, num_heads, seq_len, d_k)
            mask: Optional mask tensor to prevent attention to certain positions
            
        Returns:
            Tuple of (attention_output, attention_weights)
        """
        
        # Step 1: Compute attention scores
        # QK^T gives us similarity scores between queries and keys
        # Shape: (batch_size, num_heads, seq_len, seq_len)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))
        
        # Step 2: Scale by √d_k
        # WHY DO WE DIVIDE BY √d_k?
        # As d_k gets larger, the dot products grow larger in magnitude, pushing
        # the softmax function into regions where gradients are extremely small.
        # Scaling by √d_k keeps the variance of the dot products approximately 1,
        # ensuring that the softmax doesn't saturate.
        #
        # Mathematical intuition:
        # If Q and K have unit variance, QK^T will have variance ≈ d_k
        # Dividing by √d_k normalizes this back to unit variance
        attention_scores = attention_scores / math.sqrt(self.d_k)
        
        # Step 3: Apply mask (if provided)
        # Masks are used to prevent attention to certain positions
        # (e.g., padding tokens, future tokens in causal attention)
        if mask is not None:
            # Set masked positions to a very large negative number
            # so they become ~0 after softmax
            attention_scores.masked_fill_(mask == 0, -1e9)
        
        # Step 4: Apply softmax to get attention weights
        # This ensures attention weights sum to 1 across the sequence dimension
        # Shape: (batch_size, num_heads, seq_len, seq_len)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Step 5: Apply dropout to attention weights
        # This prevents overfitting and improves generalization
        # Note: During training, dropout zeros out some attention weights
        # but we don't renormalize, which is the standard practice in transformers
        if self.training:
            attention_weights = self.dropout(attention_weights)
        
        # Step 6: Apply attention weights to values
        # This is where we actually aggregate information based on attention
        # Each position gets a weighted combination of all value vectors
        # Shape: (batch_size, num_heads, seq_len, d_k)
        attention_output = torch.matmul(attention_weights, V)
        
        return attention_output, attention_weights
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of multi-head attention.
        
        This implements the full multi-head attention mechanism:
        1. Linear projections to Q, K, V
        2. Reshape for multiple heads
        3. Apply scaled dot-product attention for each head
        4. Concatenate heads
        5. Final linear projection
        
        Args:
            query: Query tensor of shape (batch_size, seq_len, d_model)
            key: Key tensor of shape (batch_size, seq_len, d_model)
            value: Value tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
            - output: Shape (batch_size, seq_len, d_model)
            - attention_weights: Shape (batch_size, num_heads, seq_len, seq_len)
        """
        
        batch_size, query_seq_len, d_model = query.size()
        key_seq_len = key.size(1)
        value_seq_len = value.size(1)
        
        # Validate input dimensions
        assert d_model == self.d_model, (
            f"Input d_model ({d_model}) doesn't match initialized d_model ({self.d_model})"
        )
        assert key_seq_len == value_seq_len, (
            f"Key sequence length ({key_seq_len}) must match value sequence length ({value_seq_len})"
        )
        
        # Step 1: Linear projections
        # Transform inputs into query, key, and value representations
        Q = self.W_q(query)  # (batch_size, query_seq_len, d_model)
        K = self.W_k(key)    # (batch_size, key_seq_len, d_model)
        V = self.W_v(value)  # (batch_size, value_seq_len, d_model)
        
        # Step 2: Reshape for multiple heads
        # We need to split the d_model dimension into num_heads * d_k
        # and rearrange so each head can operate independently
        
        # Reshape: (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, d_k)
        Q = Q.view(batch_size, query_seq_len, self.num_heads, self.d_k)
        K = K.view(batch_size, key_seq_len, self.num_heads, self.d_k)
        V = V.view(batch_size, value_seq_len, self.num_heads, self.d_k)
        
        # Transpose: (batch_size, seq_len, num_heads, d_k) -> (batch_size, num_heads, seq_len, d_k)
        # This puts the head dimension second, making it easier to apply attention
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Step 3: Apply scaled dot-product attention for each head
        # Each head learns to focus on different types of relationships
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Step 4: Concatenate heads
        # Transpose back: (batch_size, num_heads, query_seq_len, d_k) -> (batch_size, query_seq_len, num_heads, d_k)
        attention_output = attention_output.transpose(1, 2)
        
        # Reshape to concatenate heads: (batch_size, query_seq_len, num_heads, d_k) -> (batch_size, query_seq_len, d_model)
        attention_output = attention_output.contiguous().view(batch_size, query_seq_len, self.d_model)
        
        # Step 5: Final linear projection
        # This allows the model to learn how to best combine the different heads
        output = self.W_o(attention_output)
        
        return output, attention_weights


def create_padding_mask(seq: torch.Tensor, pad_token: int = 0) -> torch.Tensor:
    """
    Create a padding mask to prevent attention to padding tokens.
    
    Args:
        seq: Input sequence tensor of shape (batch_size, seq_len)
        pad_token: Token ID used for padding
        
    Returns:
        Mask tensor of shape (batch_size, 1, 1, seq_len) where 1 = attend, 0 = mask
    """
    mask = (seq != pad_token).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
    return mask


def visualize_attention_pattern(attention_weights: torch.Tensor, tokens: list, head_idx: int = 0):
    """
    Simple attention visualization helper.
    
    Args:
        attention_weights: Tensor of shape (batch_size, num_heads, seq_len, seq_len)
        tokens: List of token strings
        head_idx: Which attention head to visualize
    """
    import numpy as np
    
    # Extract attention weights for the specified head (first batch item)
    attn = attention_weights[0, head_idx].detach().numpy()
    
    print(f"\nAttention Pattern for Head {head_idx}:")
    print("Rows = Query positions, Columns = Key positions")
    print("Higher values = stronger attention\n")
    
    # Print header
    print("Query\\Key", end="\t")
    for token in tokens:
        print(f"{token[:8]:>8}", end="\t")
    print()
    
    # Print attention matrix
    for i, query_token in enumerate(tokens):
        print(f"{query_token[:8]:>8}", end="\t")
        for j in range(len(tokens)):
            print(f"{attn[i, j]:.3f}", end="\t\t")
        print()


def demo_attention_mechanism():
    """
    Educational demonstration of multi-head attention with detailed explanations.
    """
    print("=" * 80)
    print("MULTI-HEAD ATTENTION EDUCATIONAL DEMONSTRATION")
    print("=" * 80)
    
    # Model parameters
    d_model = 512      # Model dimension
    num_heads = 8      # Number of attention heads
    seq_len = 6        # Sequence length
    batch_size = 1     # Batch size for demo
    
    print(f"\nModel Configuration:")
    print(f"- d_model: {d_model}")
    print(f"- num_heads: {num_heads}")
    print(f"- d_k (dimension per head): {d_model // num_heads}")
    
    # Initialize the attention module
    attention = MultiHeadAttention(d_model, num_heads, dropout=0.0)  # No dropout for demo
    
    # Create sample input (simulating token embeddings)
    # In practice, these would come from an embedding layer
    torch.manual_seed(42)  # For reproducible results
    sample_input = torch.randn(batch_size, seq_len, d_model)
    
    print(f"\nInput Shape: {sample_input.shape}")
    print("(batch_size, seq_len, d_model)")
    
    # For self-attention, query, key, and value are all the same input
    print("\n" + "="*50)
    print("SELF-ATTENTION CONCEPT")
    print("="*50)
    print("In self-attention:")
    print("- Query = Key = Value = Input embeddings")
    print("- Each position can attend to all positions (including itself)")
    print("- This allows tokens to gather context from the entire sequence")
    
    # Forward pass
    print("\nPerforming forward pass...")
    output, attention_weights = attention(
        query=sample_input,
        key=sample_input, 
        value=sample_input
    )
    
    print(f"\nOutput Shape: {output.shape}")
    print(f"Attention Weights Shape: {attention_weights.shape}")
    print("(batch_size, num_heads, seq_len, seq_len)")
    
    # Analyze attention patterns
    print("\n" + "="*50)
    print("ATTENTION PATTERN ANALYSIS")
    print("="*50)
    
    # Sample tokens for visualization
    sample_tokens = ["The", "cat", "sat", "on", "the", "mat"]
    
    print("\nAttention weights tell us how much each position attends to every other position.")
    print("Each row sums to 1.0 (softmax normalization).")
    
    # Show attention pattern for first head
    visualize_attention_pattern(attention_weights, sample_tokens, head_idx=0)
    
    # Show different heads learn different patterns
    if num_heads > 1:
        print(f"\n" + "-"*50)
        print("COMPARING DIFFERENT ATTENTION HEADS")
        print("-"*50)
        print("Different heads often learn to focus on different types of relationships:")
        print("- Some heads might focus on syntactic relationships")
        print("- Others might capture semantic similarities")
        print("- Some might learn positional patterns")
        
        visualize_attention_pattern(attention_weights, sample_tokens, head_idx=1)
    
    # Demonstrate masking
    print("\n" + "="*50)
    print("ATTENTION MASKING DEMONSTRATION")
    print("="*50)
    
    # Create a simple padding mask (mask last 2 positions)
    mask = torch.ones(batch_size, 1, 1, seq_len)
    mask[:, :, :, -2:] = 0  # Mask last 2 positions
    
    print("Masking last 2 positions to simulate padding tokens...")
    masked_output, masked_attention = attention(
        query=sample_input,
        key=sample_input,
        value=sample_input,
        mask=mask
    )
    
    print("\nMasked Attention Pattern (Head 0):")
    masked_tokens = ["The", "cat", "sat", "on", "[PAD]", "[PAD]"]
    visualize_attention_pattern(masked_attention, masked_tokens, head_idx=0)
    print("Note: Attention to [PAD] positions should be ~0.0")
    
    # Key insights
    print("\n" + "="*50)
    print("KEY INSIGHTS ABOUT MULTI-HEAD ATTENTION")
    print("="*50)
    print("""
1. CONTEXT AGGREGATION: Each output position is a weighted combination of all input positions
2. PARALLEL PROCESSING: Multiple heads can learn different types of relationships simultaneously
3. SCALABILITY: Attention can handle sequences of varying lengths efficiently
4. INTERPRETABILITY: Attention weights provide insight into what the model is focusing on
5. FLEXIBILITY: The same mechanism works for self-attention, encoder-decoder attention, etc.
    """)
    
    # Mathematical verification
    print("MATHEMATICAL VERIFICATION:")
    print(f"- Each attention row sums to ~1.0: {attention_weights[0, 0, 0, :].sum().item():.6f}")
    print(f"- Output preserves d_model dimension: {output.shape[-1] == d_model}")
    print(f"- Batch and sequence dimensions unchanged: {output.shape[:2] == sample_input.shape[:2]}")


if __name__ == "__main__":
    # Run the educational demonstration
    demo_attention_mechanism()
    
    print("\n" + "="*80)
    print("ADDITIONAL LEARNING EXERCISES")
    print("="*80)
    print("""
Try these modifications to deepen your understanding:

1. Change num_heads and observe how attention patterns change
2. Experiment with different d_model sizes
3. Try cross-attention by using different query vs key/value inputs
4. Implement causal masking for language modeling
5. Add position encodings to the inputs
6. Visualize attention patterns on real text sequences

Key files to explore next:
- Positional encodings
- Layer normalization
- Feed-forward networks
- Complete transformer blocks
    """)