"""
Complete Transformer Encoder Implementation
==========================================

This module implements a complete Transformer encoder by integrating all the components
we've built: multi-head attention, positional encoding, feed-forward networks, 
layer normalization, and residual connections.

Educational Focus:
- How all components work together in the encoder stack
- Information flow through multiple encoder layers
- Shape transformations and data processing pipeline
- Real-world usage patterns and best practices
- Performance considerations and optimization strategies

Architecture Overview:
Input → Token Embeddings → Positional Encoding → Encoder Stack → Output Representations
        ↓
    Each Encoder Layer:
    1. Multi-Head Self-Attention + Residual + LayerNorm
    2. Feed-Forward Network + Residual + LayerNorm

Author: AI Intensive Bootcamp - Week 5-6
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, List, Dict, Union
import warnings

# Import our custom components
try:
    from attention_mechanism import MultiHeadAttention
    from positional_encoding import SinusoidalPositionalEncoding
    from feedforward import PositionwiseFeedForward
    from normalization_residuals import LayerNorm
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Using built-in implementations...")
    
    # Fallback implementations
    class MultiHeadAttention(nn.Module):
        def __init__(self, d_model, num_heads, dropout=0.1):
            super().__init__()
            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads
            
            self.w_q = nn.Linear(d_model, d_model)
            self.w_k = nn.Linear(d_model, d_model)
            self.w_v = nn.Linear(d_model, d_model)
            self.w_o = nn.Linear(d_model, d_model)
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, query, key, value, mask=None):
            batch_size, seq_len, d_model = query.shape
            
            Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            context = torch.matmul(attn_weights, V)
            context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
            
            output = self.w_o(context)
            return output, attn_weights
    
    class SinusoidalPositionalEncoding(nn.Module):
        def __init__(self, d_model, max_seq_length=5000, dropout=0.1):
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            
            pe = torch.zeros(max_seq_length, d_model)
            position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)
            
            self.register_buffer('pe', pe)
        
        def forward(self, x):
            x = x + self.pe[:x.size(0), :]
            return self.dropout(x)
    
    class PositionwiseFeedForward(nn.Module):
        def __init__(self, d_model, d_ff, dropout=0.1, activation='gelu'):
            super().__init__()
            self.linear1 = nn.Linear(d_model, d_ff)
            self.linear2 = nn.Linear(d_ff, d_model)
            self.dropout = nn.Dropout(dropout)
            self.activation = getattr(F, activation) if hasattr(F, activation) else F.relu
        
        def forward(self, x):
            return self.linear2(self.dropout(self.activation(self.linear1(x))))
    
    class LayerNorm(nn.Module):
        def __init__(self, d_model, eps=1e-6):
            super().__init__()
            self.gamma = nn.Parameter(torch.ones(d_model))
            self.beta = nn.Parameter(torch.zeros(d_model))
            self.eps = eps
        
        def forward(self, x):
            mean = x.mean(-1, keepdim=True)
            std = x.std(-1, keepdim=True)
            return self.gamma * (x - mean) / (std + self.eps) + self.beta


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer
    
    Architecture (Pre-Norm):
    x → LayerNorm → Multi-Head Attention → Add & Residual → 
    → LayerNorm → Feed-Forward → Add & Residual → Output
    
    This is the fundamental building block of the Transformer encoder.
    Each layer transforms the input representations, allowing the model
    to build increasingly sophisticated representations of the input.
    
    Key Components:
    1. Multi-Head Self-Attention: Captures relationships between positions
    2. Position-wise Feed-Forward: Processes information at each position
    3. Residual Connections: Enable training of deep networks
    4. Layer Normalization: Stabilizes training and improves convergence
    
    Educational Notes:
    - Pre-norm architecture is used (modern standard)
    - Self-attention allows each position to attend to all positions
    - Feed-forward provides non-linear transformations
    - Residual connections preserve information flow
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_pre_norm: bool = True
    ):
        """
        Initialize a single Transformer encoder layer.
        
        Args:
            d_model: Model dimension (embedding size)
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension (usually 4 * d_model)
            dropout: Dropout probability for regularization
            use_pre_norm: Whether to use pre-norm (True) or post-norm (False)
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.use_pre_norm = use_pre_norm
        
        # Multi-head self-attention mechanism
        self.self_attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Position-wise feed-forward network
        self.feed_forward = PositionwiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation='gelu'  # GELU is commonly used in modern Transformers
        )
        
        # Layer normalization and residual connections
        if use_pre_norm:
            # Pre-norm: LayerNorm → Sublayer → Residual
            self.attn_layer_norm = LayerNorm(d_model)
            self.ff_layer_norm = LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
        else:
            # Post-norm: Sublayer → Residual → LayerNorm
            self.attn_layer_norm = LayerNorm(d_model)
            self.ff_layer_norm = LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the encoder layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask of shape (batch_size, 1, 1, seq_len)
            return_attention: Whether to return attention weights
            
        Returns:
            If return_attention=False: Output tensor of shape (batch_size, seq_len, d_model)
            If return_attention=True: Tuple of (output, attention_weights)
        
        Information Flow:
        1. Input representations pass through multi-head self-attention
        2. Attention output is combined with input via residual connection
        3. Result passes through feed-forward network
        4. Feed-forward output is combined with residual connection
        5. Final output maintains same shape as input
        """
        
        # Store original input for residual connections
        residual_1 = x
        
        if self.use_pre_norm:
            # Pre-norm architecture (modern standard)
            
            # Step 1: Self-Attention Sublayer
            # LayerNorm → Self-Attention → Dropout → Residual
            x_norm = self.attn_layer_norm(x)
            attn_output, attention_weights = self.self_attention(
                query=x_norm,
                key=x_norm,
                value=x_norm,
                mask=mask
            )
            # Apply dropout and residual connection
            x = residual_1 + self.dropout(attn_output)
            
            # Step 2: Feed-Forward Sublayer
            # LayerNorm → Feed-Forward → Dropout → Residual
            residual_2 = x
            x_norm = self.ff_layer_norm(x)
            ff_output = self.feed_forward(x_norm)
            x = residual_2 + self.dropout(ff_output)
            
        else:
            # Post-norm architecture (original Transformer)
            
            # Step 1: Self-Attention Sublayer
            # Self-Attention → Dropout → Residual → LayerNorm
            attn_output, attention_weights = self.self_attention(
                query=x,
                key=x,
                value=x,
                mask=mask
            )
            x = self.attn_layer_norm(residual_1 + self.dropout(attn_output))
            
            # Step 2: Feed-Forward Sublayer
            # Feed-Forward → Dropout → Residual → LayerNorm
            residual_2 = x
            ff_output = self.feed_forward(x)
            x = self.ff_layer_norm(residual_2 + self.dropout(ff_output))
        
        if return_attention:
            return x, attention_weights
        else:
            return x


class TransformerEncoder(nn.Module):
    """
    Complete Transformer Encoder Stack
    
    The encoder consists of:
    1. Input embedding layer (token → vector)
    2. Positional encoding (position information)
    3. Stack of N encoder layers
    4. Optional final layer normalization
    
    This architecture enables the model to:
    - Process sequences of variable length
    - Capture complex relationships between tokens
    - Build hierarchical representations across layers
    - Handle various NLP tasks (classification, generation, etc.)
    
    Educational Insights:
    - Early layers capture syntactic patterns
    - Middle layers learn semantic relationships  
    - Later layers build task-specific representations
    - Each layer refines and transforms the representations
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_seq_length: int = 5000,
        dropout: float = 0.1,
        use_pre_norm: bool = True,
        use_final_norm: bool = False
    ):
        """
        Initialize the complete Transformer encoder.
        
        Args:
            vocab_size: Size of the vocabulary (number of unique tokens)
            d_model: Model dimension (embedding dimension)
            num_heads: Number of attention heads in each layer
            num_layers: Number of encoder layers
            d_ff: Feed-forward hidden dimension
            max_seq_length: Maximum sequence length for positional encoding
            dropout: Dropout probability
            use_pre_norm: Whether to use pre-norm architecture
            use_final_norm: Whether to add final layer norm after all layers
        """
        super().__init__()
        
        # Store configuration
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.max_seq_length = max_seq_length
        self.use_pre_norm = use_pre_norm
        self.use_final_norm = use_final_norm
        
        # Input embedding layer
        # Converts token IDs to dense vector representations
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        # Adds position information to token embeddings
        self.positional_encoding = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_seq_length=max_seq_length,
            dropout=dropout
        )
        
        # Stack of encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                use_pre_norm=use_pre_norm
            )
            for _ in range(num_layers)
        ])
        
        # Optional final layer normalization
        if use_final_norm:
            self.final_norm = LayerNorm(d_model)
        
        # Dropout for input embeddings
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """
        Initialize model parameters using appropriate strategies.
        
        Proper initialization is crucial for:
        - Training stability
        - Convergence speed
        - Final model performance
        
        Initialization strategies:
        - Embeddings: Normal distribution with std = 1/sqrt(d_model)
        - Other parameters: Xavier/Glorot initialization (handled by individual components)
        """
        # Initialize token embeddings
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=1.0/math.sqrt(self.d_model))
    
    def create_padding_mask(self, input_ids: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
        """
        Create padding mask to ignore padding tokens in attention.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            pad_token_id: Token ID used for padding
            
        Returns:
            Mask tensor of shape (batch_size, 1, 1, seq_len)
            where 1 = attend, 0 = ignore
        
        Why masking is important:
        - Prevents model from attending to meaningless padding tokens
        - Ensures consistent behavior regardless of sequence length
        - Critical for batched training with variable-length sequences
        """
        # Create mask where non-padding tokens are 1, padding tokens are 0
        mask = (input_ids != pad_token_id).float()
        
        # Reshape for attention: (batch_size, seq_len) → (batch_size, 1, 1, seq_len)
        mask = mask.unsqueeze(1).unsqueeze(2)
        
        return mask
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_all_hidden_states: bool = False,
        return_attention_weights: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete Transformer encoder.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask of shape (batch_size, seq_len)
            return_all_hidden_states: Whether to return hidden states from all layers
            return_attention_weights: Whether to return attention weights from all layers
            
        Returns:
            Dictionary containing:
            - 'last_hidden_state': Final layer output (batch_size, seq_len, d_model)
            - 'hidden_states': All layer outputs if requested (num_layers+1, batch_size, seq_len, d_model)
            - 'attention_weights': All attention weights if requested (num_layers, batch_size, num_heads, seq_len, seq_len)
        
        Information Processing Pipeline:
        1. Token IDs → Dense embeddings (vocab_size → d_model)
        2. Add positional information (maintain shape)
        3. Process through encoder layers (build representations)
        4. Return final representations for downstream tasks
        """
        
        batch_size, seq_len = input_ids.shape
        
        # Validate inputs
        if seq_len > self.max_seq_length:
            warnings.warn(
                f"Sequence length ({seq_len}) exceeds maximum ({self.max_seq_length}). "
                f"Consider using a larger max_seq_length or truncating the input."
            )
        
        # Step 1: Convert token IDs to embeddings
        # Shape: (batch_size, seq_len) → (batch_size, seq_len, d_model)
        token_embeddings = self.token_embedding(input_ids)
        
        # Scale embeddings by sqrt(d_model) as in original Transformer paper
        # This scaling helps balance the magnitude of embeddings and positional encodings
        token_embeddings = token_embeddings * math.sqrt(self.d_model)
        
        # Step 2: Add positional encodings
        # The positional encoding expects (seq_len, batch_size, d_model) format
        token_embeddings_transposed = token_embeddings.transpose(0, 1)
        encoded_embeddings = self.positional_encoding(token_embeddings_transposed)
        
        # Transpose back to (batch_size, seq_len, d_model)
        x = encoded_embeddings.transpose(0, 1)
        
        # Apply dropout to input embeddings
        x = self.dropout(x)
        
        # Step 3: Create attention mask if not provided
        if attention_mask is None:
            # Assume all tokens are valid (no padding)
            attention_mask = torch.ones_like(input_ids).float()
        
        # Convert attention mask to the format expected by attention layers
        # Shape: (batch_size, seq_len) → (batch_size, 1, 1, seq_len)
        if attention_mask.dim() == 2:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        # Step 4: Process through encoder layers
        all_hidden_states = []
        all_attention_weights = []
        
        if return_all_hidden_states:
            all_hidden_states.append(x)  # Input embeddings
        
        for layer_idx, layer in enumerate(self.layers):
            if return_attention_weights:
                x, attention_weights = layer(x, mask=attention_mask, return_attention=True)
                all_attention_weights.append(attention_weights)
            else:
                x = layer(x, mask=attention_mask, return_attention=False)
            
            if return_all_hidden_states:
                all_hidden_states.append(x)
        
        # Step 5: Apply final layer normalization if specified
        if self.use_final_norm and hasattr(self, 'final_norm'):
            x = self.final_norm(x)
        
        # Prepare output dictionary
        outputs = {
            'last_hidden_state': x
        }
        
        if return_all_hidden_states:
            outputs['hidden_states'] = torch.stack(all_hidden_states, dim=0)
        
        if return_attention_weights:
            outputs['attention_weights'] = torch.stack(all_attention_weights, dim=0)
        
        return outputs
    
    def get_num_parameters(self) -> Dict[str, int]:
        """
        Get the number of parameters in different components.
        
        Returns:
            Dictionary with parameter counts for analysis
        """
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        embedding_params = self.token_embedding.weight.numel()
        
        # Calculate encoder layer parameters
        layer_params = sum(p.numel() for p in self.layers[0].parameters() if p.requires_grad)
        total_layer_params = layer_params * self.num_layers
        
        other_params = total_params - embedding_params - total_layer_params
        
        return {
            'total': total_params,
            'token_embedding': embedding_params,
            'encoder_layers': total_layer_params,
            'per_layer': layer_params,
            'other': other_params
        }


class TransformerEncoderAnalyzer:
    """
    Analysis and visualization tools for understanding Transformer encoder behavior.
    
    This class provides educational tools to understand:
    - How representations evolve through layers
    - Attention pattern analysis
    - Information flow and processing
    - Performance characteristics
    """
    
    def __init__(self, model: TransformerEncoder):
        """Initialize analyzer with a Transformer encoder model."""
        self.model = model
        self.model.eval()  # Set to evaluation mode
    
    def analyze_representation_evolution(
        self,
        input_ids: torch.Tensor,
        token_strings: List[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Analyze how token representations evolve through encoder layers.
        
        This shows the learning hierarchy:
        - Early layers: syntactic patterns, local dependencies
        - Middle layers: semantic relationships, longer dependencies  
        - Later layers: task-specific representations, global context
        
        Args:
            input_ids: Input token IDs of shape (1, seq_len)
            token_strings: Optional list of token strings for analysis
            
        Returns:
            Dictionary with representation analysis
        """
        with torch.no_grad():
            # Get representations from all layers
            outputs = self.model(
                input_ids=input_ids,
                return_all_hidden_states=True,
                return_attention_weights=True
            )
            
            hidden_states = outputs['hidden_states']  # (num_layers+1, batch_size, seq_len, d_model)
            attention_weights = outputs['attention_weights']  # (num_layers, batch_size, num_heads, seq_len, seq_len)
            
            seq_len = input_ids.shape[1]
            
            # Analyze representation similarity across layers
            layer_similarities = []
            for i in range(len(hidden_states) - 1):
                current_layer = hidden_states[i][0]  # (seq_len, d_model)
                next_layer = hidden_states[i + 1][0]  # (seq_len, d_model)
                
                # Calculate cosine similarity between corresponding positions
                similarities = F.cosine_similarity(current_layer, next_layer, dim=1)
                layer_similarities.append(similarities.mean().item())
            
            # Analyze attention pattern diversity across layers
            attention_diversity = []
            for layer_idx in range(attention_weights.shape[0]):
                layer_attn = attention_weights[layer_idx, 0]  # (num_heads, seq_len, seq_len)
                
                # Calculate entropy of attention distributions (diversity measure)
                entropy_per_head = []
                for head_idx in range(layer_attn.shape[0]):
                    head_attn = layer_attn[head_idx]  # (seq_len, seq_len)
                    
                    # Calculate entropy for each query position
                    entropies = []
                    for pos in range(seq_len):
                        attn_dist = head_attn[pos]  # (seq_len,)
                        # Add small epsilon to prevent log(0)
                        entropy = -torch.sum(attn_dist * torch.log(attn_dist + 1e-10))
                        entropies.append(entropy.item())
                    
                    entropy_per_head.append(np.mean(entropies))
                
                attention_diversity.append(np.mean(entropy_per_head))
            
            return {
                'hidden_states': hidden_states,
                'attention_weights': attention_weights,
                'layer_similarities': layer_similarities,
                'attention_diversity': attention_diversity,
                'representation_norms': [torch.norm(layer[0], dim=1).mean().item() 
                                       for layer in hidden_states]
            }
    
    def demonstrate_attention_patterns(
        self,
        input_ids: torch.Tensor,
        token_strings: List[str],
        layer_to_analyze: int = -1
    ):
        """
        Demonstrate and visualize attention patterns.
        
        Args:
            input_ids: Input token IDs
            token_strings: List of token strings
            layer_to_analyze: Which layer to analyze (-1 for last layer)
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                return_attention_weights=True
            )
            
            attention_weights = outputs['attention_weights']  # (num_layers, batch_size, num_heads, seq_len, seq_len)
            
            # Select layer to analyze
            if layer_to_analyze == -1:
                layer_to_analyze = attention_weights.shape[0] - 1
            
            layer_attention = attention_weights[layer_to_analyze, 0]  # (num_heads, seq_len, seq_len)
            
            print(f"Attention Patterns Analysis - Layer {layer_to_analyze + 1}")
            print("=" * 60)
            
            # Analyze different heads
            for head_idx in range(min(4, layer_attention.shape[0])):  # Show first 4 heads
                head_attn = layer_attention[head_idx].numpy()
                
                print(f"\nAttention Head {head_idx + 1}:")
                print("-" * 30)
                
                # Print attention matrix with token labels
                print("Query\\Key", end="\t")
                for token in token_strings:
                    print(f"{token[:8]:>8}", end="\t")
                print()
                
                for i, query_token in enumerate(token_strings):
                    print(f"{query_token[:8]:>8}", end="\t")
                    for j in range(len(token_strings)):
                        print(f"{head_attn[i, j]:.3f}", end="\t\t")
                    print()
                
                # Find most attended positions for each query
                max_attentions = []
                for i in range(len(token_strings)):
                    max_idx = head_attn[i].argmax()
                    max_val = head_attn[i, max_idx]
                    max_attentions.append((i, max_idx, max_val))
                
                print(f"\nMost attended positions in Head {head_idx + 1}:")
                for query_idx, key_idx, attention_val in max_attentions:
                    print(f"  '{token_strings[query_idx]}' → '{token_strings[key_idx]}' ({attention_val:.3f})")
    
    def memory_and_compute_analysis(self, seq_len: int, batch_size: int = 1) -> Dict[str, Union[int, float]]:
        """
        Analyze memory usage and computational complexity.
        
        Args:
            seq_len: Sequence length
            batch_size: Batch size
            
        Returns:
            Dictionary with memory and compute statistics
        """
        d_model = self.model.d_model
        num_heads = self.model.num_heads
        d_ff = self.model.d_ff
        num_layers = self.model.num_layers
        
        # Parameter count
        param_counts = self.model.get_num_parameters()
        
        # Memory usage (in elements, multiply by 4 for bytes in float32)
        activation_memory = batch_size * seq_len * d_model  # Base activation size
        attention_memory = batch_size * num_heads * seq_len * seq_len  # Attention weights
        ff_memory = batch_size * seq_len * d_ff  # Feed-forward intermediate
        
        total_activation_memory = (
            activation_memory * (2 * num_layers + 1) +  # Input + output of each layer
            attention_memory * num_layers +  # Attention weights for each layer
            ff_memory * num_layers  # Feed-forward intermediate for each layer
        )
        
        # Computational complexity (FLOPs)
        attention_flops = 2 * batch_size * num_layers * num_heads * seq_len * seq_len * (d_model // num_heads)
        ff_flops = 2 * batch_size * num_layers * seq_len * d_model * d_ff
        total_flops = attention_flops + ff_flops
        
        return {
            'parameters': param_counts['total'],
            'embedding_params': param_counts['token_embedding'],
            'encoder_params': param_counts['encoder_layers'],
            'memory_mb': (total_activation_memory * 4) / (1024 * 1024),  # Convert to MB
            'attention_memory_mb': (attention_memory * num_layers * 4) / (1024 * 1024),
            'total_flops': total_flops,
            'attention_flops': attention_flops,
            'ff_flops': ff_flops,
            'flops_per_token': total_flops / (batch_size * seq_len)
        }


def educational_demonstration():
    """
    Comprehensive educational demonstration of the complete Transformer encoder.
    
    This demonstration shows:
    1. How to create and configure the encoder
    2. Input processing pipeline
    3. Information flow through layers
    4. Attention pattern analysis
    5. Performance characteristics
    """
    
    print("COMPLETE TRANSFORMER ENCODER EDUCATIONAL DEMONSTRATION")
    print("=" * 80)
    
    # Configuration parameters
    vocab_size = 10000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    seq_len = 20
    
    print(f"\nModel Configuration:")
    print(f"- Vocabulary size: {vocab_size:,}")
    print(f"- Model dimension: {d_model}")
    print(f"- Number of attention heads: {num_heads}")
    print(f"- Number of encoder layers: {num_layers}")
    print(f"- Feed-forward dimension: {d_ff}")
    print(f"- Maximum sequence length: 5000")
    
    # Create the complete encoder
    encoder = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=0.1,
        use_pre_norm=True
    )
    
    # Parameter analysis
    param_counts = encoder.get_num_parameters()
    print(f"\nParameter Analysis:")
    print(f"- Total parameters: {param_counts['total']:,}")
    print(f"- Token embedding: {param_counts['token_embedding']:,} ({param_counts['token_embedding']/param_counts['total']*100:.1f}%)")
    print(f"- Encoder layers: {param_counts['encoder_layers']:,} ({param_counts['encoder_layers']/param_counts['total']*100:.1f}%)")
    print(f"- Per encoder layer: {param_counts['per_layer']:,}")
    
    # Create sample input
    batch_size = 2
    torch.manual_seed(42)  # For reproducible results
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))  # Random token IDs (avoid 0 = padding)
    
    print(f"\nSample Input:")
    print(f"- Input shape: {input_ids.shape}")
    print(f"- Batch size: {batch_size}")
    print(f"- Sequence length: {seq_len}")
    print(f"- Sample tokens: {input_ids[0, :10].tolist()}")
    
    # Forward pass demonstration
    print(f"\n" + "="*60)
    print("FORWARD PASS DEMONSTRATION")
    print("="*60)
    
    encoder.eval()
    with torch.no_grad():
        # Basic forward pass
        outputs = encoder(input_ids)
        last_hidden_state = outputs['last_hidden_state']
        
        print(f"Output shape: {last_hidden_state.shape}")
        print(f"✓ Shape preserved: (batch_size, seq_len, d_model)")
        
        # Forward pass with all hidden states
        outputs_detailed = encoder(
            input_ids,
            return_all_hidden_states=True,
            return_attention_weights=True
        )
        
        hidden_states = outputs_detailed['hidden_states']
        attention_weights = outputs_detailed['attention_weights']
        
        print(f"\nDetailed Analysis:")
        print(f"- Hidden states shape: {hidden_states.shape}")
        print(f"  → {hidden_states.shape[0]} layers (including input embeddings)")
        print(f"- Attention weights shape: {attention_weights.shape}")
        print(f"  → {attention_weights.shape[0]} layers × {attention_weights.shape[2]} heads")
    
    # Representation evolution analysis
    print(f"\n" + "="*60)
    print("REPRESENTATION EVOLUTION ANALYSIS")
    print("="*60)
    
    # Create analyzer
    analyzer = TransformerEncoderAnalyzer(encoder)
    
    # Analyze single sequence
    single_input = input_ids[0:1]  # First sequence only
    evolution_analysis = analyzer.analyze_representation_evolution(single_input)
    
    layer_similarities = evolution_analysis['layer_similarities']
    attention_diversity = evolution_analysis['attention_diversity']
    representation_norms = evolution_analysis['representation_norms']
    
    print(f"Layer-to-layer similarity (cosine):")
    for i, similarity in enumerate(layer_similarities):
        print(f"  Layer {i} → Layer {i+1}: {similarity:.4f}")
    
    print(f"\nAttention diversity (entropy) by layer:")
    for i, diversity in enumerate(attention_diversity):
        print(f"  Layer {i+1}: {diversity:.4f}")
    
    print(f"\nRepresentation norms by layer:")
    for i, norm in enumerate(representation_norms):
        layer_name = "Input" if i == 0 else f"Layer {i}"
        print(f"  {layer_name}: {norm:.4f}")
    
    # Performance analysis
    print(f"\n" + "="*60)
    print("PERFORMANCE ANALYSIS")
    print("="*60)
    
    perf_analysis = analyzer.memory_and_compute_analysis(seq_len=seq_len, batch_size=batch_size)
    
    print(f"Memory Usage:")
    print(f"- Total activation memory: {perf_analysis['memory_mb']:.1f} MB")
    print(f"- Attention memory: {perf_analysis['attention_memory_mb']:.1f} MB")
    
    print(f"\nComputational Complexity:")
    print(f"- Total FLOPs: {perf_analysis['total_flops']:,}")
    print(f"- Attention FLOPs: {perf_analysis['attention_flops']:,} ({perf_analysis['attention_flops']/perf_analysis['total_flops']*100:.1f}%)")
    print(f"- Feed-forward FLOPs: {perf_analysis['ff_flops']:,} ({perf_analysis['ff_flops']/perf_analysis['total_flops']*100:.1f}%)")
    print(f"- FLOPs per token: {perf_analysis['flops_per_token']:,.0f}")
    
    # Attention pattern demonstration
    print(f"\n" + "="*60)
    print("ATTENTION PATTERN DEMONSTRATION")
    print("="*60)
    
    # Create a more meaningful example with actual words (simulated)
    sample_tokens = ["The", "cat", "sat", "on", "the", "mat", "and", "looked", "around", "carefully"]
    sample_input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])  # Simulated token IDs
    
    print(f"Sample sentence: {' '.join(sample_tokens)}")
    
    # Analyze attention patterns for the last layer
    analyzer.demonstrate_attention_patterns(
        input_ids=sample_input_ids,
        token_strings=sample_tokens,
        layer_to_analyze=-1  # Last layer
    )
    
    # Architecture insights
    print(f"\n" + "="*60)
    print("ARCHITECTURE INSIGHTS")
    print("="*60)
    
    print("""
Key Components Integration:

1. TOKEN EMBEDDING LAYER:
   - Converts discrete tokens to continuous vectors
   - Learnable lookup table (vocab_size × d_model)
   - Scaled by √d_model to balance with positional encodings

2. POSITIONAL ENCODING:
   - Adds position information using sinusoidal functions
   - No additional parameters (deterministic)
   - Enables model to understand token order

3. ENCODER LAYER STACK:
   - Each layer refines representations
   - Self-attention captures relationships
   - Feed-forward processes individual positions
   - Residual connections enable deep training

4. NORMALIZATION & RESIDUALS:
   - Layer normalization stabilizes training
   - Residual connections prevent vanishing gradients
   - Pre-norm architecture improves stability

Information Flow Hierarchy:
- Layer 1-2: Syntactic patterns, local dependencies
- Layer 3-4: Semantic relationships, phrase-level understanding  
- Layer 5-6: Global context, task-specific representations
    """)
    
    # Usage recommendations
    print(f"\n" + "="*60)
    print("PRACTICAL USAGE RECOMMENDATIONS")
    print("="*60)
    
    print("""
Configuration Guidelines:

1. SMALL MODELS (Mobile/Edge):
   - d_model: 256-384
   - num_layers: 6-8
   - num_heads: 4-6
   - d_ff: 1024-1536

2. MEDIUM MODELS (Research/Development):
   - d_model: 512-768
   - num_layers: 8-12
   - num_heads: 8-12
   - d_ff: 2048-3072

3. LARGE MODELS (Production/SOTA):
   - d_model: 768-1024
   - num_layers: 12-24
   - num_heads: 12-16
   - d_ff: 3072-4096

Memory Optimization:
- Use gradient checkpointing for large models
- Consider mixed precision training (FP16)
- Batch size affects memory quadratically for long sequences
- Attention memory scales as O(seq_len²)

Training Tips:
- Use learning rate warmup
- Apply dropout for regularization
- Pre-norm architecture is more stable
- Monitor gradient norms during training
    """)


if __name__ == "__main__":
    # Run the comprehensive educational demonstration
    educational_demonstration()
    
    print(f"\n" + "="*80)
    print("SUMMARY: TRANSFORMER ENCODER COMPLETE")
    print("="*80)
    
    print("""
🎯 WHAT WE'VE BUILT:

✅ Multi-Head Attention: Captures relationships between positions
✅ Positional Encoding: Provides sequence order information
✅ Feed-Forward Networks: Processes information at each position
✅ Layer Normalization: Stabilizes training dynamics
✅ Residual Connections: Enables deep network training
✅ Complete Encoder: Integrates all components seamlessly

🔬 KEY EDUCATIONAL INSIGHTS:

1. MODULAR DESIGN: Each component has a specific purpose and can be understood independently
2. INFORMATION FLOW: Representations become more sophisticated through encoder layers
3. ATTENTION PATTERNS: Different heads learn different types of relationships
4. SCALING PROPERTIES: Memory and compute scale predictably with model size
5. ARCHITECTURAL CHOICES: Pre-norm vs post-norm, layer depth, head count all matter

🚀 NEXT STEPS:

- Implement decoder for sequence generation tasks
- Add positional encodings variants (RoPE, ALiBi)
- Experiment with different attention mechanisms
- Build task-specific heads (classification, generation)
- Explore model compression and optimization techniques

This complete encoder implementation provides a solid foundation for:
- Text classification and understanding
- Feature extraction for downstream tasks
- Building more complex architectures (BERT, GPT, T5)
- Research and experimentation with Transformer variants
    """)