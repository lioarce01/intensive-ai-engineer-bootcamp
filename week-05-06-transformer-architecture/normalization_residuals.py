"""
Comprehensive Layer Normalization and Residual Connections Implementation

This module provides educational implementations of two crucial Transformer components:
1. Layer Normalization - normalizes across feature dimension for each sample
2. Residual Connections - skip connections that enable training of very deep networks

Key Educational Concepts:
- Why Layer Norm over Batch Norm in Transformers
- Mathematical properties of normalization
- Gradient flow and training stability
- Pre-norm vs Post-norm architectures
- Deep network training challenges and solutions

Author: AI Intensive Bootcamp
Week: 5-6 (Transformer Architecture)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any
import matplotlib.pyplot as plt
import numpy as np


class LayerNorm(nn.Module):
    """
    Layer Normalization Implementation
    
    Mathematical Formula:
    y = γ * (x - μ) / σ + β
    
    Where:
    - μ = mean across feature dimension
    - σ = standard deviation across feature dimension  
    - γ = learnable scale parameter (initialized to 1)
    - β = learnable shift parameter (initialized to 0)
    
    Key Properties:
    1. Normalizes across features (not batch dimension)
    2. Each sample normalized independently
    3. Invariant to batch size
    4. Reduces internal covariate shift
    5. Improves gradient flow
    
    Why Layer Norm over Batch Norm in Transformers:
    - Batch Norm depends on batch statistics (problematic for variable sequence lengths)
    - Layer Norm works with batch size = 1 (crucial for inference)
    - More stable for sequential data
    - Better for attention mechanisms
    """
    
    def __init__(self, 
                 normalized_shape: int, 
                 eps: float = 1e-5,
                 elementwise_affine: bool = True):
        """
        Initialize Layer Normalization
        
        Args:
            normalized_shape: Size of feature dimension to normalize over
            eps: Small value for numerical stability
            elementwise_affine: Whether to include learnable parameters
        """
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            # Learnable parameters
            self.weight = nn.Parameter(torch.ones(normalized_shape))   # γ (gamma)
            self.bias = nn.Parameter(torch.zeros(normalized_shape))    # β (beta)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply Layer Normalization
        
        Args:
            input_tensor: Input of shape (..., normalized_shape)
            
        Returns:
            Normalized tensor of same shape
        """
        # Calculate mean and variance across the last dimension (features)
        # For input shape (batch_size, seq_len, d_model), normalizes across d_model
        mean = input_tensor.mean(dim=-1, keepdim=True)
        variance = input_tensor.var(dim=-1, keepdim=True, unbiased=False)
        
        # Normalize: (x - μ) / σ
        normalized = (input_tensor - mean) / torch.sqrt(variance + self.eps)
        
        # Apply learnable affine transformation: γ * normalized + β
        if self.elementwise_affine:
            normalized = normalized * self.weight + self.bias
            
        return normalized
    
    def extra_repr(self) -> str:
        return f'normalized_shape={self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'


class ResidualConnection(nn.Module):
    """
    Residual Connection Implementation
    
    Mathematical Formula:
    output = x + F(x)
    
    Where:
    - x = input (identity/skip connection)
    - F(x) = transformation applied to input
    - + = element-wise addition
    
    Key Benefits:
    1. Enables training of very deep networks (100+ layers)
    2. Addresses vanishing gradient problem
    3. Allows gradient to flow directly through skip connections
    4. Provides multiple paths for information flow
    5. Makes optimization landscape smoother
    
    Transformer Usage:
    - Around multi-head attention: x + MultiHeadAttention(LayerNorm(x))
    - Around feed-forward: x + FeedForward(LayerNorm(x))
    """
    
    def __init__(self, dropout_rate: float = 0.1):
        """
        Initialize Residual Connection
        
        Args:
            dropout_rate: Dropout probability for regularization
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, 
                x: torch.Tensor, 
                sublayer_output: torch.Tensor) -> torch.Tensor:
        """
        Apply residual connection: x + sublayer_output
        
        Args:
            x: Original input (skip connection)
            sublayer_output: Output from sublayer transformation
            
        Returns:
            Sum of input and transformed output
        """
        # Apply dropout to sublayer output for regularization
        return x + self.dropout(sublayer_output)


class PreNormResidualBlock(nn.Module):
    """
    Pre-Norm Residual Block (Modern Transformer Architecture)
    
    Architecture: x + F(LayerNorm(x))
    
    Advantages:
    - More stable training
    - Better gradient flow
    - Less prone to optimization difficulties
    - Used in GPT-2, T5, and most modern Transformers
    """
    
    def __init__(self, 
                 d_model: int, 
                 dropout_rate: float = 0.1):
        super().__init__()
        self.layer_norm = LayerNorm(d_model)
        self.residual = ResidualConnection(dropout_rate)
    
    def forward(self, 
                x: torch.Tensor, 
                sublayer_fn) -> torch.Tensor:
        """
        Apply pre-norm residual connection
        
        Args:
            x: Input tensor
            sublayer_fn: Function to apply (attention or feedforward)
            
        Returns:
            Output after pre-norm residual connection
        """
        # Pre-norm: normalize first, then apply transformation
        normalized_x = self.layer_norm(x)
        sublayer_output = sublayer_fn(normalized_x)
        return self.residual(x, sublayer_output)


class PostNormResidualBlock(nn.Module):
    """
    Post-Norm Residual Block (Original Transformer Architecture)
    
    Architecture: LayerNorm(x + F(x))
    
    Characteristics:
    - Original Transformer design
    - Can be harder to train for very deep networks
    - Requires careful initialization and learning rates
    """
    
    def __init__(self, 
                 d_model: int, 
                 dropout_rate: float = 0.1):
        super().__init__()
        self.layer_norm = LayerNorm(d_model)
        self.residual = ResidualConnection(dropout_rate)
    
    def forward(self, 
                x: torch.Tensor, 
                sublayer_fn) -> torch.Tensor:
        """
        Apply post-norm residual connection
        
        Args:
            x: Input tensor
            sublayer_fn: Function to apply
            
        Returns:
            Output after post-norm residual connection
        """
        # Post-norm: apply transformation, add residual, then normalize
        sublayer_output = sublayer_fn(x)
        residual_output = self.residual(x, sublayer_output)
        return self.layer_norm(residual_output)


class BatchNormComparison(nn.Module):
    """
    Batch Normalization for Educational Comparison
    
    Mathematical Formula:
    y = γ * (x - μ_batch) / σ_batch + β
    
    Key Differences from Layer Norm:
    - Normalizes across batch dimension (not features)
    - Depends on batch statistics
    - Different behavior during training vs inference
    - Problematic for variable sequence lengths
    - Not suitable for Transformers
    """
    
    def __init__(self, num_features: int):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(num_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply batch normalization (for comparison only)"""
        # Reshape for BatchNorm1d: (batch, features) or (batch, features, length)
        original_shape = x.shape
        if x.dim() == 3:  # (batch, seq_len, features)
            x = x.transpose(1, 2)  # (batch, features, seq_len)
        elif x.dim() == 2:  # (batch, features)
            pass
        else:
            raise ValueError(f"Unsupported input shape: {original_shape}")
        
        normalized = self.batch_norm(x)
        
        # Restore original shape
        if len(original_shape) == 3:
            normalized = normalized.transpose(1, 2)
        
        return normalized


class NormalizationDemo:
    """
    Educational demonstrations of normalization effects
    """
    
    @staticmethod
    def demonstrate_normalization_effects():
        """
        Show the effects of different normalization techniques
        """
        # Create sample data with different scales
        batch_size, seq_len, d_model = 4, 10, 512
        
        # Create data with varying scales across features
        data = torch.randn(batch_size, seq_len, d_model)
        data[:, :, :256] *= 10  # First half has larger scale
        data[:, :, 256:] *= 0.1  # Second half has smaller scale
        
        # Initialize normalization layers
        layer_norm = LayerNorm(d_model)
        
        # Apply normalizations
        ln_output = layer_norm(data)
        
        # Calculate statistics
        stats = {
            'original': {
                'mean': data.mean(dim=-1).mean(),
                'std': data.std(dim=-1).mean(),
                'feature_mean': data.mean(dim=(0, 1)),
                'feature_std': data.std(dim=(0, 1))
            },
            'layer_norm': {
                'mean': ln_output.mean(dim=-1).mean(),
                'std': ln_output.std(dim=-1).mean(),
                'feature_mean': ln_output.mean(dim=(0, 1)),
                'feature_std': ln_output.std(dim=(0, 1))
            }
        }
        
        return data, ln_output, stats
    
    @staticmethod
    def demonstrate_gradient_flow():
        """
        Show how residual connections improve gradient flow
        """
        d_model = 512
        seq_len = 100
        batch_size = 32
        
        # Create deep network without residuals
        class DeepNetworkWithoutResiduals(nn.Module):
            def __init__(self, depth=10):
                super().__init__()
                self.layers = nn.ModuleList([
                    nn.Linear(d_model, d_model) for _ in range(depth)
                ])
                self.activation = nn.ReLU()
            
            def forward(self, x):
                for layer in self.layers:
                    x = self.activation(layer(x))
                return x
        
        # Create deep network with residuals
        class DeepNetworkWithResiduals(nn.Module):
            def __init__(self, depth=10):
                super().__init__()
                self.layers = nn.ModuleList([
                    nn.Linear(d_model, d_model) for _ in range(depth)
                ])
                self.activation = nn.ReLU()
                self.residual = ResidualConnection()
            
            def forward(self, x):
                for layer in self.layers:
                    x = self.residual(x, self.activation(layer(x)))
                return x
        
        # Test gradient flow
        input_data = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        target = torch.randn(batch_size, seq_len, d_model)
        
        # Without residuals
        net_without = DeepNetworkWithoutResiduals()
        output_without = net_without(input_data.clone())
        loss_without = F.mse_loss(output_without, target)
        loss_without.backward()
        grad_without = input_data.grad.abs().mean() if input_data.grad is not None else 0
        
        # Clear gradients
        input_data.grad = None
        
        # With residuals
        net_with = DeepNetworkWithResiduals()
        output_with = net_with(input_data.clone())
        loss_with = F.mse_loss(output_with, target)
        loss_with.backward()
        grad_with = input_data.grad.abs().mean() if input_data.grad is not None else 0
        
        return {
            'gradient_without_residuals': grad_without,
            'gradient_with_residuals': grad_with,
            'gradient_ratio': grad_with / grad_without if grad_without > 0 else float('inf')
        }


class TransformerBlock(nn.Module):
    """
    Complete Transformer Block combining normalization and residuals
    
    Demonstrates both Pre-Norm and Post-Norm architectures
    """
    
    def __init__(self, 
                 d_model: int, 
                 n_heads: int = 8,
                 d_ff: int = 2048,
                 dropout_rate: float = 0.1,
                 use_pre_norm: bool = True):
        """
        Initialize Transformer Block
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout_rate: Dropout probability
            use_pre_norm: Whether to use pre-norm (True) or post-norm (False)
        """
        super().__init__()
        self.d_model = d_model
        self.use_pre_norm = use_pre_norm
        
        # Multi-head attention (simplified)
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout_rate, batch_first=True
        )
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout_rate)
        )
        
        # Normalization and residual blocks
        if use_pre_norm:
            self.attn_block = PreNormResidualBlock(d_model, dropout_rate)
            self.ff_block = PreNormResidualBlock(d_model, dropout_rate)
        else:
            self.attn_block = PostNormResidualBlock(d_model, dropout_rate)
            self.ff_block = PostNormResidualBlock(d_model, dropout_rate)
    
    def forward(self, 
                x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through Transformer block
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor of same shape
        """
        # Self-attention sublayer
        def attn_fn(x_norm):
            # Self-attention with residual and normalization
            attn_output, _ = self.attention(x_norm, x_norm, x_norm, attn_mask=mask)
            return attn_output
        
        # Apply attention with residual connection
        x = self.attn_block(x, attn_fn)
        
        # Feed-forward sublayer  
        def ff_fn(x_norm):
            return self.feed_forward(x_norm)
        
        # Apply feed-forward with residual connection
        x = self.ff_block(x, ff_fn)
        
        return x


def educational_examples():
    """
    Educational examples demonstrating key concepts
    """
    
    # Example 1: Layer Norm vs Batch Norm behavior
    batch_size, seq_len, d_model = 2, 5, 4
    
    # Create sample data
    x = torch.tensor([
        [[1.0, 2.0, 3.0, 4.0],   # Sample 1, Token 1
         [2.0, 3.0, 4.0, 5.0],   # Sample 1, Token 2
         [3.0, 4.0, 5.0, 6.0],   # Sample 1, Token 3
         [4.0, 5.0, 6.0, 7.0],   # Sample 1, Token 4
         [5.0, 6.0, 7.0, 8.0]],  # Sample 1, Token 5
        
        [[10.0, 20.0, 30.0, 40.0],  # Sample 2, Token 1 (different scale)
         [20.0, 30.0, 40.0, 50.0],  # Sample 2, Token 2
         [30.0, 40.0, 50.0, 60.0],  # Sample 2, Token 3
         [40.0, 50.0, 60.0, 70.0],  # Sample 2, Token 4
         [50.0, 60.0, 70.0, 80.0]]  # Sample 2, Token 5
    ])
    
    layer_norm = LayerNorm(d_model)
    ln_output = layer_norm(x)
    
    # Example 2: Residual connection benefits
    identity_input = torch.randn(batch_size, seq_len, d_model)
    
    # Simulate a transformation that might cause vanishing gradients
    noisy_transform = identity_input * 0.01 + torch.randn_like(identity_input) * 0.001
    
    # Without residual: just the noisy transformation
    without_residual = noisy_transform
    
    # With residual: original + transformation
    residual_conn = ResidualConnection()
    with_residual = residual_conn(identity_input, noisy_transform)
    
    # Example 3: Pre-norm vs Post-norm comparison
    transformer_input = torch.randn(batch_size, seq_len, d_model)
    
    # Pre-norm block
    pre_norm_block = PreNormResidualBlock(d_model)
    
    # Post-norm block  
    post_norm_block = PostNormResidualBlock(d_model)
    
    # Simple transformation function
    def simple_transform(x):
        return torch.relu(torch.linear(x, torch.randn(d_model, d_model)))
    
    # This is a conceptual example - in practice you'd use actual sublayers
    
    return {
        'layer_norm_example': {
            'input': x,
            'output': ln_output,
            'input_means': x.mean(dim=-1),
            'output_means': ln_output.mean(dim=-1),
            'input_stds': x.std(dim=-1),
            'output_stds': ln_output.std(dim=-1)
        },
        'residual_example': {
            'input': identity_input,
            'without_residual': without_residual,
            'with_residual': with_residual,
            'signal_preservation': torch.norm(with_residual - identity_input) / torch.norm(without_residual - identity_input)
        }
    }


class ArchitectureAnalysis:
    """
    Analysis tools for understanding normalization and residual effects
    """
    
    @staticmethod
    def analyze_gradient_norms(model: nn.Module, 
                              input_data: torch.Tensor, 
                              target: torch.Tensor) -> Dict[str, float]:
        """
        Analyze gradient norms throughout the model
        """
        # Forward pass
        output = model(input_data)
        loss = F.mse_loss(output, target)
        
        # Backward pass
        loss.backward()
        
        # Collect gradient norms
        gradient_norms = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradient_norms[name] = param.grad.norm().item()
        
        return gradient_norms
    
    @staticmethod
    def compare_architectures():
        """
        Compare pre-norm vs post-norm architectures
        """
        d_model = 512
        seq_len = 128
        batch_size = 16
        
        # Create models
        pre_norm_transformer = TransformerBlock(d_model, use_pre_norm=True)
        post_norm_transformer = TransformerBlock(d_model, use_pre_norm=False)
        
        # Sample data
        input_data = torch.randn(batch_size, seq_len, d_model)
        target = torch.randn(batch_size, seq_len, d_model)
        
        # Analyze both architectures
        pre_norm_grads = ArchitectureAnalysis.analyze_gradient_norms(
            pre_norm_transformer, input_data.clone(), target
        )
        
        # Clear gradients
        post_norm_transformer.zero_grad()
        
        post_norm_grads = ArchitectureAnalysis.analyze_gradient_norms(
            post_norm_transformer, input_data.clone(), target
        )
        
        return {
            'pre_norm_gradients': pre_norm_grads,
            'post_norm_gradients': post_norm_grads
        }


# Mathematical Properties Documentation
"""
MATHEMATICAL PROPERTIES OF LAYER NORMALIZATION:

1. Invariance Properties:
   - Scale invariant: LayerNorm(cx) = LayerNorm(x) for c > 0
   - Shift invariant after normalization: LayerNorm(x + c) ≠ LayerNorm(x) but similar
   
2. Gradient Properties:
   - Gradient of LayerNorm w.r.t. input has specific structure
   - Helps maintain gradient scale throughout network
   
3. Statistical Properties:
   - Output has zero mean and unit variance across features
   - Reduces internal covariate shift
   
4. Computational Complexity:
   - O(d) for feature dimension d
   - Very efficient compared to other normalization techniques

RESIDUAL CONNECTIONS MATHEMATICAL ANALYSIS:

1. Gradient Flow:
   - Direct path: ∂L/∂x = ∂L/∂output (identity gradient)
   - Transformed path: ∂L/∂x = ∂L/∂output * ∂F(x)/∂x
   - Combined: ∂L/∂x = ∂L/∂output * (1 + ∂F(x)/∂x)
   
2. Information Flow:
   - Multiple paths for information propagation
   - Shortest path is always length 1 (identity)
   - Enables training of networks with 1000+ layers
   
3. Optimization Landscape:
   - Smooths loss landscape
   - Reduces optimization difficulties
   - Makes network less prone to vanishing gradients

WHY TRANSFORMERS USE THESE COMPONENTS:

1. Layer Norm Benefits:
   - Works with variable sequence lengths
   - Independent of batch size (crucial for inference)
   - Stable training dynamics
   - Better for attention mechanisms
   
2. Residual Connection Benefits:
   - Enables very deep architectures (Transformer-XL, etc.)
   - Stable gradient flow
   - Easier optimization
   - Information highway through the network

3. Combined Effects:
   - Pre-norm: More stable, modern choice
   - Post-norm: Original design, requires careful tuning
   - Both enable training of models with 100+ layers
"""

# Key Takeaways for Transformer Architecture:
"""
DESIGN PRINCIPLES:

1. Normalization Strategy:
   - Use Layer Norm, not Batch Norm
   - Normalize across feature dimension
   - Apply before or after transformation (pre/post-norm)

2. Residual Strategy:
   - Always include skip connections
   - Apply dropout to sublayer outputs
   - Maintain same dimensions for addition

3. Architecture Choices:
   - Pre-norm: Better for very deep networks
   - Post-norm: Original design, still used
   - Consider model depth when choosing

4. Implementation Details:
   - eps=1e-5 for numerical stability
   - Initialize gamma=1, beta=0
   - Use dropout for regularization
"""

if __name__ == "__main__":
    # Run educational examples
    examples = educational_examples()
    
    # Demonstrate normalization effects
    demo = NormalizationDemo()
    _, _, norm_stats = demo.demonstrate_normalization_effects()
    
    # Demonstrate gradient flow
    grad_stats = demo.demonstrate_gradient_flow()
    
    # Architecture comparison
    arch_comparison = ArchitectureAnalysis.compare_architectures()