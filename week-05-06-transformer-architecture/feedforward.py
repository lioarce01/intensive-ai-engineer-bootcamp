"""
Transformer Feed-Forward Network (FFN) Implementation
===================================================

This module implements the position-wise feed-forward network used in the Transformer architecture.
The FFN is a crucial component that provides non-linear transformations and enables the model
to learn complex patterns beyond what attention mechanisms alone can capture.

Educational Focus:
- Understanding why Transformers need feed-forward networks
- How the two-layer structure works (expand then compress)
- Different activation functions and their properties
- The role of dropout in preventing overfitting
- How FFN complements the attention mechanism

Author: AI Bootcamp - Week 5-6
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Callable, Literal
import matplotlib.pyplot as plt
import numpy as np


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (FFN) for Transformer architecture.
    
    The FFN applies the same linear transformations to each position separately
    and identically. This is why it's called "position-wise" - each position
    in the sequence gets the same transformation applied to it.
    
    Architecture:
    FFN(x) = max(0, xW1 + b1)W2 + b2
    
    Where:
    - W1: First linear layer (d_model -> d_ff)
    - W2: Second linear layer (d_ff -> d_model)
    - max(0, x) is the ReLU activation (or other activation functions)
    
    Educational Notes:
    ==================
    
    Why do Transformers need FFN?
    -----------------------------
    1. **Non-linearity**: Attention is essentially a linear operation. Without FFN,
       the entire Transformer would be linear, severely limiting its expressiveness.
    
    2. **Feature interaction**: While attention captures relationships between positions,
       FFN allows for complex feature interactions within each position.
    
    3. **Representation transformation**: FFN transforms the attention output into
       a different representation space, enabling the model to learn complex patterns.
    
    4. **Increased capacity**: The expansion to d_ff (usually 4x d_model) provides
       additional parameters and computational capacity.
    
    Two-layer structure (Expand then Compress):
    ------------------------------------------
    1. **Expansion**: First layer expands from d_model to d_ff (typically 4x larger)
       - Creates a higher-dimensional space for complex transformations
       - Allows the model to consider many more feature combinations
    
    2. **Compression**: Second layer compresses back to d_model
       - Ensures output dimension matches input for residual connections
       - Forces the model to learn meaningful compressed representations
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: Literal['relu', 'gelu', 'swish'] = 'relu'
    ):
        """
        Initialize the Position-wise Feed-Forward Network.
        
        Args:
            d_model: Model dimension (input/output dimension)
            d_ff: Feed-forward dimension (hidden layer dimension, typically 4 * d_model)
            dropout: Dropout probability for regularization
            activation: Activation function to use ('relu', 'gelu', 'swish')
        """
        super(PositionwiseFeedForward, self).__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_prob = dropout
        
        # First linear layer: expand from d_model to d_ff
        self.linear1 = nn.Linear(d_model, d_ff)
        
        # Second linear layer: compress from d_ff back to d_model
        self.linear2 = nn.Linear(d_ff, d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        self.activation_name = activation
        self.activation = self._get_activation_function(activation)
        
        # Initialize weights using Xavier/Glorot initialization
        self._initialize_weights()
    
    def _get_activation_function(self, activation: str) -> Callable:
        """
        Get the specified activation function.
        
        Activation Functions Explained:
        ==============================
        
        1. **ReLU (Rectified Linear Unit)**:
           - f(x) = max(0, x)
           - Pros: Simple, computationally efficient, helps with vanishing gradients
           - Cons: Can cause "dying ReLU" problem (neurons always output 0)
           - Best for: General use, especially when computational efficiency matters
        
        2. **GELU (Gaussian Error Linear Unit)**:
           - f(x) = x * Φ(x) where Φ is the CDF of standard normal distribution
           - Approximation: f(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
           - Pros: Smooth, no "dying neuron" problem, works well with Transformers
           - Cons: More computationally expensive than ReLU
           - Best for: Transformer models (used in BERT, GPT)
        
        3. **Swish (SiLU - Sigmoid Linear Unit)**:
           - f(x) = x * σ(x) where σ is the sigmoid function
           - Pros: Smooth, self-gated, unbounded above, bounded below
           - Cons: More expensive than ReLU
           - Best for: Deep networks, when you want smooth gradients
        """
        activation_functions = {
            'relu': F.relu,
            'gelu': F.gelu,
            'swish': lambda x: x * torch.sigmoid(x)  # Swish/SiLU
        }
        
        if activation not in activation_functions:
            raise ValueError(f"Unsupported activation: {activation}. "
                           f"Choose from {list(activation_functions.keys())}")
        
        return activation_functions[activation]
    
    def _initialize_weights(self):
        """
        Initialize weights using Xavier/Glorot initialization.
        
        Why proper initialization matters:
        ================================
        - Prevents vanishing/exploding gradients
        - Helps with training stability
        - Ensures good initial gradient flow
        """
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.constant_(self.linear2.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feed-forward network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        
        The forward pass implements:
        1. Linear transformation to higher dimension (expand)
        2. Non-linear activation
        3. Dropout for regularization
        4. Linear transformation back to original dimension (compress)
        """
        # Step 1: Expand to higher dimension and apply activation
        # Shape: (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff)
        hidden = self.activation(self.linear1(x))
        
        # Step 2: Apply dropout for regularization
        hidden = self.dropout(hidden)
        
        # Step 3: Compress back to original dimension
        # Shape: (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        output = self.linear2(hidden)
        
        return output
    
    def get_activation_stats(self, x: torch.Tensor) -> dict:
        """
        Get statistics about activations for analysis.
        
        This method helps understand how different activation functions
        affect the internal representations.
        """
        with torch.no_grad():
            # Forward through first layer
            linear_output = self.linear1(x)
            activated_output = self.activation(linear_output)
            
            stats = {
                'linear_mean': linear_output.mean().item(),
                'linear_std': linear_output.std().item(),
                'activated_mean': activated_output.mean().item(),
                'activated_std': activated_output.std().item(),
                'zero_fraction': (activated_output == 0).float().mean().item(),
                'activation_name': self.activation_name
            }
            
            return stats


class FFNAnalyzer:
    """
    Utility class for analyzing and visualizing FFN behavior.
    
    This class helps understand how different components of the FFN
    affect the transformations and learning process.
    """
    
    @staticmethod
    def compare_activations(input_tensor: torch.Tensor, d_model: int = 512, d_ff: int = 2048):
        """
        Compare different activation functions on the same input.
        
        This demonstrates how different activation functions transform
        the same input differently, affecting learning dynamics.
        """
        print("=" * 60)
        print("ACTIVATION FUNCTION COMPARISON")
        print("=" * 60)
        
        activations = ['relu', 'gelu', 'swish']
        results = {}
        
        for activation in activations:
            ffn = PositionwiseFeedForward(d_model, d_ff, activation=activation)
            
            # Forward pass
            with torch.no_grad():
                output = ffn(input_tensor)
                stats = ffn.get_activation_stats(input_tensor)
                
                results[activation] = {
                    'output': output,
                    'stats': stats
                }
                
                print(f"\n{activation.upper()} Activation:")
                print(f"  - Linear output mean: {stats['linear_mean']:.4f}")
                print(f"  - Linear output std:  {stats['linear_std']:.4f}")
                print(f"  - Activated mean:     {stats['activated_mean']:.4f}")
                print(f"  - Activated std:      {stats['activated_std']:.4f}")
                print(f"  - Zero fraction:      {stats['zero_fraction']:.4f}")
        
        return results
    
    @staticmethod
    def demonstrate_expansion_compression(d_model: int = 512, seq_len: int = 10):
        """
        Demonstrate the expand-then-compress mechanism of FFN.
        
        This shows how the FFN first expands the representation to a higher
        dimension for complex transformations, then compresses back.
        """
        print("\n" + "=" * 60)
        print("FFN EXPANSION-COMPRESSION DEMONSTRATION")
        print("=" * 60)
        
        # Create sample input
        batch_size = 2
        input_tensor = torch.randn(batch_size, seq_len, d_model)
        
        d_ff = 4 * d_model  # Standard expansion factor
        ffn = PositionwiseFeedForward(d_model, d_ff)
        
        print(f"Input shape:  {input_tensor.shape}")
        print(f"d_model:      {d_model}")
        print(f"d_ff:         {d_ff} (expansion factor: {d_ff // d_model}x)")
        
        # Forward pass with intermediate outputs
        with torch.no_grad():
            # Step 1: Expansion
            expanded = ffn.linear1(input_tensor)
            print(f"After expansion: {expanded.shape}")
            
            # Step 2: Activation
            activated = ffn.activation(expanded)
            print(f"After activation: {activated.shape}")
            
            # Step 3: Compression
            output = ffn.linear2(activated)
            print(f"Final output: {output.shape}")
            
            # Analysis
            expansion_ratio = expanded.numel() / input_tensor.numel()
            print(f"\nParameter analysis:")
            print(f"  - Expansion ratio: {expansion_ratio:.1f}x")
            print(f"  - Linear1 params: {ffn.linear1.weight.numel() + ffn.linear1.bias.numel():,}")
            print(f"  - Linear2 params: {ffn.linear2.weight.numel() + ffn.linear2.bias.numel():,}")
            print(f"  - Total FFN params: {sum(p.numel() for p in ffn.parameters()):,}")
    
    @staticmethod
    def analyze_dropout_effect(input_tensor: torch.Tensor, dropout_rates: list = [0.0, 0.1, 0.3, 0.5]):
        """
        Analyze the effect of different dropout rates on FFN output.
        
        Dropout is crucial for preventing overfitting in deep networks.
        This demonstrates how different dropout rates affect the output.
        """
        print("\n" + "=" * 60)
        print("DROPOUT EFFECT ANALYSIS")
        print("=" * 60)
        
        d_model = input_tensor.shape[-1]
        d_ff = 4 * d_model
        
        print("Dropout and Regularization:")
        print("- Dropout randomly sets neurons to zero during training")
        print("- Prevents over-reliance on specific neurons")
        print("- Improves generalization to unseen data")
        print("- Higher dropout = more regularization but slower training")
        
        for dropout_rate in dropout_rates:
            ffn = PositionwiseFeedForward(d_model, d_ff, dropout=dropout_rate)
            ffn.train()  # Enable training mode for dropout
            
            # Multiple forward passes to see dropout variance
            outputs = []
            for _ in range(5):
                with torch.no_grad():
                    output = ffn(input_tensor)
                    outputs.append(output)
            
            # Calculate variance across runs
            output_stack = torch.stack(outputs)
            output_variance = output_stack.var(dim=0).mean()
            
            print(f"\nDropout rate: {dropout_rate:.1f}")
            print(f"  - Output variance: {output_variance:.6f}")
            print(f"  - Effect: {'Higher variance (more regularization)' if dropout_rate > 0.2 else 'Lower variance (less regularization)'}")


def educational_demonstrations():
    """
    Run comprehensive educational demonstrations of the FFN.
    
    This function showcases all the key concepts and components
    of the feed-forward network in Transformers.
    """
    print("TRANSFORMER FEED-FORWARD NETWORK EDUCATIONAL DEMO")
    print("=" * 80)
    
    # Create sample input (batch_size=2, seq_len=8, d_model=512)
    batch_size, seq_len, d_model = 2, 8, 512
    input_tensor = torch.randn(batch_size, seq_len, d_model)
    
    print(f"Sample input shape: {input_tensor.shape}")
    print(f"Input represents: {batch_size} sequences of {seq_len} tokens, each with {d_model} features")
    
    # 1. Basic FFN demonstration
    print("\n" + "1. BASIC FFN FORWARD PASS")
    print("-" * 40)
    
    d_ff = 2048  # Standard 4x expansion
    ffn = PositionwiseFeedForward(d_model, d_ff)
    
    output = ffn(input_tensor)
    print(f"FFN output shape: {output.shape}")
    print(f"[✓] Shape preserved for residual connections")
    
    # 2. Activation function comparison
    FFNAnalyzer.compare_activations(input_tensor, d_model, d_ff)
    
    # 3. Expansion-compression demonstration
    FFNAnalyzer.demonstrate_expansion_compression(d_model, seq_len)
    
    # 4. Dropout analysis
    FFNAnalyzer.analyze_dropout_effect(input_tensor)
    
    # 5. Why FFN is needed
    print("\n" + "=" * 60)
    print("WHY TRANSFORMERS NEED FEED-FORWARD NETWORKS")
    print("=" * 60)
    
    print("""
    1. NON-LINEARITY:
       - Attention mechanisms are essentially linear operations
       - Without FFN, the entire Transformer would be linear
       - FFN introduces crucial non-linear transformations
    
    2. FEATURE INTERACTION:
       - Attention captures relationships BETWEEN positions
       - FFN enables complex feature interactions WITHIN each position
       - Allows learning of position-specific transformations
    
    3. REPRESENTATION LEARNING:
       - Expands representation to higher dimension (more expressiveness)
       - Compresses back to original dimension (forces meaningful learning)
       - Creates hierarchical feature representations
    
    4. COMPUTATIONAL CAPACITY:
       - FFN typically contains 2/3 of the model's parameters
       - Provides the main "thinking" capacity of the Transformer
       - Enables complex pattern recognition and feature extraction
    
    5. COMPLEMENTARY TO ATTENTION:
       - Attention: "What to focus on" (relationship modeling)
       - FFN: "How to transform" (feature processing)
       - Together: Complete information processing pipeline
    """)
    
    # 6. Architecture insights
    print("\n" + "=" * 60)
    print("ARCHITECTURE INSIGHTS")
    print("=" * 60)
    
    total_params = sum(p.numel() for p in ffn.parameters())
    linear1_params = ffn.linear1.weight.numel() + ffn.linear1.bias.numel()
    linear2_params = ffn.linear2.weight.numel() + ffn.linear2.bias.numel()
    
    print(f"Parameter Distribution:")
    print(f"  - Linear1 (expand):  {linear1_params:,} params ({linear1_params/total_params*100:.1f}%)")
    print(f"  - Linear2 (compress): {linear2_params:,} params ({linear2_params/total_params*100:.1f}%)")
    print(f"  - Total FFN:         {total_params:,} params")
    print(f"  - Memory overhead:   {d_ff/d_model:.1f}x during computation")
    
    print(f"\nComputational Complexity:")
    print(f"  - FLOPs per token: ~{2 * d_model * d_ff:,}")
    print(f"  - Memory peak:     {d_ff/d_model:.1f}x input size")
    print(f"  - Parallelizable:  Yes (position-wise operation)")


if __name__ == "__main__":
    # Run educational demonstrations
    educational_demonstrations()
    
    print("\n" + "=" * 80)
    print("SUMMARY: KEY TAKEAWAYS")
    print("=" * 80)
    print("""
    1. FFN provides essential NON-LINEARITY to Transformers
    2. Two-layer structure: EXPAND (learn) → COMPRESS (refine)
    3. Different activations have different properties:
       - ReLU: Fast, simple, can have dying neurons
       - GELU: Smooth, good for Transformers, more expensive
       - Swish: Self-gated, smooth gradients, deep network friendly
    4. Dropout prevents overfitting and improves generalization
    5. FFN complements attention: attention finds relationships, FFN processes features
    6. Contains majority of model parameters (computational capacity)
    7. Position-wise: same transformation applied to each sequence position
    """)