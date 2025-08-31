"""
Comprehensive Positional Encoding Implementation for Transformer Architecture
===========================================================================

This module implements and demonstrates positional encodings used in Transformer models,
with detailed educational explanations and visualizations.

Educational Focus:
- Why Transformers need positional information
- Mathematical foundation of sinusoidal encodings
- Comparison between fixed and learned encodings
- Visual demonstrations and mathematical verification

Author: AI Bootcamp - Week 5-6 Transformer Architecture
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional
import math


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding from "Attention Is All You Need"
    
    MATHEMATICAL FOUNDATION:
    The original Transformer paper uses sinusoidal functions to create positional encodings:
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Where:
    - pos: position in the sequence
    - i: dimension index (0 to d_model/2)
    - d_model: model dimension
    
    WHY THIS WORKS:
    1. Each position gets a unique encoding vector
    2. Similar positions have similar encodings (continuity)
    3. The model can learn relative positions through dot products
    4. Different frequencies handle various sequence lengths
    5. No training required - deterministic and consistent
    """
    
    def __init__(self, d_model: int, max_seq_length: int = 5000, dropout: float = 0.1):
        """
        Initialize sinusoidal positional encoding.
        
        Args:
            d_model: Model dimension (must be even)
            max_seq_length: Maximum sequence length to precompute
            dropout: Dropout rate for regularization
        """
        super().__init__()
        
        if d_model % 2 != 0:
            raise ValueError(f"d_model ({d_model}) must be even for sinusoidal encoding")
        
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding table
        pe = torch.zeros(max_seq_length, d_model)
        
        # Create position indices [0, 1, 2, ..., max_seq_length-1]
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        # Create dimension indices for the encoding formula
        # div_term represents 10000^(2i/d_model) for each dimension pair
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        # Apply sine to even indices (0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices (1, 3, 5, ...)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0).transpose(0, 1)  # Shape: [max_seq_length, 1, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input embeddings [seq_length, batch_size, d_model]
            
        Returns:
            x + positional encodings with same shape
        """
        # Add positional encoding (broadcasting across batch dimension)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
    def get_positional_encoding(self, seq_length: int) -> torch.Tensor:
        """Get positional encoding for visualization purposes."""
        return self.pe[:seq_length, 0, :].detach()


class LearnedPositionalEncoding(nn.Module):
    """
    Learned Positional Encoding - Alternative to sinusoidal encoding.
    
    Instead of using fixed sinusoidal functions, this approach learns
    optimal positional representations during training.
    
    ADVANTAGES:
    - Can adapt to specific tasks and data patterns
    - Potentially better performance on specific domains
    
    DISADVANTAGES:
    - Requires training data
    - Cannot easily extrapolate to longer sequences than seen in training
    - Uses additional parameters
    """
    
    def __init__(self, d_model: int, max_seq_length: int = 5000, dropout: float = 0.1):
        """
        Initialize learned positional encoding.
        
        Args:
            d_model: Model dimension
            max_seq_length: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        # Learnable positional embeddings
        self.pe = nn.Embedding(max_seq_length, d_model)
        
        # Initialize with small random values
        nn.init.normal_(self.pe.weight, mean=0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learned positional encoding to input embeddings.
        
        Args:
            x: Input embeddings [seq_length, batch_size, d_model]
            
        Returns:
            x + positional encodings with same shape
        """
        seq_length = x.size(0)
        positions = torch.arange(seq_length, device=x.device)
        
        # Get positional embeddings and add to input
        pos_embeddings = self.pe(positions).unsqueeze(1)  # [seq_length, 1, d_model]
        x = x + pos_embeddings
        
        return self.dropout(x)


class PositionalEncodingVisualizer:
    """
    Educational visualization tools for understanding positional encodings.
    """
    
    @staticmethod
    def visualize_sinusoidal_patterns(d_model: int = 128, max_pos: int = 100):
        """
        Visualize the sinusoidal patterns in positional encoding.
        
        This shows how different dimensions oscillate at different frequencies,
        creating unique patterns for each position.
        """
        # Create positional encoding
        pe_layer = SinusoidalPositionalEncoding(d_model, max_pos)
        pe = pe_layer.get_positional_encoding(max_pos).numpy()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Sinusoidal Positional Encoding Patterns', fontsize=16)
        
        # 1. Heatmap of positional encodings
        sns.heatmap(pe.T, cmap='RdBu', center=0, ax=axes[0,0], 
                   cbar_kws={'label': 'Encoding Value'})
        axes[0,0].set_title('Complete Positional Encoding Matrix')
        axes[0,0].set_xlabel('Position')
        axes[0,0].set_ylabel('Dimension')
        
        # 2. Individual dimension patterns
        dimensions_to_show = [0, 1, d_model//4, d_model//2, d_model-2, d_model-1]
        for i, dim in enumerate(dimensions_to_show):
            if i < 6:  # Limit number of lines
                axes[0,1].plot(pe[:, dim], label=f'Dim {dim}', alpha=0.7)
        
        axes[0,1].set_title('Individual Dimension Patterns')
        axes[0,1].set_xlabel('Position')
        axes[0,1].set_ylabel('Encoding Value')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Frequency analysis - show how frequencies change across dimensions
        positions = np.arange(max_pos)
        frequencies = []
        
        for dim in range(0, d_model, 2):  # Even dimensions (sine components)
            div_term = np.exp((dim/2) * (-math.log(10000.0) / d_model))
            frequencies.append(div_term)
        
        axes[1,0].plot(frequencies)
        axes[1,0].set_title('Frequency Distribution Across Dimensions')
        axes[1,0].set_xlabel('Dimension Pair Index')
        axes[1,0].set_ylabel('Frequency (1/wavelength)')
        axes[1,0].set_yscale('log')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Distance between consecutive positions
        position_distances = []
        for pos in range(min(50, max_pos-1)):
            dist = np.linalg.norm(pe[pos+1] - pe[pos])
            position_distances.append(dist)
        
        axes[1,1].plot(position_distances)
        axes[1,1].set_title('Distance Between Consecutive Positions')
        axes[1,1].set_xlabel('Position')
        axes[1,1].set_ylabel('Euclidean Distance')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return pe
    
    @staticmethod
    def demonstrate_position_uniqueness(d_model: int = 64, positions_to_test: int = 20):
        """
        Demonstrate that each position gets a unique encoding vector.
        
        This is crucial for the model to distinguish between different positions.
        """
        pe_layer = SinusoidalPositionalEncoding(d_model, positions_to_test)
        pe = pe_layer.get_positional_encoding(positions_to_test)
        
        # Calculate pairwise distances
        distances = torch.cdist(pe, pe)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Distance matrix heatmap
        sns.heatmap(distances.numpy(), annot=True if positions_to_test <= 10 else False,
                   cmap='viridis', ax=axes[0])
        axes[0].set_title('Pairwise Distances Between Position Encodings')
        axes[0].set_xlabel('Position')
        axes[0].set_ylabel('Position')
        
        # 2. Minimum distances (excluding diagonal)
        mask = torch.eye(positions_to_test).bool()
        distances_masked = distances.clone()
        distances_masked[mask] = float('inf')
        min_distances = distances_masked.min(dim=1)[0]
        
        axes[1].bar(range(positions_to_test), min_distances.numpy())
        axes[1].set_title('Minimum Distance to Any Other Position')
        axes[1].set_xlabel('Position')
        axes[1].set_ylabel('Minimum Distance')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"\nPosition Uniqueness Analysis:")
        print(f"Average minimum distance: {min_distances.mean():.4f}")
        print(f"Smallest minimum distance: {min_distances.min():.4f}")
        print(f"Standard deviation of distances: {min_distances.std():.4f}")
        
        return distances
    
    @staticmethod
    def compare_encodings(d_model: int = 64, seq_length: int = 50):
        """
        Compare sinusoidal vs learned positional encodings.
        """
        # Create both encoding types
        sin_pe = SinusoidalPositionalEncoding(d_model, seq_length)
        learned_pe = LearnedPositionalEncoding(d_model, seq_length)
        
        # Get encodings
        sin_encoding = sin_pe.get_positional_encoding(seq_length)
        
        # For learned encoding, we'll use random initialization to simulate untrained state
        positions = torch.arange(seq_length)
        learned_encoding = learned_pe.pe(positions).detach()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Sinusoidal vs Learned Positional Encodings', fontsize=16)
        
        # Sinusoidal encoding
        sns.heatmap(sin_encoding.T.numpy(), cmap='RdBu', center=0, 
                   ax=axes[0,0], cbar_kws={'label': 'Value'})
        axes[0,0].set_title('Sinusoidal Positional Encoding')
        axes[0,0].set_xlabel('Position')
        axes[0,0].set_ylabel('Dimension')
        
        # Learned encoding (random initialization)
        sns.heatmap(learned_encoding.T.numpy(), cmap='RdBu', center=0,
                   ax=axes[0,1], cbar_kws={'label': 'Value'})
        axes[0,1].set_title('Learned Positional Encoding (Random Init)')
        axes[0,1].set_xlabel('Position')
        axes[0,1].set_ylabel('Dimension')
        
        # Distance analysis for both
        sin_distances = torch.cdist(sin_encoding, sin_encoding)
        learned_distances = torch.cdist(learned_encoding, learned_encoding)
        
        # Plot distance distributions
        sin_dist_flat = sin_distances[torch.triu(torch.ones_like(sin_distances), diagonal=1).bool()]
        learned_dist_flat = learned_distances[torch.triu(torch.ones_like(learned_distances), diagonal=1).bool()]
        
        axes[1,0].hist(sin_dist_flat.numpy(), bins=30, alpha=0.7, label='Sinusoidal', density=True)
        axes[1,0].hist(learned_dist_flat.numpy(), bins=30, alpha=0.7, label='Learned (Random)', density=True)
        axes[1,0].set_title('Distribution of Pairwise Distances')
        axes[1,0].set_xlabel('Distance')
        axes[1,0].set_ylabel('Density')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Pattern regularity comparison
        sin_autocorr = np.correlate(sin_encoding[:, 0], sin_encoding[:, 0], mode='full')
        learned_autocorr = np.correlate(learned_encoding[:, 0], learned_encoding[:, 0], mode='full')
        
        axes[1,1].plot(sin_autocorr[len(sin_autocorr)//2:len(sin_autocorr)//2+20], 
                      label='Sinusoidal', marker='o')
        axes[1,1].plot(learned_autocorr[len(learned_autocorr)//2:len(learned_autocorr)//2+20], 
                      label='Learned (Random)', marker='s')
        axes[1,1].set_title('Autocorrelation of First Dimension')
        axes[1,1].set_xlabel('Lag')
        axes[1,1].set_ylabel('Correlation')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def demonstrate_why_transformers_need_positions():
    """
    Educational demonstration of why Transformers need positional information.
    
    This shows how attention mechanisms are position-agnostic without
    explicit positional encodings.
    """
    print("=" * 80)
    print("WHY TRANSFORMERS NEED POSITIONAL ENCODING")
    print("=" * 80)
    
    print("\n1. THE PROBLEM: Attention is Position-Agnostic")
    print("-" * 50)
    
    # Example sentence
    sentence = ["The", "cat", "sat", "on", "the", "mat"]
    permuted = ["cat", "The", "mat", "on", "sat", "the"]
    
    print(f"Original sentence: {' '.join(sentence)}")
    print(f"Permuted sentence: {' '.join(permuted)}")
    
    # Simulate word embeddings (random for demonstration)
    torch.manual_seed(42)
    d_model = 8
    embeddings_original = torch.randn(len(sentence), d_model)
    
    # Same embeddings but reordered
    word_to_embedding = {word: embeddings_original[i] for i, word in enumerate(sentence)}
    embeddings_permuted = torch.stack([word_to_embedding[word] for word in permuted])
    
    print(f"\nWithout positional encoding, these embeddings are identical:")
    print(f"Set of original embeddings equals set of permuted: {torch.allclose(torch.sort(embeddings_original.flatten())[0], torch.sort(embeddings_permuted.flatten())[0])}")
    
    print("\n2. THE SOLUTION: Add Position Information")
    print("-" * 50)
    
    # Add positional encodings
    pe_layer = SinusoidalPositionalEncoding(d_model, len(sentence))
    
    original_with_pos = embeddings_original + pe_layer.get_positional_encoding(len(sentence))
    permuted_with_pos = embeddings_permuted + pe_layer.get_positional_encoding(len(permuted))
    
    print(f"With positional encoding, they become different:")
    print(f"Embeddings are now different: {not torch.allclose(original_with_pos, permuted_with_pos, atol=1e-6)}")
    
    # Show the difference
    difference = torch.norm(original_with_pos - permuted_with_pos, dim=1)
    print(f"L2 differences per position: {difference.tolist()}")
    
    print("\n3. MATHEMATICAL VERIFICATION")
    print("-" * 50)
    
    # Show that each position gets unique encoding
    pe_matrix = pe_layer.get_positional_encoding(6)
    pairwise_distances = torch.cdist(pe_matrix, pe_matrix)
    
    print("Pairwise distances between positional encodings:")
    for i in range(len(sentence)):
        for j in range(len(sentence)):
            if i < j:
                print(f"  Distance between pos {i} and pos {j}: {pairwise_distances[i,j]:.4f}")


def mathematical_deep_dive():
    """
    Deep mathematical explanation of sinusoidal positional encoding.
    """
    print("\n" + "=" * 80)
    print("MATHEMATICAL DEEP DIVE: Sinusoidal Positional Encoding")
    print("=" * 80)
    
    print("\n1. THE FORMULA")
    print("-" * 50)
    print("PE(pos, 2i) = sin(pos / 10000^(2i/d_model))")
    print("PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))")
    print()
    print("Where:")
    print("  - pos: position in sequence (0, 1, 2, ...)")
    print("  - i: dimension index (0 to d_model/2)")
    print("  - 2i, 2i+1: even and odd dimension indices")
    print("  - 10000: chosen base for wavelength scaling")
    
    print("\n2. WHY SINE AND COSINE?")
    print("-" * 50)
    print("Key mathematical properties:")
    print("  a) Bounded values: sin/cos ∈ [-1, 1]")
    print("  b) Periodic: creates regular patterns")
    print("  c) Orthogonal: sin and cos are 90° phase-shifted")
    print("  d) Smooth: continuous and differentiable")
    print("  e) Deterministic: same input always gives same output")
    
    print("\n3. FREQUENCY SCALING")
    print("-" * 50)
    
    d_model = 512  # Standard Transformer size
    print(f"For d_model = {d_model}:")
    
    # Show frequency progression
    frequencies = []
    wavelengths = []
    
    for i in range(0, min(10, d_model//2)):
        freq = 1 / (10000 ** (2*i / d_model))
        wavelength = 2 * math.pi / freq
        frequencies.append(freq)
        wavelengths.append(wavelength)
        
        print(f"  Dimension pair {i:2d}: frequency = {freq:.2e}, wavelength = {wavelength:.1f}")
    
    print(f"\n  Lowest frequency (dim 0): {frequencies[0]:.2e}")
    print(f"  Highest frequency (dim {min(9, d_model//2-1)}): {frequencies[-1]:.2e}")
    print(f"  Frequency ratio: {frequencies[0]/frequencies[-1]:.0f}x")
    
    print("\n4. RELATIVE POSITION ENCODING")
    print("-" * 50)
    print("The dot product of two positional encodings reveals relative distance:")
    
    # Demonstrate with actual calculations
    pe_layer = SinusoidalPositionalEncoding(64)
    pe = pe_layer.get_positional_encoding(20)
    
    pos_5 = pe[5]
    pos_10 = pe[10]
    pos_15 = pe[15]
    
    dot_5_10 = torch.dot(pos_5, pos_10).item()
    dot_5_15 = torch.dot(pos_5, pos_15).item()
    dot_10_15 = torch.dot(pos_10, pos_15).item()
    
    print(f"  Dot product PE(5) · PE(10) = {dot_5_10:.4f} (distance: 5)")
    print(f"  Dot product PE(5) · PE(15) = {dot_5_15:.4f} (distance: 10)")
    print(f"  Dot product PE(10) · PE(15) = {dot_10_15:.4f} (distance: 5)")
    print("  → Similar distances give similar dot products!")


def practical_implementation_guide():
    """
    Practical guide for implementing and using positional encodings.
    """
    print("\n" + "=" * 80)
    print("PRACTICAL IMPLEMENTATION GUIDE")
    print("=" * 80)
    
    print("\n1. BASIC USAGE")
    print("-" * 50)
    
    # Example usage
    batch_size, seq_length, d_model = 32, 100, 512
    
    # Create dummy input embeddings
    input_embeddings = torch.randn(seq_length, batch_size, d_model)
    
    # Initialize positional encoding
    pos_encoder = SinusoidalPositionalEncoding(d_model)
    
    # Add positional information
    encoded_input = pos_encoder(input_embeddings)
    
    print(f"Input shape: {input_embeddings.shape}")
    print(f"Output shape: {encoded_input.shape}")
    print(f"Same shape: {input_embeddings.shape == encoded_input.shape}")
    
    print("\n2. MEMORY CONSIDERATIONS")
    print("-" * 50)
    
    max_seq = 5000
    memory_usage = max_seq * d_model * 4  # 4 bytes per float32
    print(f"Memory for PE table (max_seq={max_seq}, d_model={d_model}): {memory_usage/1024/1024:.1f} MB")
    
    print("Optimization strategies:")
    print("  - Pre-compute and cache PE for common sequence lengths")
    print("  - Use buffer registration (no gradients needed)")
    print("  - Consider half precision for inference")
    
    print("\n3. WHEN TO USE WHICH ENCODING")
    print("-" * 50)
    
    print("Sinusoidal Encoding - Use When:")
    print("  ✓ You want deterministic, training-free positions")
    print("  ✓ You need to handle variable sequence lengths")
    print("  ✓ You want to extrapolate to longer sequences")
    print("  ✓ You're implementing the original Transformer")
    
    print("\nLearned Encoding - Use When:")
    print("  ✓ You have task-specific position patterns")
    print("  ✓ Fixed maximum sequence length is acceptable")
    print("  ✓ You have enough training data")
    print("  ✓ Task benefits from learned position representations")
    
    print("\n4. COMMON IMPLEMENTATION MISTAKES")
    print("-" * 50)
    
    print("❌ Adding PE after layer normalization")
    print("✅ Add PE before the first attention layer")
    print()
    print("❌ Training PE parameters when using sinusoidal")
    print("✅ Register PE as buffer (no gradients)")
    print()
    print("❌ Wrong dimension order in implementation")
    print("✅ Ensure consistent [seq_len, batch, d_model] ordering")
    print()
    print("❌ Not handling variable sequence lengths")
    print("✅ Slice PE table based on actual sequence length")


def main():
    """
    Main educational demonstration of positional encodings.
    """
    print("TRANSFORMER POSITIONAL ENCODING - EDUCATIONAL IMPLEMENTATION")
    print("=" * 80)
    
    # 1. Explain why we need positional encoding
    demonstrate_why_transformers_need_positions()
    
    # 2. Mathematical deep dive
    mathematical_deep_dive()
    
    # 3. Visual demonstrations
    print("\n" + "=" * 80)
    print("VISUAL DEMONSTRATIONS")
    print("=" * 80)
    
    visualizer = PositionalEncodingVisualizer()
    
    print("\n1. Visualizing sinusoidal patterns...")
    pe_matrix = visualizer.visualize_sinusoidal_patterns(d_model=128, max_pos=100)
    
    print("\n2. Demonstrating position uniqueness...")
    distances = visualizer.demonstrate_position_uniqueness(d_model=64, positions_to_test=20)
    
    print("\n3. Comparing encoding types...")
    visualizer.compare_encodings(d_model=64, seq_length=50)
    
    # 4. Practical implementation guide
    practical_implementation_guide()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print("Key Takeaways:")
    print("1. Transformers need explicit positional information (no inherent order)")
    print("2. Sinusoidal encoding provides unique, consistent position representations")
    print("3. Different frequencies handle various sequence lengths and patterns")
    print("4. Mathematical properties enable relative position understanding")
    print("5. Choice between sinusoidal vs learned depends on specific requirements")
    
    print("\nNext Steps:")
    print("- Implement multi-head attention mechanism")
    print("- Combine positional encoding with attention layers")
    print("- Experiment with different encoding strategies")
    print("- Study relative positional encodings (RoPE, etc.)")


if __name__ == "__main__":
    main()