"""
LoRA (Low-Rank Adaptation) Implementation from Scratch
=====================================================

This module provides a comprehensive educational implementation of LoRA for fine-tuning
neural networks, particularly transformer models, with detailed mathematical explanations.

Mathematical Foundation:
-----------------------
LoRA decomposes weight updates into low-rank matrices:
    W = W₀ + ΔW = W₀ + BA

Where:
- W₀: Original pre-trained weights (frozen)
- ΔW: Weight update decomposed as B @ A
- B: Down-projection matrix (d × r)
- A: Up-projection matrix (r × k) 
- r: Rank (r << min(d, k))

Key Insights:
- Most weight updates during fine-tuning are low-rank
- We can capture essential adaptations with much fewer parameters
- Original model knowledge is preserved while enabling task-specific learning

Author: AI Intensive Bootcamp - Fine-tuning Specialist
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
import math


class LoRALayer(nn.Module):
    """
    Educational LoRA implementation for any linear layer.
    
    This class demonstrates the core LoRA concept: decomposing weight updates
    into low-rank matrices B and A, where the full update ΔW = B @ A.
    
    Mathematical Breakdown:
    ----------------------
    Original forward pass: y = xW₀
    LoRA forward pass: y = x(W₀ + αΔW) = xW₀ + α(xA^T)B^T
    
    Where:
    - α (alpha): Scaling factor for adaptation strength
    - A: r × input_dim (initialized with Kaiming uniform)
    - B: output_dim × r (initialized with zeros)
    
    This ensures the adaptation starts at zero and gradually learns.
    """
    
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        merge_weights: bool = False
    ):
        """
        Initialize LoRA adaptation for a linear layer.
        
        Args:
            original_layer: The pre-trained linear layer to adapt
            rank: Bottleneck dimension (r). Higher rank = more capacity, more parameters
            alpha: Scaling factor for LoRA updates. Common values: 1, 8, 16, 32
            dropout: Dropout probability for LoRA path (regularization)
            merge_weights: Whether to merge LoRA weights into original layer
        
        Parameter Efficiency Analysis:
        -----------------------------
        Original parameters: input_dim × output_dim
        LoRA parameters: (input_dim + output_dim) × rank
        Reduction ratio: (input_dim × output_dim) / ((input_dim + output_dim) × rank)
        
        Example: 4096 → 4096 layer with rank=4
        - Original: 16,777,216 parameters
        - LoRA: 32,768 parameters  
        - Reduction: 512× fewer parameters!
        """
        super().__init__()
        
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank  # Normalize by rank for stable training
        self.merged = False
        self.merge_weights = merge_weights
        
        # Get dimensions from original layer
        self.input_dim = original_layer.in_features
        self.output_dim = original_layer.out_features
        
        # Freeze original weights - this preserves pre-trained knowledge
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # LoRA decomposition matrices
        # A: Maps input to bottleneck (r dimensions)
        self.lora_A = nn.Parameter(torch.empty(rank, self.input_dim))
        # B: Maps bottleneck to output  
        self.lora_B = nn.Parameter(torch.empty(self.output_dim, rank))
        
        # Optional dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        Initialize LoRA parameters using best practices.
        
        Strategy:
        - A: Kaiming uniform (like original transformer layers)
        - B: Zero initialization (ensures adaptation starts at identity)
        
        This initialization ensures:
        1. The adaptation starts with no change to the original model
        2. Learning begins smoothly without disrupting pre-trained features
        3. Gradients flow properly from the beginning
        """
        # Initialize A with Kaiming uniform (preserves variance)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        
        # Initialize B to zero (adaptation starts at identity)
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining original layer with LoRA adaptation.
        
        Mathematical Flow:
        -----------------
        1. Original path: y₀ = x @ W₀^T + b₀
        2. LoRA path: y_lora = (x @ A^T @ B^T) * scaling
        3. Combined: y = y₀ + y_lora
        
        Computational Efficiency:
        ------------------------
        Instead of materializing full ΔW = B @ A, we compute:
        x @ A^T @ B^T = (x @ A^T) @ B^T
        
        This reduces memory from O(d²) to O(dr) where d >> r.
        """
        # Original pre-trained computation
        result = self.original_layer(x)
        
        if not self.merged:
            # LoRA adaptation path
            # Step 1: Project to bottleneck dimension
            lora_x = x @ self.lora_A.T  # (batch, seq, rank)
            lora_x = self.dropout(lora_x)  # Optional regularization
            
            # Step 2: Project back to output dimension  
            lora_result = lora_x @ self.lora_B.T  # (batch, seq, output_dim)
            
            # Step 3: Scale and add to original
            result = result + lora_result * self.scaling
        
        return result
    
    def merge_weights(self):
        """
        Merge LoRA weights into the original layer for inference efficiency.
        
        This computes W_merged = W₀ + α * B @ A and updates the original layer.
        After merging, the forward pass becomes a single matrix multiplication.
        
        Use this for:
        - Production deployment (faster inference)
        - Model sharing (single set of weights)
        - Memory optimization during inference
        """
        if not self.merged:
            # Compute the full LoRA update: ΔW = B @ A
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            
            # Add to original weights
            self.original_layer.weight.data += delta_w
            self.merged = True
    
    def unmerge_weights(self):
        """
        Separate LoRA weights from original layer.
        
        This undoes the merge operation by subtracting the LoRA update.
        Useful for:
        - Switching between different LoRA adapters
        - Debugging and analysis
        - Continued training after inference
        """
        if self.merged:
            # Compute and subtract the LoRA update
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.original_layer.weight.data -= delta_w
            self.merged = False
    
    def get_parameter_analysis(self) -> Dict[str, int]:
        """
        Analyze parameter efficiency of LoRA vs full fine-tuning.
        
        Returns detailed breakdown of parameter counts and efficiency gains.
        """
        original_params = self.input_dim * self.output_dim
        lora_params = (self.input_dim + self.output_dim) * self.rank
        
        # Add bias if present
        if self.original_layer.bias is not None:
            original_params += self.output_dim
        
        return {
            'original_parameters': original_params,
            'lora_parameters': lora_params,
            'trainable_parameters': lora_params,
            'frozen_parameters': original_params,
            'reduction_factor': original_params / lora_params if lora_params > 0 else 0,
            'efficiency_percentage': (1 - lora_params / original_params) * 100,
            'rank': self.rank,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim
        }


class LoRATransformerLayer(nn.Module):
    """
    Educational example of applying LoRA to a complete transformer layer.
    
    This demonstrates how to integrate LoRA with:
    - Multi-head attention (query, key, value, output projections)
    - Feed-forward network layers
    - Layer normalization (typically not adapted)
    
    Common LoRA Application Patterns:
    --------------------------------
    1. Attention-only: Apply LoRA to Q, K, V, O projections
    2. FFN-only: Apply LoRA to feed-forward layers
    3. Full: Apply LoRA to all linear layers
    4. Selective: Apply based on gradient analysis or importance scores
    """
    
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        rank: int = 4,
        alpha: float = 16.0,
        dropout: float = 0.1,
        apply_to_attention: bool = True,
        apply_to_ffn: bool = True
    ):
        """
        Initialize a transformer layer with LoRA adaptations.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward network dimension
            rank: LoRA rank for all adaptations
            alpha: LoRA scaling factor
            dropout: Dropout probability
            apply_to_attention: Whether to apply LoRA to attention layers
            apply_to_ffn: Whether to apply LoRA to feed-forward layers
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.head_dim = d_model // num_heads
        
        # Original transformer components
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Attention projections (these are candidates for LoRA)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        # Feed-forward network (also candidates for LoRA)
        self.ffn_up = nn.Linear(d_model, d_ff)
        self.ffn_down = nn.Linear(d_ff, d_model)
        
        # Apply LoRA adaptations based on configuration
        self.lora_layers = nn.ModuleDict()
        
        if apply_to_attention:
            self.lora_layers['q_proj'] = LoRALayer(self.q_proj, rank, alpha, dropout)
            self.lora_layers['k_proj'] = LoRALayer(self.k_proj, rank, alpha, dropout)
            self.lora_layers['v_proj'] = LoRALayer(self.v_proj, rank, alpha, dropout)
            self.lora_layers['o_proj'] = LoRALayer(self.o_proj, rank, alpha, dropout)
        
        if apply_to_ffn:
            self.lora_layers['ffn_up'] = LoRALayer(self.ffn_up, rank, alpha, dropout)
            self.lora_layers['ffn_down'] = LoRALayer(self.ffn_down, rank, alpha, dropout)
    
    def attention(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Multi-head attention with optional LoRA adaptations.
        
        This shows how LoRA integrates seamlessly with existing architectures.
        The attention computation remains identical - only the projection
        layers are enhanced with low-rank adaptations.
        """
        batch_size, seq_len = x.shape[:2]
        
        # Apply LoRA-adapted projections if available
        if 'q_proj' in self.lora_layers:
            q = self.lora_layers['q_proj'](x)
            k = self.lora_layers['k_proj'](x)
            v = self.lora_layers['v_proj'](x)
        else:
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project back
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Output projection (potentially LoRA-adapted)
        if 'o_proj' in self.lora_layers:
            return self.lora_layers['o_proj'](attn_output)
        else:
            return self.o_proj(attn_output)
    
    def feed_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Feed-forward network with optional LoRA adaptations.
        
        Demonstrates LoRA application to MLP layers, which often contain
        the majority of parameters in transformer models.
        """
        # First projection (potentially LoRA-adapted)
        if 'ffn_up' in self.lora_layers:
            hidden = self.lora_layers['ffn_up'](x)
        else:
            hidden = self.ffn_up(x)
        
        # Activation function
        hidden = F.gelu(hidden)
        hidden = self.dropout(hidden)
        
        # Second projection (potentially LoRA-adapted)
        if 'ffn_down' in self.lora_layers:
            return self.lora_layers['ffn_down'](hidden)
        else:
            return self.ffn_down(hidden)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Standard transformer layer forward pass with LoRA adaptations.
        """
        # Self-attention with residual connection
        attn_output = self.attention(self.ln1(x), mask)
        x = x + attn_output
        
        # Feed-forward with residual connection
        ffn_output = self.feed_forward(self.ln2(x))
        x = x + ffn_output
        
        return x
    
    def get_parameter_analysis(self) -> Dict[str, any]:
        """
        Comprehensive analysis of parameter efficiency across the layer.
        """
        analysis = {
            'layer_type': 'transformer',
            'lora_adaptations': list(self.lora_layers.keys()),
            'total_lora_params': 0,
            'total_original_params': 0,
            'layer_breakdown': {}
        }
        
        # Analyze each LoRA layer
        for name, lora_layer in self.lora_layers.items():
            layer_analysis = lora_layer.get_parameter_analysis()
            analysis['layer_breakdown'][name] = layer_analysis
            analysis['total_lora_params'] += layer_analysis['lora_parameters']
            analysis['total_original_params'] += layer_analysis['original_parameters']
        
        # Calculate overall efficiency
        if analysis['total_original_params'] > 0:
            analysis['overall_reduction_factor'] = (
                analysis['total_original_params'] / analysis['total_lora_params']
            )
            analysis['overall_efficiency_percentage'] = (
                (1 - analysis['total_lora_params'] / analysis['total_original_params']) * 100
            )
        
        return analysis


def demonstrate_lora_efficiency(
    input_dim: int = 4096,
    output_dim: int = 4096,
    ranks: List[int] = [1, 2, 4, 8, 16, 32, 64]
) -> Dict[str, List[float]]:
    """
    Educational demonstration of LoRA parameter efficiency across different ranks.
    
    This function shows how rank selection affects:
    1. Parameter count reduction
    2. Model capacity (approximated by rank)
    3. Memory requirements
    
    Args:
        input_dim: Input dimension for analysis
        output_dim: Output dimension for analysis  
        ranks: List of ranks to analyze
    
    Returns:
        Dictionary containing efficiency metrics for each rank
    """
    original_params = input_dim * output_dim
    
    results = {
        'ranks': ranks,
        'lora_parameters': [],
        'reduction_factors': [],
        'efficiency_percentages': [],
        'parameter_ratios': []
    }
    
    for rank in ranks:
        lora_params = (input_dim + output_dim) * rank
        reduction_factor = original_params / lora_params
        efficiency_pct = (1 - lora_params / original_params) * 100
        param_ratio = lora_params / original_params
        
        results['lora_parameters'].append(lora_params)
        results['reduction_factors'].append(reduction_factor)
        results['efficiency_percentages'].append(efficiency_pct)
        results['parameter_ratios'].append(param_ratio)
    
    return results


def compare_training_approaches(model_size: str = "7B") -> Dict[str, Dict[str, any]]:
    """
    Educational comparison of different fine-tuning approaches.
    
    Compares:
    1. Full fine-tuning: Train all parameters
    2. LoRA (various ranks): Train low-rank adaptations
    3. Partial fine-tuning: Train only certain layers
    
    Args:
        model_size: Model size for analysis ("1B", "7B", "13B", "70B")
    
    Returns:
        Comparison metrics for different approaches
    """
    # Approximate parameter counts for different model sizes
    model_params = {
        "1B": 1_000_000_000,
        "7B": 7_000_000_000, 
        "13B": 13_000_000_000,
        "70B": 70_000_000_000
    }
    
    total_params = model_params.get(model_size, 7_000_000_000)
    
    # Estimate linear layer parameters (typically ~2/3 of total)
    linear_params = int(total_params * 0.67)
    
    approaches = {}
    
    # Full fine-tuning
    approaches['full_fine_tuning'] = {
        'trainable_parameters': total_params,
        'memory_multiplier': 4.0,  # Gradients + optimizer states
        'training_speed': 1.0,  # Baseline
        'adaptation_capacity': 1.0  # Full capacity
    }
    
    # LoRA with different ranks
    for rank in [4, 8, 16, 32]:
        # Approximate LoRA params (applied to ~50% of linear layers)
        lora_params = int(linear_params * 0.5 * rank / 2048)  # Rough estimate
        
        approaches[f'lora_r{rank}'] = {
            'trainable_parameters': lora_params,
            'memory_multiplier': 1.2,  # Much lower memory overhead
            'training_speed': 3.0,  # Faster due to fewer parameters
            'adaptation_capacity': rank / 64.0  # Relative to max practical rank
        }
    
    # Partial fine-tuning (last 25% of layers)
    approaches['partial_fine_tuning'] = {
        'trainable_parameters': int(total_params * 0.25),
        'memory_multiplier': 2.0,  # Moderate overhead
        'training_speed': 2.0,  # Faster than full
        'adaptation_capacity': 0.5  # Limited capacity
    }
    
    return approaches


def lora_mathematical_intuition() -> Dict[str, str]:
    """
    Educational explanation of the mathematical intuition behind LoRA.
    
    Returns key insights about why LoRA works so effectively for fine-tuning.
    """
    return {
        'core_hypothesis': (
            "The change in weights during fine-tuning has a low 'intrinsic rank', "
            "meaning it can be represented as the product of two much smaller matrices."
        ),
        
        'mathematical_form': (
            "W_adapted = W_pretrained + ΔW = W_pretrained + B·A\n"
            "Where B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), and r << min(d,k)"
        ),
        
        'intuitive_explanation': (
            "Think of LoRA as learning a 'compression' of the weight updates. "
            "Instead of changing all connections, we learn a low-dimensional "
            "subspace that captures the most important adaptations."
        ),
        
        'biological_analogy': (
            "Like how evolution makes small, targeted changes rather than "
            "redesigning entire organisms, LoRA makes focused adaptations "
            "while preserving the pre-trained 'genetic code' of the model."
        ),
        
        'rank_interpretation': (
            "Rank controls the 'bandwidth' of adaptation:\n"
            "- Low rank (2-4): Simple, targeted changes\n"
            "- Medium rank (8-16): Balanced adaptation\n" 
            "- High rank (32-64): Complex, nuanced changes"
        ),
        
        'scaling_factor_alpha': (
            "Alpha controls adaptation strength:\n"
            "- Low α: Conservative changes, preserves pre-training\n"
            "- High α: Aggressive adaptation, may overwrite pre-training\n"
            "- Typical values: α = r (balanced) or α = 2r (stronger)"
        ),
        
        'convergence_properties': (
            "LoRA often converges faster than full fine-tuning because:\n"
            "1. Smaller parameter space to search\n"
            "2. Regularization effect of low-rank constraint\n"
            "3. Preserved pre-trained features provide good initialization"
        )
    }


def create_lora_integration_example():
    """
    Educational example showing how to integrate LoRA with pre-trained models.
    
    This demonstrates the practical workflow for applying LoRA to real models.
    """
    
    # Example integration pattern
    integration_code = '''
# Practical LoRA Integration Example
# =================================

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

def apply_lora_to_model(model, target_modules=['q_proj', 'v_proj'], rank=4, alpha=16):
    """
    Apply LoRA to specific modules in a pre-trained model.
    
    Args:
        model: Pre-trained model (e.g., from Hugging Face)
        target_modules: List of module names to adapt
        rank: LoRA rank
        alpha: LoRA scaling factor
    """
    lora_modules = {}
    
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # Replace with LoRA-adapted version
                lora_module = LoRALayer(module, rank=rank, alpha=alpha)
                lora_modules[name] = lora_module
                
                # Update the model
                parent_module = model
                for attr in name.split('.')[:-1]:
                    parent_module = getattr(parent_module, attr)
                setattr(parent_module, name.split('.')[-1], lora_module)
    
    return lora_modules

# Usage example
model = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Apply LoRA to attention projections
lora_modules = apply_lora_to_model(
    model, 
    target_modules=['query', 'value'], 
    rank=8, 
    alpha=16
)

# Fine-tune only LoRA parameters
optimizer = torch.optim.AdamW([
    param for module in lora_modules.values() 
    for param in module.parameters()
], lr=1e-4)

# Training loop would go here...
'''
    
    return integration_code


if __name__ == "__main__":
    # Educational demonstrations
    
    # 1. Basic LoRA layer analysis
    original_layer = nn.Linear(4096, 4096)
    lora_layer = LoRALayer(original_layer, rank=8, alpha=16)
    
    analysis = lora_layer.get_parameter_analysis()
    
    # 2. Transformer layer with LoRA
    transformer_layer = LoRATransformerLayer(
        d_model=512,
        num_heads=8, 
        rank=4,
        alpha=16,
        apply_to_attention=True,
        apply_to_ffn=True
    )
    
    transformer_analysis = transformer_layer.get_parameter_analysis()
    
    # 3. Efficiency demonstration
    efficiency_results = demonstrate_lora_efficiency()
    
    # 4. Training approach comparison
    comparison = compare_training_approaches("7B")
    
    # 5. Mathematical insights
    insights = lora_mathematical_intuition()
    
    # 6. Integration example
    integration_example = create_lora_integration_example()
    
    # These results can be used for educational analysis and visualization