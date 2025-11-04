"""
Mini Language Model Architecture

GPT-style decoder-only transformer for language modeling.
Designed to be trained from scratch on ~100M tokens.
"""

import torch
import torch.nn as nn
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for Mini-LM model."""

    vocab_size: int = 32000
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_ratio: int = 4
    max_position_embeddings: int = 1024
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    use_cache: bool = False


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.hidden_size % config.num_heads == 0

        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.hidden_size = config.hidden_size

        # QKV projections
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # Compute QKV
        qkv = self.qkv(hidden_states)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask
        if attention_mask is None:
            # Create causal mask
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=hidden_states.device),
                diagonal=1
            ).bool()
            scores = scores.masked_fill(causal_mask, float('-inf'))
        else:
            scores = scores + attention_mask

        # Softmax and apply to values
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        attn_output = self.out_proj(attn_output)

        return attn_output


class MLP(nn.Module):
    """Feed-forward network."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.hidden_size * config.mlp_ratio)
        self.fc2 = nn.Linear(config.hidden_size * config.mlp_ratio, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = nn.functional.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class MiniLM(nn.Module):
    """Mini Language Model - GPT-style decoder-only transformer."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token and position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size
        )
        self.dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])

        # Final layer norm and LM head
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Weight tying (share weights between embedding and lm_head)
        self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ):
        batch_size, seq_len = input_ids.shape

        # Get embeddings
        token_embeds = self.token_embedding(input_ids)

        # Position IDs
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeds = self.position_embedding(position_ids)

        # Combine embeddings
        hidden_states = token_embeds + position_embeds
        hidden_states = self.dropout(hidden_states)

        # Apply transformer blocks
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask)

        # Final layer norm
        hidden_states = self.ln_f(hidden_states)

        # LM head
        logits = self.lm_head(hidden_states)

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )

        return type('Output', (), {
            'loss': loss,
            'logits': logits,
            'hidden_states': hidden_states
        })()

    def get_num_params(self, non_embedding: bool = False):
        """
        Return the number of parameters in the model.

        Args:
            non_embedding: Exclude embedding parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.position_embedding.weight.numel()
            n_params -= self.token_embedding.weight.numel()
        return n_params


# Predefined configurations
MINI_LM_CONFIGS = {
    'mini-50m': ModelConfig(
        vocab_size=32000,
        hidden_size=512,
        num_layers=8,
        num_heads=8,
        max_position_embeddings=1024
    ),
    'mini-100m': ModelConfig(
        vocab_size=32000,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        max_position_embeddings=1024
    ),
    'mini-250m': ModelConfig(
        vocab_size=32000,
        hidden_size=1024,
        num_layers=16,
        num_heads=16,
        max_position_embeddings=2048
    )
}


def create_mini_lm(config_name: str = 'mini-50m') -> MiniLM:
    """
    Create a Mini-LM model with predefined configuration.

    Args:
        config_name: One of 'mini-50m', 'mini-100m', 'mini-250m'

    Returns:
        MiniLM model
    """
    if config_name not in MINI_LM_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Choose from {list(MINI_LM_CONFIGS.keys())}")

    config = MINI_LM_CONFIGS[config_name]
    model = MiniLM(config)

    print(f"Created {config_name} model:")
    print(f"  Total parameters: {model.get_num_params() / 1e6:.2f}M")
    print(f"  Non-embedding parameters: {model.get_num_params(non_embedding=True) / 1e6:.2f}M")

    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing Mini-LM Models")
    print("=" * 60)

    for config_name in MINI_LM_CONFIGS.keys():
        print(f"\n{config_name}:")
        model = create_mini_lm(config_name)

        # Test forward pass
        batch_size = 2
        seq_len = 128
        input_ids = torch.randint(0, 32000, (batch_size, seq_len))
        labels = torch.randint(0, 32000, (batch_size, seq_len))

        outputs = model(input_ids, labels=labels)
        print(f"  Loss: {outputs.loss.item():.4f}")
        print(f"  Logits shape: {outputs.logits.shape}")
