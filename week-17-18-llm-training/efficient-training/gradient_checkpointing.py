"""
Gradient Checkpointing for Memory-Efficient Training

Gradient checkpointing trades compute for memory by:
- Not storing intermediate activations during forward pass
- Recomputing them during backward pass when needed

Memory savings: ~40-50%
Compute overhead: ~20-30%

Ideal for training large models with limited GPU memory.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Optional, Dict, Callable
import functools


def make_checkpointed(module: nn.Module) -> nn.Module:
    """
    Wrap a module to use gradient checkpointing.

    Args:
        module: PyTorch module

    Returns:
        Wrapped module with checkpointing
    """
    original_forward = module.forward

    @functools.wraps(original_forward)
    def checkpointed_forward(*args, **kwargs):
        return checkpoint(original_forward, *args, use_reentrant=False, **kwargs)

    module.forward = checkpointed_forward
    return module


class CheckpointedTransformerBlock(nn.Module):
    """
    Example transformer block with gradient checkpointing.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        use_checkpointing: bool = False
    ):
        super().__init__()
        self.use_checkpointing = use_checkpointing

        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.norm2 = nn.LayerNorm(hidden_size)

    def _forward(self, x):
        """Actual forward pass logic."""
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # MLP
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)

        return x

    def forward(self, x):
        if self.use_checkpointing and self.training:
            return checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class CheckpointedModel(nn.Module):
    """
    Example model with selective gradient checkpointing.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        checkpoint_frequency: int = 2  # Checkpoint every N layers
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        self.layers = nn.ModuleList([
            CheckpointedTransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                use_checkpointing=(i % checkpoint_frequency == 0)  # Selective checkpointing
            )
            for i in range(num_layers)
        ])

        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, labels=None):
        x = self.embedding(input_ids)

        for layer in self.layers:
            x = layer(x)

        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )

        return type('Output', (), {'loss': loss, 'logits': logits})()


def enable_gradient_checkpointing(model: nn.Module, modules_to_checkpoint: list = None):
    """
    Enable gradient checkpointing for specific modules in a model.

    Args:
        model: PyTorch model
        modules_to_checkpoint: List of module names/types to checkpoint (None = all)
    """
    if modules_to_checkpoint is None:
        # Checkpoint all transformer/attention blocks
        modules_to_checkpoint = ['TransformerBlock', 'AttentionBlock', 'Block']

    for name, module in model.named_modules():
        module_type = module.__class__.__name__
        if module_type in modules_to_checkpoint:
            make_checkpointed(module)
            print(f"Enabled gradient checkpointing for: {name} ({module_type})")


def benchmark_gradient_checkpointing():
    """
    Benchmark memory usage with and without gradient checkpointing.
    """
    import time

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 8
    seq_len = 512

    configs = {
        'without_checkpointing': {'checkpoint_frequency': 999},
        'with_checkpointing': {'checkpoint_frequency': 1}
    }

    results = {}

    for name, config in configs.items():
        print(f"\nBenchmarking: {name}")
        print("=" * 60)

        # Create model
        model = CheckpointedModel(
            vocab_size=32000,
            hidden_size=768,
            num_layers=12,
            **config
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

        # Dummy data
        input_ids = torch.randint(0, 32000, (batch_size, seq_len)).to(device)
        labels = torch.randint(0, 32000, (batch_size, seq_len)).to(device)

        # Reset memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        # Training step
        model.train()

        start_time = time.time()

        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed_time = time.time() - start_time

        # Memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.max_memory_allocated() / 1e9  # GB
        else:
            memory_allocated = 0

        results[name] = {
            'time': elapsed_time,
            'memory_gb': memory_allocated
        }

        print(f"Time: {elapsed_time:.3f}s")
        print(f"Memory: {memory_allocated:.2f} GB")

        # Cleanup
        del model, optimizer, input_ids, labels
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Print comparison
    if len(results) == 2:
        without = results['without_checkpointing']
        with_cp = results['with_checkpointing']

        memory_savings = (1 - with_cp['memory_gb'] / without['memory_gb']) * 100
        time_overhead = (with_cp['time'] / without['time'] - 1) * 100

        print("\n" + "=" * 60)
        print("Comparison:")
        print(f"  Memory savings: {memory_savings:.1f}%")
        print(f"  Time overhead: {time_overhead:.1f}%")
        print("=" * 60)


# Example usage
def example_usage():
    """Demonstrate gradient checkpointing."""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create model with checkpointing
    model = CheckpointedModel(
        vocab_size=10000,
        hidden_size=256,
        num_layers=6,
        checkpoint_frequency=2  # Checkpoint every 2 layers
    ).to(device)

    print(f"Model created with gradient checkpointing")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Dummy data
    input_ids = torch.randint(0, 10000, (4, 128)).to(device)
    labels = torch.randint(0, 10000, (4, 128)).to(device)

    # Forward and backward
    model.train()
    outputs = model(input_ids, labels=labels)
    loss = outputs.loss

    print(f"\nLoss: {loss.item():.4f}")

    loss.backward()
    print("Backward pass completed successfully!")


if __name__ == "__main__":
    print("Gradient Checkpointing Example")
    print("=" * 60)
    example_usage()

    if torch.cuda.is_available():
        print("\n\nRunning benchmark...")
        benchmark_gradient_checkpointing()
