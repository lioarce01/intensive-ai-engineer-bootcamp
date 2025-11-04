"""
Perplexity Calculation for Language Models

Perplexity is the primary metric for evaluating language models during training.
It measures how well a probability model predicts a sample.

Lower perplexity = better model
Perplexity = exp(cross_entropy_loss)
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Dict
import math


class PerplexityCalculator:
    """Calculate perplexity for language models."""

    def __init__(self, ignore_index: int = -100):
        """
        Initialize perplexity calculator.

        Args:
            ignore_index: Token index to ignore (typically padding token)
        """
        self.ignore_index = ignore_index

    def calculate_from_logits(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: Optional[int] = None
    ) -> float:
        """
        Calculate perplexity from model logits.

        Args:
            logits: Model output logits [batch_size, seq_len, vocab_size]
            targets: Target token IDs [batch_size, seq_len]
            ignore_index: Token index to ignore

        Returns:
            Perplexity value
        """
        if ignore_index is None:
            ignore_index = self.ignore_index

        # Reshape for cross entropy
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)

        # Calculate cross entropy loss
        loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=ignore_index,
            reduction='mean'
        )

        # Convert to perplexity
        perplexity = torch.exp(loss).item()

        return perplexity

    def calculate_from_loss(self, loss: float) -> float:
        """
        Calculate perplexity from cross entropy loss.

        Args:
            loss: Cross entropy loss value

        Returns:
            Perplexity value
        """
        return math.exp(loss)

    def calculate_per_token(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: Optional[int] = None
    ) -> torch.Tensor:
        """
        Calculate per-token perplexity.

        Args:
            logits: Model output logits [batch_size, seq_len, vocab_size]
            targets: Target token IDs [batch_size, seq_len]
            ignore_index: Token index to ignore

        Returns:
            Per-token perplexity [batch_size, seq_len]
        """
        if ignore_index is None:
            ignore_index = self.ignore_index

        batch_size, seq_len, vocab_size = logits.shape

        # Calculate per-token cross entropy
        log_probs = F.log_softmax(logits, dim=-1)
        target_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=targets.unsqueeze(-1)
        ).squeeze(-1)

        # Calculate per-token perplexity
        per_token_perplexity = torch.exp(-target_log_probs)

        # Mask ignored tokens
        if ignore_index is not None:
            mask = (targets != ignore_index)
            per_token_perplexity = per_token_perplexity * mask

        return per_token_perplexity


class ValidationPerplexity:
    """Track and analyze perplexity during validation."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset accumulated statistics."""
        self.total_loss = 0.0
        self.total_tokens = 0
        self.perplexities = []

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int = -100
    ):
        """
        Update with a batch of predictions.

        Args:
            logits: Model output logits
            targets: Target token IDs
            ignore_index: Token index to ignore
        """
        # Calculate loss
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)

        loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=ignore_index,
            reduction='sum'
        )

        # Count valid tokens
        num_tokens = (targets != ignore_index).sum().item()

        # Accumulate
        self.total_loss += loss.item()
        self.total_tokens += num_tokens

        # Calculate batch perplexity
        batch_loss = loss.item() / num_tokens if num_tokens > 0 else 0
        batch_perplexity = math.exp(batch_loss)
        self.perplexities.append(batch_perplexity)

    def compute(self) -> Dict[str, float]:
        """
        Compute final perplexity statistics.

        Returns:
            Dictionary with perplexity metrics
        """
        if self.total_tokens == 0:
            return {
                'perplexity': float('inf'),
                'loss': float('inf'),
                'min_perplexity': float('inf'),
                'max_perplexity': float('inf'),
                'avg_perplexity': float('inf')
            }

        # Overall perplexity from accumulated loss
        avg_loss = self.total_loss / self.total_tokens
        perplexity = math.exp(avg_loss)

        # Statistics from per-batch perplexities
        min_ppl = min(self.perplexities) if self.perplexities else float('inf')
        max_ppl = max(self.perplexities) if self.perplexities else float('inf')
        avg_ppl = sum(self.perplexities) / len(self.perplexities) if self.perplexities else float('inf')

        return {
            'perplexity': perplexity,
            'loss': avg_loss,
            'min_perplexity': min_ppl,
            'max_perplexity': max_ppl,
            'avg_batch_perplexity': avg_ppl,
            'total_tokens': self.total_tokens
        }


def analyze_perplexity_by_position(
    logits: torch.Tensor,
    targets: torch.Tensor,
    max_positions: int = 1024
) -> List[float]:
    """
    Analyze how perplexity varies by position in sequence.

    Args:
        logits: Model output logits [batch_size, seq_len, vocab_size]
        targets: Target token IDs [batch_size, seq_len]
        max_positions: Maximum number of positions to analyze

    Returns:
        List of perplexity values for each position
    """
    batch_size, seq_len, vocab_size = logits.shape
    seq_len = min(seq_len, max_positions)

    position_perplexities = []

    for pos in range(seq_len):
        # Get logits and targets for this position
        pos_logits = logits[:, pos, :]  # [batch_size, vocab_size]
        pos_targets = targets[:, pos]    # [batch_size]

        # Calculate loss for this position
        loss = F.cross_entropy(pos_logits, pos_targets, reduction='mean')
        perplexity = math.exp(loss.item())

        position_perplexities.append(perplexity)

    return position_perplexities


def compare_perplexities(
    model_outputs: Dict[str, tuple[torch.Tensor, torch.Tensor]],
    ignore_index: int = -100
) -> Dict[str, float]:
    """
    Compare perplexities from multiple models.

    Args:
        model_outputs: Dict mapping model names to (logits, targets) tuples
        ignore_index: Token index to ignore

    Returns:
        Dict mapping model names to perplexity values
    """
    calculator = PerplexityCalculator(ignore_index=ignore_index)
    results = {}

    for model_name, (logits, targets) in model_outputs.items():
        perplexity = calculator.calculate_from_logits(logits, targets)
        results[model_name] = perplexity

    return results


# Example usage
def example_usage():
    """Demonstrate perplexity calculation."""

    # Create dummy model outputs
    batch_size = 4
    seq_len = 128
    vocab_size = 32000

    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Calculate perplexity
    calculator = PerplexityCalculator()
    perplexity = calculator.calculate_from_logits(logits, targets)

    print(f"Perplexity: {perplexity:.2f}")

    # Track validation perplexity
    val_tracker = ValidationPerplexity()

    # Simulate multiple validation batches
    for _ in range(10):
        batch_logits = torch.randn(batch_size, seq_len, vocab_size)
        batch_targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        val_tracker.update(batch_logits, batch_targets)

    # Get final metrics
    metrics = val_tracker.compute()
    print("\nValidation Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.2f}")

    # Analyze by position
    position_ppls = analyze_perplexity_by_position(logits, targets, max_positions=20)
    print("\nPerplexity by position (first 20):")
    for i, ppl in enumerate(position_ppls):
        print(f"  Position {i}: {ppl:.2f}")


if __name__ == "__main__":
    example_usage()
