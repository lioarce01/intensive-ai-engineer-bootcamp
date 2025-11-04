"""
Gradient Accumulation for Training with Large Effective Batch Sizes

Gradient accumulation allows training with large effective batch sizes
on hardware with limited memory by accumulating gradients over multiple
forward/backward passes before updating weights.

Example: With gradient_accumulation_steps=4 and batch_size=8,
effective batch size = 32
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
from tqdm import tqdm


class GradientAccumulationTrainer:
    """
    Trainer with gradient accumulation support.

    Enables training with large effective batch sizes on limited hardware.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: Optional[float] = 1.0,
        device: str = 'cuda'
    ):
        """
        Initialize gradient accumulation trainer.

        Args:
            model: PyTorch model
            optimizer: Optimizer
            gradient_accumulation_steps: Number of steps to accumulate gradients
            max_grad_norm: Maximum gradient norm for clipping (None to disable)
            device: Device to train on
        """
        self.model = model
        self.optimizer = optimizer
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.device = device

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        step: int
    ) -> Dict[str, float]:
        """
        Perform a single training step with gradient accumulation.

        Args:
            batch: Dictionary containing 'input_ids' and 'labels'
            step: Current training step

        Returns:
            Dictionary with loss and other metrics
        """
        # Get data
        input_ids = batch['input_ids'].to(self.device)
        labels = batch.get('labels', input_ids).to(self.device)

        # Forward pass
        outputs = self.model(input_ids, labels=labels)

        # Get loss
        if isinstance(outputs, torch.Tensor):
            loss = outputs
        else:
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]

        # Scale loss by accumulation steps
        loss = loss / self.gradient_accumulation_steps

        # Backward pass
        loss.backward()

        # Only update weights every gradient_accumulation_steps
        if (step + 1) % self.gradient_accumulation_steps == 0:
            # Clip gradients if specified
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )

            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()

        return {
            'loss': loss.item() * self.gradient_accumulation_steps
        }

    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch with gradient accumulation.

        Args:
            dataloader: Training data loader
            epoch: Current epoch number

        Returns:
            Dictionary with epoch metrics
        """
        self.model.train()
        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Epoch {epoch}"
        )

        for step, batch in progress_bar:
            metrics = self.training_step(batch, step)

            total_loss += metrics['loss']
            num_batches += 1

            # Update progress bar
            avg_loss = total_loss / num_batches
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})

        return {
            'train_loss': total_loss / num_batches,
            'num_batches': num_batches
        }


def calculate_effective_batch_size(
    batch_size: int,
    gradient_accumulation_steps: int,
    num_gpus: int = 1
) -> int:
    """
    Calculate effective batch size.

    Args:
        batch_size: Per-device batch size
        gradient_accumulation_steps: Number of accumulation steps
        num_gpus: Number of GPUs

    Returns:
        Effective batch size
    """
    return batch_size * gradient_accumulation_steps * num_gpus


def adjust_learning_rate_for_batch_size(
    base_lr: float,
    base_batch_size: int,
    actual_batch_size: int,
    scaling_rule: str = 'linear'
) -> float:
    """
    Adjust learning rate based on batch size.

    Args:
        base_lr: Learning rate for base batch size
        base_batch_size: Base batch size
        actual_batch_size: Actual (effective) batch size
        scaling_rule: 'linear' or 'sqrt'

    Returns:
        Adjusted learning rate
    """
    scale_factor = actual_batch_size / base_batch_size

    if scaling_rule == 'linear':
        return base_lr * scale_factor
    elif scaling_rule == 'sqrt':
        return base_lr * (scale_factor ** 0.5)
    else:
        raise ValueError(f"Unknown scaling rule: {scaling_rule}")


class MemoryEfficientTrainer(GradientAccumulationTrainer):
    """
    Enhanced trainer with additional memory optimizations.

    Combines gradient accumulation with other memory-saving techniques.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: Optional[float] = 1.0,
        device: str = 'cuda',
        empty_cache_steps: Optional[int] = None
    ):
        """
        Initialize memory-efficient trainer.

        Args:
            empty_cache_steps: Empty CUDA cache every N steps (None to disable)
        """
        super().__init__(
            model=model,
            optimizer=optimizer,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
            device=device
        )
        self.empty_cache_steps = empty_cache_steps

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        step: int
    ) -> Dict[str, float]:
        """Training step with memory optimizations."""

        # Standard training step
        metrics = super().training_step(batch, step)

        # Periodically empty CUDA cache
        if self.empty_cache_steps and (step + 1) % self.empty_cache_steps == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return metrics


# Example usage and testing
def example_usage():
    """Demonstrate gradient accumulation."""

    # Create dummy model
    class SimpleModel(nn.Module):
        def __init__(self, vocab_size=10000, hidden_size=512):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.lm_head = nn.Linear(hidden_size, vocab_size)

        def forward(self, input_ids, labels=None):
            hidden = self.embedding(input_ids)
            logits = self.lm_head(hidden)

            loss = None
            if labels is not None:
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )

            return type('Output', (), {'loss': loss, 'logits': logits})()

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimpleModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Create trainer with gradient accumulation
    trainer = GradientAccumulationTrainer(
        model=model,
        optimizer=optimizer,
        gradient_accumulation_steps=4,  # Effective batch size = batch_size * 4
        max_grad_norm=1.0,
        device=device
    )

    # Create dummy dataset
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 100

        def __getitem__(self, idx):
            return {
                'input_ids': torch.randint(0, 10000, (128,)),
                'labels': torch.randint(0, 10000, (128,))
            }

    dataset = DummyDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)

    # Train for one epoch
    print("Training with gradient accumulation...")
    print(f"Batch size: 8")
    print(f"Gradient accumulation steps: 4")
    print(f"Effective batch size: {calculate_effective_batch_size(8, 4)}")
    print()

    metrics = trainer.train_epoch(dataloader, epoch=1)

    print("\nTraining complete!")
    print(f"Average loss: {metrics['train_loss']:.4f}")


if __name__ == "__main__":
    example_usage()
