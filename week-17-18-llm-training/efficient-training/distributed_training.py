"""
Distributed Training for Multi-GPU Training

Implements:
- Data Parallel (DP): Simple multi-GPU training
- Distributed Data Parallel (DDP): More efficient, recommended for multi-node
- Fully Sharded Data Parallel (FSDP): For very large models

DDP is the recommended approach for most use cases.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import os
from typing import Optional, Dict


def setup_distributed(backend: str = 'nccl'):
    """
    Setup distributed training environment.

    Args:
        backend: 'nccl' for GPU, 'gloo' for CPU
    """
    # Initialize process group
    dist.init_process_group(backend=backend)

    # Set device for this process
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)

    return local_rank


def cleanup_distributed():
    """Cleanup distributed training."""
    dist.destroy_process_group()


class DistributedTrainer:
    """
    Trainer for distributed training with DDP.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        local_rank: int,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: Optional[float] = 1.0
    ):
        """
        Initialize distributed trainer.

        Args:
            model: PyTorch model (will be wrapped with DDP)
            optimizer: Optimizer
            local_rank: Local rank of this process
            gradient_accumulation_steps: Number of gradient accumulation steps
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.local_rank = local_rank
        self.device = f'cuda:{local_rank}'
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm

        # Move model to device
        self.model = model.to(self.device)

        # Wrap with DDP
        self.model = DDP(
            self.model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False  # Set True if you have unused parameters
        )

        self.optimizer = optimizer

        # Get world size
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        step: int
    ) -> Dict[str, float]:
        """
        Perform a single training step.

        Args:
            batch: Dictionary containing 'input_ids' and 'labels'
            step: Current training step

        Returns:
            Dictionary with metrics
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

        # Update weights every gradient_accumulation_steps
        if (step + 1) % self.gradient_accumulation_steps == 0:
            # Clip gradients
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

    def save_checkpoint(self, checkpoint_path: str, epoch: int, global_step: int):
        """
        Save checkpoint (only on rank 0).

        Args:
            checkpoint_path: Path to save checkpoint
            epoch: Current epoch
            global_step: Global training step
        """
        if self.rank == 0:
            checkpoint = {
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': self.model.module.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
        """
        # Map to correct device
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.module.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint.get('epoch', 0), checkpoint.get('global_step', 0)


def create_distributed_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """
    Create DataLoader with DistributedSampler.

    Args:
        dataset: PyTorch dataset
        batch_size: Batch size per GPU
        shuffle: Whether to shuffle
        num_workers: Number of data loading workers

    Returns:
        DataLoader with DistributedSampler
    """
    sampler = DistributedSampler(
        dataset,
        shuffle=shuffle,
        drop_last=True  # Ensure all ranks have same number of batches
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


def calculate_total_batch_size(
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    world_size: int
) -> int:
    """
    Calculate total effective batch size.

    Args:
        per_device_batch_size: Batch size per device
        gradient_accumulation_steps: Number of accumulation steps
        world_size: Number of GPUs/processes

    Returns:
        Total effective batch size
    """
    return per_device_batch_size * gradient_accumulation_steps * world_size


# Example usage (run with torchrun)
def example_usage():
    """
    Example distributed training setup.

    Run with:
    torchrun --nproc_per_node=2 distributed_training.py
    """

    # Setup distributed
    local_rank = setup_distributed(backend='nccl')

    # Create model
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

    model = SimpleModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Create trainer
    trainer = DistributedTrainer(
        model=model,
        optimizer=optimizer,
        local_rank=local_rank,
        gradient_accumulation_steps=2,
        max_grad_norm=1.0
    )

    # Create dataset and dataloader
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 1000

        def __getitem__(self, idx):
            return {
                'input_ids': torch.randint(0, 10000, (128,)),
                'labels': torch.randint(0, 10000, (128,))
            }

    dataset = DummyDataset()
    dataloader = create_distributed_dataloader(
        dataset=dataset,
        batch_size=8,
        shuffle=True
    )

    # Print info on rank 0
    if trainer.rank == 0:
        world_size = dist.get_world_size()
        total_batch_size = calculate_total_batch_size(
            per_device_batch_size=8,
            gradient_accumulation_steps=2,
            world_size=world_size
        )
        print(f"Distributed Training Setup:")
        print(f"  World size: {world_size}")
        print(f"  Per-device batch size: 8")
        print(f"  Gradient accumulation steps: 2")
        print(f"  Total batch size: {total_batch_size}")

    # Training loop
    for epoch in range(1):
        # Set epoch for sampler (important for proper shuffling)
        dataloader.sampler.set_epoch(epoch)

        for step, batch in enumerate(dataloader):
            metrics = trainer.training_step(batch, step)

            if step % 10 == 0 and trainer.rank == 0:
                print(f"Step {step}, Loss: {metrics['loss']:.4f}")

            if step >= 50:  # Short demo
                break

    # Save checkpoint
    trainer.save_checkpoint('checkpoint.pt', epoch=1, global_step=50)

    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    print("Distributed Training Example")
    print("=" * 60)
    print("Run with: torchrun --nproc_per_node=2 distributed_training.py")
    print("=" * 60)

    example_usage()
