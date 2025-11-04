"""
Mixed Precision Training (FP16/BF16)

Mixed precision training uses lower precision (float16) for most operations
while keeping critical operations in float32. This provides:
- ~2x faster training
- ~2x memory reduction
- Minimal accuracy impact

Uses PyTorch's Automatic Mixed Precision (AMP) for easy integration.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Dict
from tqdm import tqdm


class MixedPrecisionTrainer:
    """
    Trainer with mixed precision (FP16) support using PyTorch AMP.

    Automatically handles:
    - Loss scaling to prevent underflow
    - Mixed precision forward/backward passes
    - Gradient unscaling before optimizer step
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        use_amp: bool = True,
        amp_dtype: torch.dtype = torch.float16,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: Optional[float] = 1.0,
        device: str = 'cuda'
    ):
        """
        Initialize mixed precision trainer.

        Args:
            model: PyTorch model
            optimizer: Optimizer
            use_amp: Whether to use automatic mixed precision
            amp_dtype: Data type for AMP (torch.float16 or torch.bfloat16)
            gradient_accumulation_steps: Number of gradient accumulation steps
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to train on
        """
        self.model = model
        self.optimizer = optimizer
        self.use_amp = use_amp and torch.cuda.is_available()
        self.amp_dtype = amp_dtype
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.device = device

        # Initialize gradient scaler for FP16
        self.scaler = GradScaler() if self.use_amp else None

        print(f"Mixed Precision Training: {self.use_amp}")
        if self.use_amp:
            print(f"  AMP dtype: {self.amp_dtype}")
            print(f"  Using GradScaler: {self.scaler is not None}")

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        step: int
    ) -> Dict[str, float]:
        """
        Perform a single training step with mixed precision.

        Args:
            batch: Dictionary containing 'input_ids' and 'labels'
            step: Current training step

        Returns:
            Dictionary with loss and other metrics
        """
        # Get data
        input_ids = batch['input_ids'].to(self.device)
        labels = batch.get('labels', input_ids).to(self.device)

        # Forward pass with autocast
        with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            outputs = self.model(input_ids, labels=labels)

            # Get loss
            if isinstance(outputs, torch.Tensor):
                loss = outputs
            else:
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]

            # Scale loss by accumulation steps
            loss = loss / self.gradient_accumulation_steps

        # Backward pass with gradient scaling
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Update weights every gradient_accumulation_steps
        if (step + 1) % self.gradient_accumulation_steps == 0:
            if self.use_amp:
                # Unscale gradients before clipping
                self.scaler.unscale_(self.optimizer)

            # Clip gradients
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )

            # Optimizer step with scaler
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()

        return {
            'loss': loss.item() * self.gradient_accumulation_steps,
            'scale': self.scaler.get_scale() if self.use_amp else 1.0
        }

    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch with mixed precision.

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
            postfix = {'loss': f'{avg_loss:.4f}'}
            if self.use_amp:
                postfix['scale'] = f"{metrics['scale']:.1f}"
            progress_bar.set_postfix(postfix)

        return {
            'train_loss': total_loss / num_batches,
            'num_batches': num_batches
        }


class BFloat16Trainer(MixedPrecisionTrainer):
    """
    Trainer using BFloat16 precision.

    BF16 advantages over FP16:
    - Same dynamic range as FP32 (no gradient scaling needed)
    - More stable training
    - Requires newer hardware (A100, H100)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: Optional[float] = 1.0,
        device: str = 'cuda'
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            use_amp=True,
            amp_dtype=torch.bfloat16,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
            device=device
        )
        # No gradient scaler needed for BF16
        self.scaler = None


def benchmark_precision_types(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_steps: int = 100
):
    """
    Benchmark different precision types.

    Args:
        model: Model to benchmark
        dataloader: Data loader
        num_steps: Number of steps to benchmark

    Returns:
        Dictionary with benchmark results
    """
    import time

    results = {}
    device = next(model.parameters()).device

    # Test configurations
    configs = {
        'fp32': {'use_amp': False, 'amp_dtype': torch.float32},
        'fp16': {'use_amp': True, 'amp_dtype': torch.float16},
    }

    # Add BF16 if supported
    if torch.cuda.is_bf16_supported():
        configs['bf16'] = {'use_amp': True, 'amp_dtype': torch.bfloat16}

    for name, config in configs.items():
        print(f"\nBenchmarking {name.upper()}...")

        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

        # Create trainer
        trainer = MixedPrecisionTrainer(
            model=model,
            optimizer=optimizer,
            **config,
            device=str(device)
        )

        # Warmup
        for i, batch in enumerate(dataloader):
            if i >= 10:
                break
            trainer.training_step(batch, i)

        # Benchmark
        torch.cuda.synchronize()
        start_time = time.time()

        for i, batch in enumerate(dataloader):
            if i >= num_steps:
                break
            trainer.training_step(batch, i)

        torch.cuda.synchronize()
        elapsed_time = time.time() - start_time

        # Calculate throughput
        steps_per_sec = num_steps / elapsed_time
        samples_per_sec = steps_per_sec * dataloader.batch_size

        # Memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.max_memory_allocated() / 1e9  # GB
            torch.cuda.reset_peak_memory_stats()
        else:
            memory_allocated = 0

        results[name] = {
            'time': elapsed_time,
            'steps_per_sec': steps_per_sec,
            'samples_per_sec': samples_per_sec,
            'memory_gb': memory_allocated
        }

        print(f"  Time: {elapsed_time:.2f}s")
        print(f"  Steps/sec: {steps_per_sec:.2f}")
        print(f"  Samples/sec: {samples_per_sec:.2f}")
        print(f"  Memory: {memory_allocated:.2f} GB")

    # Print comparison
    if 'fp32' in results and 'fp16' in results:
        speedup = results['fp32']['time'] / results['fp16']['time']
        memory_reduction = results['fp32']['memory_gb'] / results['fp16']['memory_gb']
        print(f"\nFP16 vs FP32:")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Memory reduction: {memory_reduction:.2f}x")

    return results


# Example usage
def example_usage():
    """Demonstrate mixed precision training."""

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

    # Create trainer with mixed precision
    trainer = MixedPrecisionTrainer(
        model=model,
        optimizer=optimizer,
        use_amp=True,
        amp_dtype=torch.float16,
        gradient_accumulation_steps=1,
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
    print("\nTraining with mixed precision...")
    metrics = trainer.train_epoch(dataloader, epoch=1)

    print("\nTraining complete!")
    print(f"Average loss: {metrics['train_loss']:.4f}")


if __name__ == "__main__":
    example_usage()
