"""
Training Script for Mini-LM

Complete training pipeline with:
- Mixed precision training
- Gradient accumulation
- Checkpointing
- Logging and evaluation
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from tqdm import tqdm
import sys
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from model import create_mini_lm
from config import TrainingConfig, get_training_config
from efficient_training.mixed_precision import MixedPrecisionTrainer


class TokenizedDataset(Dataset):
    """Dataset for tokenized JSONL files."""

    def __init__(self, data_path: str, max_length: int = 1024):
        self.data_path = Path(data_path)
        self.max_length = max_length
        self.samples = self._load_data()

    def _load_data(self):
        """Load tokenized data from JSONL."""
        samples = []
        if not self.data_path.exists():
            print(f"Warning: Data file not found: {self.data_path}")
            return samples

        with open(self.data_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                input_ids = data['input_ids'][:self.max_length]
                samples.append(input_ids)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.samples[idx], dtype=torch.long)
        return {'input_ids': input_ids, 'labels': input_ids.clone()}


def collate_fn(batch, pad_token_id=0):
    """Collate function with padding."""
    max_len = max(len(item['input_ids']) for item in batch)

    input_ids = []
    labels = []

    for item in batch:
        seq_len = len(item['input_ids'])
        padding_len = max_len - seq_len

        # Pad input_ids
        padded_input = torch.cat([
            item['input_ids'],
            torch.full((padding_len,), pad_token_id, dtype=torch.long)
        ])
        input_ids.append(padded_input)

        # Pad labels (use -100 for padding to ignore in loss)
        padded_label = torch.cat([
            item['labels'],
            torch.full((padding_len,), -100, dtype=torch.long)
        ])
        labels.append(padded_label)

    return {
        'input_ids': torch.stack(input_ids),
        'labels': torch.stack(labels)
    }


class Trainer:
    """Complete training pipeline."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Create output directories
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)

        # Save config
        config.save(Path(config.output_dir) / 'config.json')

        # Create model
        print("Creating model...")
        self.model = create_mini_lm(config.model_name).to(self.device)

        # Create optimizer
        self.optimizer = self._create_optimizer()

        # Create datasets
        print("Loading datasets...")
        self.train_dataset = TokenizedDataset(
            config.train_data_path,
            max_length=config.max_seq_length
        )
        self.val_dataset = TokenizedDataset(
            config.val_data_path,
            max_length=config.max_seq_length
        )

        print(f"  Train samples: {len(self.train_dataset)}")
        print(f"  Val samples: {len(self.val_dataset)}")

        # Create dataloaders
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=config.num_workers
        )

        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=config.num_workers
        )

        # Create mixed precision trainer
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            optimizer=self.optimizer,
            use_amp=config.use_mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            max_grad_norm=config.max_grad_norm,
            device=self.device
        )

        self.global_step = 0
        self.best_val_loss = float('inf')

    def _create_optimizer(self):
        """Create optimizer with weight decay."""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if 'bias' in name or 'norm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_grouped_parameters = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8
        )

        return optimizer

    def train(self):
        """Main training loop."""
        print("\nStarting training...")
        print("=" * 60)

        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            self.train_epoch(epoch)

            # Validation
            val_loss = self.evaluate()
            print(f"Validation Loss: {val_loss:.4f}")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pt')

        print("\nTraining complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(self.train_dataloader, desc=f"Training")

        for step, batch in enumerate(progress_bar):
            metrics = self.mp_trainer.training_step(batch, self.global_step)

            total_loss += metrics['loss']
            num_batches += 1
            self.global_step += 1

            # Logging
            if self.global_step % self.config.logging_steps == 0:
                avg_loss = total_loss / num_batches
                progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})

            # Evaluation
            if self.global_step % self.config.eval_steps == 0:
                val_loss = self.evaluate()
                print(f"\nStep {self.global_step} - Val Loss: {val_loss:.4f}")
                self.model.train()

            # Checkpointing
            if self.global_step % self.config.save_steps == 0:
                self.save_checkpoint(f'checkpoint_{self.global_step}.pt')

    @torch.no_grad()
    def evaluate(self):
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        for batch in tqdm(self.val_dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            outputs = self.model(input_ids, labels=labels)
            total_loss += outputs.loss.item()
            num_batches += 1

        return total_loss / num_batches

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint_path = Path(self.config.checkpoint_dir) / filename

        checkpoint = {
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")


def main():
    """Main training function."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='small_fast',
                        help='Training config name')
    parser.add_argument('--model', type=str, default=None,
                        help='Model size')
    args = parser.parse_args()

    # Load config
    config = get_training_config(args.config)

    if args.model:
        config.model_name = args.model

    print("Training Configuration:")
    print("=" * 60)
    print(f"  Model: {config.model_name}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"  Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Mixed precision: {config.use_mixed_precision}")
    print("=" * 60)

    # Create trainer and train
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
