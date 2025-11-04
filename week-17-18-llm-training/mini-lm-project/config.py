"""
Training Configuration for Mini-LM
"""

from dataclasses import dataclass, asdict
import json
from pathlib import Path


@dataclass
class TrainingConfig:
    """Configuration for training Mini-LM."""

    # Model
    model_name: str = 'mini-50m'

    # Data
    train_data_path: str = 'data/tokenized/train_tokenized.jsonl'
    val_data_path: str = 'data/tokenized/val_tokenized.jsonl'
    max_seq_length: int = 1024

    # Training
    num_epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 500

    # Optimization
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = False
    optimizer: str = 'adamw'  # 'adamw' or 'sgd'

    # Evaluation
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100

    # Paths
    output_dir: str = 'outputs/mini-lm'
    checkpoint_dir: str = 'checkpoints'
    log_dir: str = 'logs'

    # Distributed
    local_rank: int = -1

    # Misc
    seed: int = 42
    num_workers: int = 4

    def save(self, path: str):
        """Save config to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str):
        """Load config from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


# Predefined training configurations
TRAINING_CONFIGS = {
    'small_fast': TrainingConfig(
        model_name='mini-50m',
        batch_size=16,
        gradient_accumulation_steps=2,
        num_epochs=3,
        learning_rate=5e-4
    ),
    'medium': TrainingConfig(
        model_name='mini-100m',
        batch_size=8,
        gradient_accumulation_steps=4,
        num_epochs=5,
        learning_rate=3e-4,
        use_gradient_checkpointing=True
    ),
    'large': TrainingConfig(
        model_name='mini-250m',
        batch_size=4,
        gradient_accumulation_steps=8,
        num_epochs=10,
        learning_rate=2e-4,
        use_gradient_checkpointing=True
    )
}


def get_training_config(config_name: str = 'small_fast') -> TrainingConfig:
    """Get predefined training configuration."""
    if config_name not in TRAINING_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}")
    return TRAINING_CONFIGS[config_name]
