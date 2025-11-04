"""
Evaluation Pipeline for Language Models

Comprehensive evaluation combining perplexity, BLEU, ROUGE, and custom metrics.
"""

import torch
from typing import Dict, List, Optional, Union
from pathlib import Path
import json
from tqdm import tqdm

from perplexity import PerplexityCalculator, ValidationPerplexity
from bleu_rouge import BLEUCalculator, ROUGECalculator
from custom_metrics import (
    DiversityMetrics,
    AccuracyMetrics,
    CoherenceMetrics,
    EntropyMetrics,
    ComprehensiveEvaluator
)


class LMEvaluator:
    """
    Comprehensive Language Model Evaluator.

    Combines multiple metrics to provide a complete picture of model performance.
    """

    def __init__(
        self,
        ignore_index: int = -100,
        calculate_bleu: bool = True,
        calculate_rouge: bool = True
    ):
        """
        Initialize evaluator.

        Args:
            ignore_index: Token index to ignore (padding)
            calculate_bleu: Whether to calculate BLEU scores
            calculate_rouge: Whether to calculate ROUGE scores
        """
        self.ignore_index = ignore_index
        self.calculate_bleu = calculate_bleu
        self.calculate_rouge = calculate_rouge

        # Initialize metric calculators
        self.perplexity_calc = PerplexityCalculator(ignore_index=ignore_index)
        self.bleu_calc = BLEUCalculator() if calculate_bleu else None
        self.rouge_calc = ROUGECalculator() if calculate_rouge else None
        self.comprehensive_eval = ComprehensiveEvaluator(ignore_index=ignore_index)

    def evaluate_batch(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Evaluate a single batch.

        Args:
            logits: Model output logits [batch_size, seq_len, vocab_size]
            targets: Target token IDs [batch_size, seq_len]

        Returns:
            Dictionary with all metrics
        """
        metrics = {}

        # Perplexity
        perplexity = self.perplexity_calc.calculate_from_logits(logits, targets)
        metrics['perplexity'] = perplexity

        # Token accuracy and other metrics
        batch_metrics = self.comprehensive_eval.evaluate_batch(logits, targets)
        metrics.update(batch_metrics)

        return metrics

    def evaluate_generation(
        self,
        generated_texts: List[str],
        reference_texts: List[str],
        tokenizer: Optional[object] = None
    ) -> Dict[str, float]:
        """
        Evaluate generated text against references.

        Args:
            generated_texts: List of generated text strings
            reference_texts: List of reference text strings
            tokenizer: Optional tokenizer for detokenization

        Returns:
            Dictionary with generation metrics
        """
        all_bleu_scores = []
        all_rouge_scores = []
        all_diversity_scores = []

        for gen_text, ref_text in zip(generated_texts, reference_texts):
            # Tokenize
            gen_tokens = gen_text.split()
            ref_tokens = ref_text.split()

            # BLEU
            if self.calculate_bleu and self.bleu_calc:
                bleu_result = self.bleu_calc.calculate_bleu(gen_tokens, [ref_tokens])
                all_bleu_scores.append(bleu_result['bleu'])

            # ROUGE
            if self.calculate_rouge and self.rouge_calc:
                rouge_result = self.rouge_calc.calculate_rouge(gen_tokens, ref_tokens)
                all_rouge_scores.append(rouge_result)

            # Diversity
            diversity = DiversityMetrics.calculate_all(gen_tokens)
            all_diversity_scores.append(diversity)

        # Aggregate metrics
        metrics = {}

        # Average BLEU
        if all_bleu_scores:
            metrics['bleu'] = sum(all_bleu_scores) / len(all_bleu_scores)

        # Average ROUGE
        if all_rouge_scores:
            for rouge_type in ['rouge-1', 'rouge-2', 'rouge-l']:
                f1_scores = [score[rouge_type]['f1'] for score in all_rouge_scores]
                metrics[f'{rouge_type}_f1'] = sum(f1_scores) / len(f1_scores)

        # Average diversity
        if all_diversity_scores:
            for key in all_diversity_scores[0].keys():
                values = [score[key] for score in all_diversity_scores]
                metrics[key] = sum(values) / len(values)

        return metrics

    def evaluate_model(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: str = 'cuda',
        max_batches: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset.

        Args:
            model: PyTorch model
            dataloader: DataLoader with evaluation data
            device: Device to run evaluation on
            max_batches: Maximum number of batches to evaluate (None = all)

        Returns:
            Dictionary with aggregated metrics
        """
        model.eval()

        val_perplexity = ValidationPerplexity()
        all_metrics = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
                if max_batches and batch_idx >= max_batches:
                    break

                # Get inputs and targets
                input_ids = batch['input_ids'].to(device)
                targets = batch.get('labels', input_ids).to(device)

                # Forward pass
                outputs = model(input_ids)
                logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits

                # Update perplexity tracker
                val_perplexity.update(logits, targets, ignore_index=self.ignore_index)

                # Calculate batch metrics
                batch_metrics = self.evaluate_batch(logits, targets)
                all_metrics.append(batch_metrics)

        # Aggregate metrics
        final_metrics = val_perplexity.compute()

        # Average other metrics
        for key in all_metrics[0].keys():
            if key != 'perplexity':  # Already computed by val_perplexity
                values = [m[key] for m in all_metrics]
                final_metrics[f'avg_{key}'] = sum(values) / len(values)

        return final_metrics


class EvaluationLogger:
    """Log evaluation results to file."""

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "evaluation_log.jsonl"

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        split: str = 'val'
    ):
        """
        Log metrics to file.

        Args:
            metrics: Dictionary of metrics
            step: Training step
            split: Dataset split (train/val/test)
        """
        log_entry = {
            'step': step,
            'split': split,
            **metrics
        }

        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def load_logs(self) -> List[Dict]:
        """Load all logged metrics."""
        logs = []
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                for line in f:
                    logs.append(json.loads(line))
        return logs

    def get_best_metric(
        self,
        metric_name: str,
        minimize: bool = True
    ) -> Dict:
        """
        Get best value for a specific metric.

        Args:
            metric_name: Name of metric
            minimize: Whether lower is better

        Returns:
            Dictionary with best metric entry
        """
        logs = self.load_logs()

        if not logs:
            return {}

        if minimize:
            best_entry = min(logs, key=lambda x: x.get(metric_name, float('inf')))
        else:
            best_entry = max(logs, key=lambda x: x.get(metric_name, float('-inf')))

        return best_entry


def run_comprehensive_evaluation(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda',
    output_file: Optional[str] = None
) -> Dict[str, float]:
    """
    Run comprehensive evaluation and optionally save results.

    Args:
        model: PyTorch model
        dataloader: DataLoader with evaluation data
        device: Device to run on
        output_file: Optional file to save results

    Returns:
        Dictionary with all metrics
    """
    print("Running comprehensive evaluation...")
    print("=" * 60)

    # Initialize evaluator
    evaluator = LMEvaluator(
        ignore_index=-100,
        calculate_bleu=False,  # Set to True if you have references
        calculate_rouge=False
    )

    # Run evaluation
    metrics = evaluator.evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device
    )

    # Print results
    print("\nEvaluation Results:")
    print("-" * 60)
    for metric, value in sorted(metrics.items()):
        print(f"  {metric}: {value:.4f}")

    # Save to file if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dumps(metrics, indent=2, f)

        print(f"\nResults saved to {output_file}")

    print("=" * 60)

    return metrics


# Example usage
def example_usage():
    """Demonstrate evaluation pipeline."""

    # Create dummy model and data
    class DummyModel(torch.nn.Module):
        def __init__(self, vocab_size=32000):
            super().__init__()
            self.embedding = torch.nn.Embedding(vocab_size, 512)
            self.lm_head = torch.nn.Linear(512, vocab_size)

        def forward(self, input_ids):
            hidden = self.embedding(input_ids)
            logits = self.lm_head(hidden)
            return logits

    # Create dummy dataset
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=100, seq_len=128, vocab_size=32000):
            self.num_samples = num_samples
            self.seq_len = seq_len
            self.vocab_size = vocab_size

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            return {
                'input_ids': torch.randint(0, self.vocab_size, (self.seq_len,)),
                'labels': torch.randint(0, self.vocab_size, (self.seq_len,))
            }

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DummyModel().to(device)
    dataset = DummyDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)

    # Run evaluation
    metrics = run_comprehensive_evaluation(
        model=model,
        dataloader=dataloader,
        device=device,
        output_file='results/evaluation_results.json'
    )


if __name__ == "__main__":
    example_usage()
