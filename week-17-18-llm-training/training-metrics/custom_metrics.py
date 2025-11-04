"""
Custom Metrics for LLM Evaluation

This module implements additional metrics for evaluating LLMs beyond perplexity and BLEU/ROUGE:
- Diversity metrics (unique n-grams, repetition)
- Coherence metrics
- Semantic similarity
- Token-level accuracy
"""

import torch
import math
from typing import List, Dict, Set, Tuple
from collections import Counter
import numpy as np


class DiversityMetrics:
    """Measure diversity and repetition in generated text."""

    @staticmethod
    def unique_ngrams(tokens: List[str], n: int) -> float:
        """
        Calculate ratio of unique n-grams.

        Args:
            tokens: List of tokens
            n: N-gram size

        Returns:
            Ratio of unique n-grams to total n-grams
        """
        if len(tokens) < n:
            return 0.0

        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams.append(ngram)

        if not ngrams:
            return 0.0

        unique_ratio = len(set(ngrams)) / len(ngrams)
        return unique_ratio

    @staticmethod
    def distinct_n(tokens: List[str], n: int) -> float:
        """
        Calculate distinct-n metric (used in dialogue generation).

        Args:
            tokens: List of tokens
            n: N-gram size

        Returns:
            Distinct-n score
        """
        return DiversityMetrics.unique_ngrams(tokens, n)

    @staticmethod
    def repetition_score(tokens: List[str], n: int = 4) -> float:
        """
        Calculate repetition score (higher = more repetition).

        Args:
            tokens: List of tokens
            n: N-gram size to check

        Returns:
            Repetition score (0 = no repetition, 1 = highly repetitive)
        """
        if len(tokens) < n:
            return 0.0

        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams.append(ngram)

        if not ngrams:
            return 0.0

        # Count occurrences
        ngram_counts = Counter(ngrams)

        # Calculate repetition as 1 - (unique ratio)
        unique_ratio = len(ngram_counts) / len(ngrams)
        repetition = 1.0 - unique_ratio

        return repetition

    @staticmethod
    def calculate_all(tokens: List[str]) -> Dict[str, float]:
        """
        Calculate all diversity metrics.

        Args:
            tokens: List of tokens

        Returns:
            Dictionary with all diversity metrics
        """
        return {
            'distinct-1': DiversityMetrics.distinct_n(tokens, 1),
            'distinct-2': DiversityMetrics.distinct_n(tokens, 2),
            'distinct-3': DiversityMetrics.distinct_n(tokens, 3),
            'distinct-4': DiversityMetrics.distinct_n(tokens, 4),
            'repetition-2': DiversityMetrics.repetition_score(tokens, 2),
            'repetition-3': DiversityMetrics.repetition_score(tokens, 3),
            'repetition-4': DiversityMetrics.repetition_score(tokens, 4),
        }


class AccuracyMetrics:
    """Token-level accuracy metrics."""

    @staticmethod
    def token_accuracy(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int = -100
    ) -> float:
        """
        Calculate token-level accuracy.

        Args:
            predictions: Predicted token IDs [batch_size, seq_len]
            targets: Target token IDs [batch_size, seq_len]
            ignore_index: Index to ignore (e.g., padding)

        Returns:
            Accuracy as fraction of correct tokens
        """
        mask = (targets != ignore_index)
        correct = ((predictions == targets) & mask).sum().item()
        total = mask.sum().item()

        return correct / total if total > 0 else 0.0

    @staticmethod
    def top_k_accuracy(
        logits: torch.Tensor,
        targets: torch.Tensor,
        k: int = 5,
        ignore_index: int = -100
    ) -> float:
        """
        Calculate top-k accuracy.

        Args:
            logits: Model output logits [batch_size, seq_len, vocab_size]
            targets: Target token IDs [batch_size, seq_len]
            k: Number of top predictions to consider
            ignore_index: Index to ignore

        Returns:
            Top-k accuracy
        """
        # Get top-k predictions
        _, top_k_preds = torch.topk(logits, k, dim=-1)  # [batch_size, seq_len, k]

        # Expand targets for comparison
        targets_expanded = targets.unsqueeze(-1).expand_as(top_k_preds)

        # Check if target is in top-k
        mask = (targets != ignore_index).unsqueeze(-1)
        correct = ((top_k_preds == targets_expanded) & mask).any(dim=-1).sum().item()
        total = (targets != ignore_index).sum().item()

        return correct / total if total > 0 else 0.0

    @staticmethod
    def sequence_accuracy(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int = -100
    ) -> float:
        """
        Calculate sequence-level accuracy (entire sequence must be correct).

        Args:
            predictions: Predicted token IDs [batch_size, seq_len]
            targets: Target token IDs [batch_size, seq_len]
            ignore_index: Index to ignore

        Returns:
            Fraction of sequences that are entirely correct
        """
        mask = (targets != ignore_index)
        matches = (predictions == targets) | ~mask

        # Check if all tokens in each sequence match
        correct_sequences = matches.all(dim=1).sum().item()
        total_sequences = predictions.size(0)

        return correct_sequences / total_sequences if total_sequences > 0 else 0.0


class CoherenceMetrics:
    """Measure coherence in generated text."""

    @staticmethod
    def sentence_coherence(sentences: List[str]) -> float:
        """
        Calculate coherence between consecutive sentences.
        Uses a simple word overlap heuristic.

        Args:
            sentences: List of sentences

        Returns:
            Average coherence score
        """
        if len(sentences) < 2:
            return 1.0

        coherence_scores = []

        for i in range(len(sentences) - 1):
            sent1_words = set(sentences[i].lower().split())
            sent2_words = set(sentences[i + 1].lower().split())

            if not sent1_words or not sent2_words:
                continue

            # Calculate Jaccard similarity
            intersection = len(sent1_words & sent2_words)
            union = len(sent1_words | sent2_words)

            coherence = intersection / union if union > 0 else 0.0
            coherence_scores.append(coherence)

        return sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0


class LengthMetrics:
    """Metrics related to sequence length."""

    @staticmethod
    def average_length(sequences: List[List[int]]) -> float:
        """Calculate average sequence length."""
        if not sequences:
            return 0.0
        return sum(len(seq) for seq in sequences) / len(sequences)

    @staticmethod
    def length_distribution(
        sequences: List[List[int]],
        bins: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get distribution of sequence lengths.

        Args:
            sequences: List of token sequences
            bins: Number of bins for histogram

        Returns:
            (counts, bin_edges)
        """
        lengths = [len(seq) for seq in sequences]
        return np.histogram(lengths, bins=bins)


class EntropyMetrics:
    """Metrics based on entropy and information theory."""

    @staticmethod
    def token_entropy(
        logits: torch.Tensor,
        reduction: str = 'mean'
    ) -> float:
        """
        Calculate entropy of token distribution.

        Args:
            logits: Model output logits [batch_size, seq_len, vocab_size]
            reduction: How to reduce ('mean', 'sum', 'none')

        Returns:
            Entropy value
        """
        # Calculate probabilities
        probs = torch.softmax(logits, dim=-1)

        # Calculate entropy: -sum(p * log(p))
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)

        if reduction == 'mean':
            return entropy.mean().item()
        elif reduction == 'sum':
            return entropy.sum().item()
        else:
            return entropy

    @staticmethod
    def vocabulary_entropy(tokens: List[int]) -> float:
        """
        Calculate entropy of vocabulary usage.

        Args:
            tokens: List of token IDs

        Returns:
            Entropy value
        """
        if not tokens:
            return 0.0

        # Count token frequencies
        token_counts = Counter(tokens)
        total = len(tokens)

        # Calculate entropy
        entropy = 0.0
        for count in token_counts.values():
            prob = count / total
            entropy -= prob * math.log2(prob)

        return entropy


class ComprehensiveEvaluator:
    """Comprehensive evaluation combining multiple metrics."""

    def __init__(self, ignore_index: int = -100):
        self.ignore_index = ignore_index

    def evaluate_generation(
        self,
        generated_tokens: List[str],
        reference_tokens: List[str],
        logits: torch.Tensor = None
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation of generated text.

        Args:
            generated_tokens: Generated text tokens
            reference_tokens: Reference text tokens
            logits: Optional logits for entropy calculation

        Returns:
            Dictionary with all metrics
        """
        metrics = {}

        # Diversity metrics
        diversity = DiversityMetrics.calculate_all(generated_tokens)
        metrics.update(diversity)

        # Length metrics
        metrics['generated_length'] = len(generated_tokens)
        metrics['reference_length'] = len(reference_tokens)
        metrics['length_ratio'] = len(generated_tokens) / len(reference_tokens) if reference_tokens else 0

        # Entropy metrics (if logits provided)
        if logits is not None:
            metrics['token_entropy'] = EntropyMetrics.token_entropy(logits)

        # Vocabulary entropy
        # Convert tokens to IDs (placeholder - in practice use actual IDs)
        token_ids = [hash(t) % 10000 for t in generated_tokens]
        metrics['vocabulary_entropy'] = EntropyMetrics.vocabulary_entropy(token_ids)

        return metrics

    def evaluate_batch(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Evaluate a batch of predictions.

        Args:
            logits: Model output logits [batch_size, seq_len, vocab_size]
            targets: Target token IDs [batch_size, seq_len]

        Returns:
            Dictionary with metrics
        """
        # Get predictions
        predictions = logits.argmax(dim=-1)

        metrics = {
            'token_accuracy': AccuracyMetrics.token_accuracy(
                predictions, targets, self.ignore_index
            ),
            'top5_accuracy': AccuracyMetrics.top_k_accuracy(
                logits, targets, k=5, self.ignore_index
            ),
            'sequence_accuracy': AccuracyMetrics.sequence_accuracy(
                predictions, targets, self.ignore_index
            ),
            'token_entropy': EntropyMetrics.token_entropy(logits)
        }

        return metrics


# Example usage
def example_usage():
    """Demonstrate custom metrics calculation."""

    # Example tokens
    tokens = ["the", "cat", "sat", "on", "the", "mat", "the", "cat", "sat"]

    # Calculate diversity metrics
    diversity = DiversityMetrics.calculate_all(tokens)
    print("Diversity Metrics:")
    for metric, value in diversity.items():
        print(f"  {metric}: {value:.4f}")

    # Create dummy logits and targets
    batch_size, seq_len, vocab_size = 4, 32, 10000
    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Evaluate batch
    evaluator = ComprehensiveEvaluator()
    metrics = evaluator.evaluate_batch(logits, targets)

    print("\nBatch Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    example_usage()
