"""
BLEU and ROUGE Metrics for Text Generation Evaluation

BLEU (Bilingual Evaluation Understudy): Measures n-gram overlap between generated and reference text
ROUGE (Recall-Oriented Understudy for Gisting Evaluation): Focuses on recall of n-grams

Both are commonly used for evaluating machine translation and text generation.
"""

import math
from collections import Counter
from typing import List, Dict, Union
import numpy as np


class BLEUCalculator:
    """
    Calculate BLEU score for text generation.

    BLEU measures precision of n-grams in generated text vs reference text.
    """

    def __init__(self, max_n: int = 4, weights: List[float] = None):
        """
        Initialize BLEU calculator.

        Args:
            max_n: Maximum n-gram size (default 4 for BLEU-4)
            weights: Weights for each n-gram size (defaults to uniform)
        """
        self.max_n = max_n
        self.weights = weights or [1.0 / max_n] * max_n

    def get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """
        Extract n-grams from token list.

        Args:
            tokens: List of tokens
            n: N-gram size

        Returns:
            Counter of n-grams
        """
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams.append(ngram)
        return Counter(ngrams)

    def modified_precision(
        self,
        hypothesis: List[str],
        references: List[List[str]],
        n: int
    ) -> float:
        """
        Calculate modified n-gram precision.

        Args:
            hypothesis: Generated text tokens
            references: List of reference text tokens
            n: N-gram size

        Returns:
            Modified precision score
        """
        hyp_ngrams = self.get_ngrams(hypothesis, n)

        if not hyp_ngrams:
            return 0.0

        # Get maximum counts from all references
        max_ref_counts = Counter()
        for reference in references:
            ref_ngrams = self.get_ngrams(reference, n)
            for ngram in hyp_ngrams:
                max_ref_counts[ngram] = max(max_ref_counts[ngram], ref_ngrams[ngram])

        # Calculate clipped counts
        clipped_counts = {
            ngram: min(count, max_ref_counts[ngram])
            for ngram, count in hyp_ngrams.items()
        }

        # Calculate precision
        numerator = sum(clipped_counts.values())
        denominator = sum(hyp_ngrams.values())

        return numerator / denominator if denominator > 0 else 0.0

    def brevity_penalty(
        self,
        hypothesis_length: int,
        reference_length: int
    ) -> float:
        """
        Calculate brevity penalty for short hypotheses.

        Args:
            hypothesis_length: Length of generated text
            reference_length: Length of reference text

        Returns:
            Brevity penalty value
        """
        if hypothesis_length >= reference_length:
            return 1.0
        else:
            return math.exp(1 - reference_length / hypothesis_length)

    def calculate_bleu(
        self,
        hypothesis: Union[str, List[str]],
        references: Union[str, List[str], List[List[str]]]
    ) -> Dict[str, float]:
        """
        Calculate BLEU score.

        Args:
            hypothesis: Generated text (string or tokens)
            references: Reference text(s) (string, tokens, or list of token lists)

        Returns:
            Dictionary with BLEU scores and components
        """
        # Convert to token lists if needed
        if isinstance(hypothesis, str):
            hypothesis = hypothesis.split()

        if isinstance(references, str):
            references = [references.split()]
        elif isinstance(references[0], str):
            references = [references]

        # Calculate modified precision for each n-gram size
        precisions = []
        for n in range(1, self.max_n + 1):
            precision = self.modified_precision(hypothesis, references, n)
            precisions.append(precision)

        # Calculate geometric mean of precisions
        # Use log to avoid numerical issues
        if min(precisions) > 0:
            log_precisions = [math.log(p) for p in precisions]
            geo_mean = math.exp(sum(w * lp for w, lp in zip(self.weights, log_precisions)))
        else:
            geo_mean = 0.0

        # Calculate brevity penalty
        hyp_length = len(hypothesis)
        ref_length = min(len(ref) for ref in references)
        bp = self.brevity_penalty(hyp_length, ref_length)

        # Final BLEU score
        bleu = bp * geo_mean

        return {
            'bleu': bleu,
            'precisions': precisions,
            'brevity_penalty': bp,
            'length_ratio': hyp_length / ref_length if ref_length > 0 else 0,
            'hypothesis_length': hyp_length,
            'reference_length': ref_length
        }


class ROUGECalculator:
    """
    Calculate ROUGE scores for text generation.

    ROUGE measures recall of n-grams from reference text in generated text.
    """

    def __init__(self):
        pass

    def get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """Extract n-grams from token list."""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams.append(ngram)
        return Counter(ngrams)

    def rouge_n(
        self,
        hypothesis: List[str],
        reference: List[str],
        n: int
    ) -> Dict[str, float]:
        """
        Calculate ROUGE-N score.

        Args:
            hypothesis: Generated text tokens
            reference: Reference text tokens
            n: N-gram size

        Returns:
            Dictionary with precision, recall, and F1
        """
        hyp_ngrams = self.get_ngrams(hypothesis, n)
        ref_ngrams = self.get_ngrams(reference, n)

        if not ref_ngrams:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

        # Calculate overlapping n-grams
        overlap = sum((hyp_ngrams & ref_ngrams).values())

        # Calculate metrics
        precision = overlap / sum(hyp_ngrams.values()) if hyp_ngrams else 0.0
        recall = overlap / sum(ref_ngrams.values()) if ref_ngrams else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def rouge_l(
        self,
        hypothesis: List[str],
        reference: List[str]
    ) -> Dict[str, float]:
        """
        Calculate ROUGE-L (Longest Common Subsequence).

        Args:
            hypothesis: Generated text tokens
            reference: Reference text tokens

        Returns:
            Dictionary with precision, recall, and F1
        """
        def lcs_length(x: List[str], y: List[str]) -> int:
            """Calculate longest common subsequence length."""
            m, n = len(x), len(y)
            dp = [[0] * (n + 1) for _ in range(m + 1)]

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i - 1] == y[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

            return dp[m][n]

        lcs_len = lcs_length(hypothesis, reference)

        # Calculate metrics
        precision = lcs_len / len(hypothesis) if hypothesis else 0.0
        recall = lcs_len / len(reference) if reference else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'lcs_length': lcs_len
        }

    def calculate_rouge(
        self,
        hypothesis: Union[str, List[str]],
        reference: Union[str, List[str]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate multiple ROUGE scores.

        Args:
            hypothesis: Generated text (string or tokens)
            reference: Reference text (string or tokens)

        Returns:
            Dictionary with ROUGE-1, ROUGE-2, and ROUGE-L scores
        """
        # Convert to token lists if needed
        if isinstance(hypothesis, str):
            hypothesis = hypothesis.split()
        if isinstance(reference, str):
            reference = reference.split()

        return {
            'rouge-1': self.rouge_n(hypothesis, reference, 1),
            'rouge-2': self.rouge_n(hypothesis, reference, 2),
            'rouge-l': self.rouge_l(hypothesis, reference)
        }


def calculate_corpus_bleu(
    hypotheses: List[List[str]],
    references: List[List[List[str]]],
    max_n: int = 4
) -> float:
    """
    Calculate corpus-level BLEU score.

    Args:
        hypotheses: List of generated text token lists
        references: List of reference token lists (one or more per hypothesis)
        max_n: Maximum n-gram size

    Returns:
        Corpus BLEU score
    """
    calculator = BLEUCalculator(max_n=max_n)

    total_hyp_length = 0
    total_ref_length = 0
    clipped_counts = [Counter() for _ in range(max_n)]
    total_counts = [Counter() for _ in range(max_n)]

    for hyp, refs in zip(hypotheses, references):
        total_hyp_length += len(hyp)

        # Find closest reference length
        ref_lengths = [len(ref) for ref in refs]
        closest_ref_length = min(ref_lengths, key=lambda x: abs(x - len(hyp)))
        total_ref_length += closest_ref_length

        # Accumulate n-gram statistics
        for n in range(1, max_n + 1):
            hyp_ngrams = calculator.get_ngrams(hyp, n)

            # Get maximum counts from references
            max_ref_counts = Counter()
            for ref in refs:
                ref_ngrams = calculator.get_ngrams(ref, n)
                for ngram in hyp_ngrams:
                    max_ref_counts[ngram] = max(max_ref_counts[ngram], ref_ngrams[ngram])

            # Accumulate clipped counts
            for ngram, count in hyp_ngrams.items():
                clipped_counts[n - 1][ngram] += min(count, max_ref_counts[ngram])
                total_counts[n - 1][ngram] += count

    # Calculate precision for each n-gram size
    precisions = []
    for n in range(max_n):
        clipped = sum(clipped_counts[n].values())
        total = sum(total_counts[n].values())
        precision = clipped / total if total > 0 else 0.0
        precisions.append(precision)

    # Calculate geometric mean
    if min(precisions) > 0:
        weights = [1.0 / max_n] * max_n
        log_precisions = [math.log(p) for p in precisions]
        geo_mean = math.exp(sum(w * lp for w, lp in zip(weights, log_precisions)))
    else:
        geo_mean = 0.0

    # Calculate brevity penalty
    bp = calculator.brevity_penalty(total_hyp_length, total_ref_length)

    return bp * geo_mean


# Example usage
def example_usage():
    """Demonstrate BLEU and ROUGE calculation."""

    # Example texts
    hypothesis = "the cat sat on the mat"
    reference1 = "the cat is sitting on the mat"
    reference2 = "a cat was sitting on the mat"

    # Calculate BLEU
    bleu_calc = BLEUCalculator()
    bleu_scores = bleu_calc.calculate_bleu(hypothesis, [reference1, reference2])

    print("BLEU Scores:")
    print(f"  BLEU: {bleu_scores['bleu']:.4f}")
    print(f"  Precisions: {[f'{p:.4f}' for p in bleu_scores['precisions']]}")
    print(f"  Brevity Penalty: {bleu_scores['brevity_penalty']:.4f}")

    # Calculate ROUGE
    rouge_calc = ROUGECalculator()
    rouge_scores = rouge_calc.calculate_rouge(hypothesis, reference1)

    print("\nROUGE Scores:")
    for metric, scores in rouge_scores.items():
        print(f"  {metric}:")
        print(f"    Precision: {scores['precision']:.4f}")
        print(f"    Recall: {scores['recall']:.4f}")
        print(f"    F1: {scores['f1']:.4f}")


if __name__ == "__main__":
    example_usage()
