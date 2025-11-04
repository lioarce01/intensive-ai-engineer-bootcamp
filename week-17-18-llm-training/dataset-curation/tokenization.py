"""
Tokenization for LLM Training

This module provides tools for training tokenizers and preparing tokenized datasets.
"""

import json
from typing import List, Dict, Iterator
from pathlib import Path
from tqdm import tqdm
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors


class TokenizerTrainer:
    """Train custom tokenizers for LLM training."""

    def __init__(self, vocab_size: int = 32000, min_frequency: int = 2):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency

    def train_bpe_tokenizer(
        self,
        texts: Iterator[str],
        special_tokens: List[str] = None
    ) -> Tokenizer:
        """
        Train a BPE (Byte Pair Encoding) tokenizer.

        Args:
            texts: Iterator of text strings
            special_tokens: List of special tokens

        Returns:
            Trained tokenizer
        """
        if special_tokens is None:
            special_tokens = [
                "<s>",      # Start of sequence
                "</s>",     # End of sequence
                "<unk>",    # Unknown token
                "<pad>",    # Padding token
                "<mask>",   # Mask token
            ]

        print(f"Training BPE tokenizer with vocab size {self.vocab_size}...")

        # Initialize BPE tokenizer
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

        # Setup pre-tokenizer
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

        # Setup trainer
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=special_tokens,
            show_progress=True
        )

        # Train tokenizer
        tokenizer.train_from_iterator(texts, trainer=trainer)

        # Setup post-processor
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

        print("Tokenizer training completed!")
        return tokenizer

    def train_from_files(
        self,
        input_files: List[str],
        output_path: str,
        text_field: str = 'text'
    ):
        """
        Train tokenizer from JSONL files.

        Args:
            input_files: List of input JSONL files
            output_path: Path to save trained tokenizer
            text_field: Field containing text in JSONL
        """
        def text_iterator():
            for file_path in input_files:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        doc = json.loads(line)
                        text = doc.get(text_field, '')
                        if text:
                            yield text

        # Train tokenizer
        tokenizer = self.train_bpe_tokenizer(text_iterator())

        # Save tokenizer
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(output_path)
        print(f"Tokenizer saved to {output_path}")

        return tokenizer


class DatasetTokenizer:
    """Tokenize datasets for LLM training."""

    def __init__(self, tokenizer_path: str):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

    def tokenize_text(self, text: str) -> List[int]:
        """Tokenize a single text."""
        encoding = self.tokenizer.encode(text)
        return encoding.ids

    def tokenize_dataset(
        self,
        input_file: str,
        output_file: str,
        text_field: str = 'text',
        max_length: int = 1024,
        add_special_tokens: bool = True
    ):
        """
        Tokenize entire dataset.

        Args:
            input_file: Input JSONL file
            output_file: Output JSONL file with tokenized data
            text_field: Field containing text
            max_length: Maximum sequence length
            add_special_tokens: Add special tokens
        """
        print(f"Tokenizing dataset from {input_file}...")

        input_path = Path(input_file)
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total_tokens = 0
        num_documents = 0
        num_sequences = 0

        with open(input_path, 'r', encoding='utf-8') as f_in:
            with open(output_path, 'w', encoding='utf-8') as f_out:
                for line in tqdm(f_in):
                    doc = json.loads(line)
                    text = doc.get(text_field, '')

                    if not text:
                        continue

                    # Tokenize
                    tokens = self.tokenize_text(text)
                    total_tokens += len(tokens)
                    num_documents += 1

                    # Split into sequences of max_length
                    for i in range(0, len(tokens), max_length):
                        sequence = tokens[i:i + max_length]

                        if len(sequence) < 10:  # Skip very short sequences
                            continue

                        # Add special tokens if requested
                        if add_special_tokens:
                            sequence = [self.tokenizer.token_to_id("<s>")] + sequence + [self.tokenizer.token_to_id("</s>")]

                        # Save tokenized sequence
                        tokenized_doc = {
                            'input_ids': sequence,
                            'length': len(sequence),
                            'source': doc.get('source', 'unknown')
                        }
                        f_out.write(json.dumps(tokenized_doc) + '\n')
                        num_sequences += 1

        print(f"\nTokenization complete!")
        print(f"  Documents processed: {num_documents}")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Sequences created: {num_sequences}")
        print(f"  Average tokens per document: {total_tokens / num_documents:.1f}")
        print(f"  Output saved to: {output_path}")


class TokenCounter:
    """Analyze token statistics in datasets."""

    def __init__(self, tokenizer_path: str):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

    def count_tokens(self, input_file: str, text_field: str = 'text') -> Dict:
        """
        Count tokens in dataset.

        Args:
            input_file: Input JSONL file
            text_field: Field containing text

        Returns:
            Dictionary with token statistics
        """
        print(f"Counting tokens in {input_file}...")

        total_tokens = 0
        total_chars = 0
        num_documents = 0
        token_lengths = []

        with open(input_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                doc = json.loads(line)
                text = doc.get(text_field, '')

                if not text:
                    continue

                tokens = self.tokenize_text(text)
                num_tokens = len(tokens)

                total_tokens += num_tokens
                total_chars += len(text)
                num_documents += 1
                token_lengths.append(num_tokens)

        stats = {
            'total_documents': num_documents,
            'total_tokens': total_tokens,
            'total_characters': total_chars,
            'avg_tokens_per_doc': total_tokens / num_documents if num_documents > 0 else 0,
            'avg_chars_per_token': total_chars / total_tokens if total_tokens > 0 else 0,
            'min_tokens': min(token_lengths) if token_lengths else 0,
            'max_tokens': max(token_lengths) if token_lengths else 0,
        }

        return stats

    def tokenize_text(self, text: str) -> List[int]:
        """Tokenize a single text."""
        encoding = self.tokenizer.encode(text)
        return encoding.ids


def analyze_tokenizer(tokenizer_path: str):
    """Analyze tokenizer properties."""
    tokenizer = Tokenizer.from_file(tokenizer_path)

    print(f"Tokenizer Analysis")
    print("=" * 60)
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Model type: {tokenizer.model.__class__.__name__}")

    # Test tokenization
    test_texts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "LLM training requires high-quality data.",
        "import torch\nimport numpy as np\n\nmodel = torch.nn.Linear(10, 5)"
    ]

    print("\nSample Tokenizations:")
    print("-" * 60)
    for text in test_texts:
        encoding = tokenizer.encode(text)
        print(f"Text: {text[:50]}...")
        print(f"Tokens ({len(encoding.ids)}): {encoding.tokens[:20]}")
        print(f"IDs: {encoding.ids[:20]}")
        print()


def main():
    """Example usage of tokenization tools."""

    # Step 1: Train tokenizer
    trainer = TokenizerTrainer(vocab_size=32000, min_frequency=2)
    trainer.train_from_files(
        input_files=["data/cleaned/cleaned_dataset.jsonl"],
        output_path="tokenizers/custom_tokenizer.json",
        text_field="text"
    )

    # Step 2: Analyze tokenizer
    analyze_tokenizer("tokenizers/custom_tokenizer.json")

    # Step 3: Tokenize dataset
    dataset_tokenizer = DatasetTokenizer("tokenizers/custom_tokenizer.json")
    dataset_tokenizer.tokenize_dataset(
        input_file="data/cleaned/cleaned_dataset.jsonl",
        output_file="data/tokenized/train_tokenized.jsonl",
        text_field="text",
        max_length=1024
    )

    # Step 4: Count tokens
    counter = TokenCounter("tokenizers/custom_tokenizer.json")
    stats = counter.count_tokens(
        input_file="data/cleaned/cleaned_dataset.jsonl",
        text_field="text"
    )

    print("\nToken Statistics:")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
