"""
Advanced Quality Filters for LLM Training Data

This module provides sophisticated quality filtering techniques including
perplexity-based filtering, toxicity detection, and PII removal.
"""

import json
import re
from typing import List, Dict, Optional
from pathlib import Path
from tqdm import tqdm


class PerplexityFilter:
    """
    Filter documents based on perplexity scores from a reference model.
    High perplexity indicates unusual or low-quality text.
    """

    def __init__(self, threshold: float = 1000.0):
        self.threshold = threshold

    def calculate_perplexity(self, text: str) -> float:
        """
        Calculate perplexity of text using a simple heuristic.
        In practice, use a pretrained model.

        Returns:
            Perplexity score
        """
        # Placeholder: In real implementation, use a pretrained LM
        # For now, use character-level perplexity as proxy
        if not text:
            return float('inf')

        # Count character frequencies
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1

        # Calculate entropy
        total = len(text)
        entropy = 0
        for count in char_counts.values():
            prob = count / total
            entropy -= prob * (prob ** 0.5)  # Simplified calculation

        # Convert to perplexity
        perplexity = 2 ** entropy
        return perplexity

    def filter_documents(
        self,
        documents: List[Dict],
        text_field: str = 'text'
    ) -> List[Dict]:
        """Filter documents based on perplexity threshold."""
        print(f"Filtering by perplexity (threshold: {self.threshold})...")

        filtered_docs = []
        high_perplexity_count = 0

        for doc in tqdm(documents):
            text = doc.get(text_field, '')
            perplexity = self.calculate_perplexity(text)

            if perplexity < self.threshold:
                filtered_docs.append(doc)
            else:
                high_perplexity_count += 1

        print(f"Removed {high_perplexity_count} high-perplexity documents")
        print(f"Kept {len(filtered_docs)} documents")

        return filtered_docs


class ToxicityFilter:
    """
    Filter documents containing toxic or offensive content.
    """

    def __init__(self):
        # List of toxic/offensive keywords (simplified for demonstration)
        # In production, use a comprehensive toxicity detection model
        self.toxic_patterns = self._load_toxic_patterns()

    def _load_toxic_patterns(self) -> List[re.Pattern]:
        """Load regex patterns for toxic content detection."""
        # Simplified list - in practice, use a comprehensive model
        toxic_keywords = [
            # Add actual toxic words in production
            r'\b(placeholder_toxic_word)\b',
        ]

        return [re.compile(pattern, re.IGNORECASE) for pattern in toxic_keywords]

    def is_toxic(self, text: str) -> bool:
        """
        Check if text contains toxic content.

        Returns:
            True if toxic content is detected
        """
        if not text:
            return False

        # Check against patterns
        for pattern in self.toxic_patterns:
            if pattern.search(text):
                return True

        return False

    def filter_documents(
        self,
        documents: List[Dict],
        text_field: str = 'text'
    ) -> List[Dict]:
        """Filter out documents with toxic content."""
        print("Filtering toxic content...")

        filtered_docs = []
        toxic_count = 0

        for doc in tqdm(documents):
            text = doc.get(text_field, '')

            if not self.is_toxic(text):
                filtered_docs.append(doc)
            else:
                toxic_count += 1

        print(f"Removed {toxic_count} toxic documents")
        print(f"Kept {len(filtered_docs)} clean documents")

        return filtered_docs


class PIIRemover:
    """
    Remove Personally Identifiable Information (PII) from text.
    """

    def __init__(self):
        # Regex patterns for common PII
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        self.phone_pattern = re.compile(
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        )
        self.ssn_pattern = re.compile(
            r'\b\d{3}-\d{2}-\d{4}\b'
        )
        self.credit_card_pattern = re.compile(
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
        )

    def remove_pii(self, text: str) -> str:
        """
        Remove PII from text by replacing with placeholders.

        Args:
            text: Input text

        Returns:
            Text with PII removed
        """
        if not text:
            return text

        # Remove emails
        text = self.email_pattern.sub('[EMAIL]', text)

        # Remove phone numbers
        text = self.phone_pattern.sub('[PHONE]', text)

        # Remove SSN
        text = self.ssn_pattern.sub('[SSN]', text)

        # Remove credit cards
        text = self.credit_card_pattern.sub('[CREDIT_CARD]', text)

        return text

    def process_documents(
        self,
        documents: List[Dict],
        text_field: str = 'text'
    ) -> List[Dict]:
        """Process documents to remove PII."""
        print("Removing PII from documents...")

        for doc in tqdm(documents):
            text = doc.get(text_field, '')
            cleaned_text = self.remove_pii(text)
            doc[text_field] = cleaned_text

        print(f"Processed {len(documents)} documents")
        return documents


class ContentFilter:
    """
    Filter documents based on content characteristics.
    """

    def __init__(
        self,
        min_alpha_ratio: float = 0.7,
        max_digit_ratio: float = 0.3,
        max_special_char_ratio: float = 0.3
    ):
        self.min_alpha_ratio = min_alpha_ratio
        self.max_digit_ratio = max_digit_ratio
        self.max_special_char_ratio = max_special_char_ratio

    def get_character_ratios(self, text: str) -> Dict[str, float]:
        """Calculate character type ratios."""
        if not text:
            return {'alpha': 0, 'digit': 0, 'special': 0}

        total = len(text)
        alpha = sum(c.isalpha() for c in text) / total
        digit = sum(c.isdigit() for c in text) / total
        special = sum(not c.isalnum() and not c.isspace() for c in text) / total

        return {'alpha': alpha, 'digit': digit, 'special': special}

    def is_valid_content(self, text: str) -> tuple[bool, str]:
        """
        Check if content meets character ratio requirements.

        Returns:
            (is_valid, reason)
        """
        ratios = self.get_character_ratios(text)

        if ratios['alpha'] < self.min_alpha_ratio:
            return False, "insufficient_alpha"

        if ratios['digit'] > self.max_digit_ratio:
            return False, "excessive_digits"

        if ratios['special'] > self.max_special_char_ratio:
            return False, "excessive_special_chars"

        return True, "ok"

    def filter_documents(
        self,
        documents: List[Dict],
        text_field: str = 'text'
    ) -> List[Dict]:
        """Filter documents based on content characteristics."""
        print("Filtering by content characteristics...")

        filtered_docs = []
        rejection_counts = {'insufficient_alpha': 0, 'excessive_digits': 0, 'excessive_special_chars': 0}

        for doc in tqdm(documents):
            text = doc.get(text_field, '')
            is_valid, reason = self.is_valid_content(text)

            if is_valid:
                filtered_docs.append(doc)
            else:
                rejection_counts[reason] += 1

        print(f"Content filtering results:")
        print(f"  Kept: {len(filtered_docs)} documents")
        print(f"  Rejected: {sum(rejection_counts.values())} documents")
        print(f"  Rejection breakdown: {rejection_counts}")

        return filtered_docs


class AdvancedQualityPipeline:
    """Complete advanced quality filtering pipeline."""

    def __init__(self, input_file: str, output_file: str, text_field: str = 'text'):
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.text_field = text_field

        # Initialize filters
        self.perplexity_filter = PerplexityFilter(threshold=1000.0)
        self.toxicity_filter = ToxicityFilter()
        self.pii_remover = PIIRemover()
        self.content_filter = ContentFilter()

    def load_documents(self) -> List[Dict]:
        """Load documents from JSONL file."""
        documents = []
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                documents.append(json.loads(line))
        return documents

    def save_documents(self, documents: List[Dict]):
        """Save documents to JSONL file."""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for doc in documents:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')

    def run(self):
        """Run the complete advanced quality pipeline."""
        print(f"Starting advanced quality pipeline for {self.input_file}")
        print("=" * 60)

        # Load documents
        documents = self.load_documents()
        print(f"Loaded {len(documents)} documents\n")

        # Step 1: Remove PII
        print("Step 1: Removing PII...")
        documents = self.pii_remover.process_documents(
            documents,
            text_field=self.text_field
        )

        # Step 2: Filter by content characteristics
        print("\nStep 2: Content filtering...")
        documents = self.content_filter.filter_documents(
            documents,
            text_field=self.text_field
        )

        # Step 3: Filter toxic content
        print("\nStep 3: Toxicity filtering...")
        documents = self.toxicity_filter.filter_documents(
            documents,
            text_field=self.text_field
        )

        # Step 4: Perplexity filtering
        print("\nStep 4: Perplexity filtering...")
        documents = self.perplexity_filter.filter_documents(
            documents,
            text_field=self.text_field
        )

        # Save filtered documents
        print(f"\nSaving {len(documents)} high-quality documents to {self.output_file}")
        self.save_documents(documents)

        print("=" * 60)
        print("Advanced quality pipeline completed!")


def main():
    """Example usage of advanced quality filters."""
    pipeline = AdvancedQualityPipeline(
        input_file="data/cleaned/cleaned_dataset.jsonl",
        output_file="data/filtered/high_quality_dataset.jsonl",
        text_field="text"
    )
    pipeline.run()


if __name__ == "__main__":
    main()
