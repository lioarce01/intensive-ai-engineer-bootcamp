"""
Data Cleaning and Preprocessing for LLM Training

This module provides tools for cleaning and preprocessing collected text data
to ensure high quality training data.
"""

import re
import json
from typing import List, Dict, Set, Optional
from pathlib import Path
from tqdm import tqdm
import unicodedata
from collections import Counter


class TextCleaner:
    """Clean and normalize text data."""

    def __init__(self):
        # Common patterns to remove
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.multiple_spaces = re.compile(r'\s+')
        self.multiple_newlines = re.compile(r'\n{3,}')

    def clean_text(
        self,
        text: str,
        remove_urls: bool = True,
        remove_emails: bool = True,
        normalize_whitespace: bool = True,
        remove_special_chars: bool = False
    ) -> str:
        """
        Clean and normalize text.

        Args:
            text: Input text
            remove_urls: Remove URLs
            remove_emails: Remove email addresses
            normalize_whitespace: Normalize whitespace
            remove_special_chars: Remove special characters

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Normalize unicode characters
        text = unicodedata.normalize('NFKC', text)

        # Remove URLs
        if remove_urls:
            text = self.url_pattern.sub('', text)

        # Remove emails
        if remove_emails:
            text = self.email_pattern.sub('', text)

        # Remove special characters (keep alphanumeric and basic punctuation)
        if remove_special_chars:
            text = re.sub(r'[^a-zA-Z0-9\s.,!?;:\'\"-]', '', text)

        # Normalize whitespace
        if normalize_whitespace:
            text = self.multiple_spaces.sub(' ', text)
            text = self.multiple_newlines.sub('\n\n', text)
            text = text.strip()

        return text

    def remove_boilerplate(self, text: str) -> str:
        """
        Remove common boilerplate text.

        Args:
            text: Input text

        Returns:
            Text with boilerplate removed
        """
        # Common boilerplate patterns
        boilerplate_patterns = [
            r'(?i)cookie policy',
            r'(?i)privacy policy',
            r'(?i)terms of service',
            r'(?i)all rights reserved',
            r'(?i)subscribe to our newsletter',
        ]

        for pattern in boilerplate_patterns:
            text = re.sub(pattern, '', text)

        return text


class DuplicateRemover:
    """Remove duplicate and near-duplicate documents."""

    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.seen_hashes: Set[str] = set()

    def get_text_hash(self, text: str) -> str:
        """Get hash of text for exact duplicate detection."""
        import hashlib
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def is_exact_duplicate(self, text: str) -> bool:
        """Check if text is an exact duplicate."""
        text_hash = self.get_text_hash(text)
        if text_hash in self.seen_hashes:
            return True
        self.seen_hashes.add(text_hash)
        return False

    def remove_duplicates(
        self,
        documents: List[Dict],
        text_field: str = 'text'
    ) -> List[Dict]:
        """
        Remove exact duplicate documents.

        Args:
            documents: List of document dictionaries
            text_field: Field containing text content

        Returns:
            Deduplicated list of documents
        """
        print("Removing duplicates...")
        unique_docs = []
        duplicates_count = 0

        for doc in tqdm(documents):
            text = doc.get(text_field, '')
            if not self.is_exact_duplicate(text):
                unique_docs.append(doc)
            else:
                duplicates_count += 1

        print(f"Removed {duplicates_count} duplicate documents")
        print(f"Remaining: {len(unique_docs)} unique documents")

        return unique_docs


class LanguageFilter:
    """Filter documents by language."""

    def __init__(self, target_languages: List[str] = ['en']):
        self.target_languages = target_languages
        try:
            from langdetect import detect
            self.detect = detect
        except ImportError:
            print("Warning: langdetect not installed. Language filtering disabled.")
            self.detect = None

    def detect_language(self, text: str) -> Optional[str]:
        """Detect language of text."""
        if not self.detect or not text:
            return None

        try:
            return self.detect(text)
        except:
            return None

    def filter_by_language(
        self,
        documents: List[Dict],
        text_field: str = 'text'
    ) -> List[Dict]:
        """
        Filter documents by language.

        Args:
            documents: List of document dictionaries
            text_field: Field containing text content

        Returns:
            Filtered list of documents
        """
        if not self.detect:
            print("Language detection not available, returning all documents")
            return documents

        print(f"Filtering for languages: {self.target_languages}")
        filtered_docs = []
        language_counts = Counter()

        for doc in tqdm(documents):
            text = doc.get(text_field, '')
            lang = self.detect_language(text)
            language_counts[lang] += 1

            if lang in self.target_languages:
                filtered_docs.append(doc)

        print(f"Language distribution: {dict(language_counts)}")
        print(f"Kept {len(filtered_docs)} documents in target languages")

        return filtered_docs


class QualityFilter:
    """Filter documents based on quality metrics."""

    def __init__(
        self,
        min_length: int = 100,
        max_length: int = 10000,
        min_words: int = 20,
        max_word_length: int = 50,
        min_unique_words_ratio: float = 0.3
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.min_words = min_words
        self.max_word_length = max_word_length
        self.min_unique_words_ratio = min_unique_words_ratio

    def is_high_quality(self, text: str) -> tuple[bool, str]:
        """
        Check if text meets quality criteria.

        Returns:
            (is_quality, reason)
        """
        if not text:
            return False, "empty"

        # Length check
        if len(text) < self.min_length:
            return False, "too_short"
        if len(text) > self.max_length:
            return False, "too_long"

        # Word count check
        words = text.split()
        if len(words) < self.min_words:
            return False, "too_few_words"

        # Check for very long words (likely garbage)
        if any(len(word) > self.max_word_length for word in words):
            return False, "invalid_words"

        # Check word diversity
        unique_words_ratio = len(set(words)) / len(words)
        if unique_words_ratio < self.min_unique_words_ratio:
            return False, "low_diversity"

        # Check for excessive repetition
        if self._has_excessive_repetition(text):
            return False, "excessive_repetition"

        return True, "ok"

    def _has_excessive_repetition(self, text: str, threshold: int = 5) -> bool:
        """Check for excessive character or word repetition."""
        # Check for repeated characters (e.g., "aaaaaa")
        if re.search(r'(.)\1{10,}', text):
            return True

        # Check for repeated n-grams
        words = text.split()
        for n in [3, 4, 5]:
            ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
            if ngrams:
                most_common_count = Counter(ngrams).most_common(1)[0][1]
                if most_common_count >= threshold:
                    return True

        return False

    def filter_documents(
        self,
        documents: List[Dict],
        text_field: str = 'text'
    ) -> List[Dict]:
        """
        Filter documents based on quality.

        Args:
            documents: List of document dictionaries
            text_field: Field containing text content

        Returns:
            High-quality documents
        """
        print("Filtering for quality...")
        filtered_docs = []
        rejection_reasons = Counter()

        for doc in tqdm(documents):
            text = doc.get(text_field, '')
            is_quality, reason = self.is_high_quality(text)

            if is_quality:
                filtered_docs.append(doc)
            else:
                rejection_reasons[reason] += 1

        print(f"Quality filtering results:")
        print(f"  Kept: {len(filtered_docs)} documents")
        print(f"  Rejected: {len(documents) - len(filtered_docs)} documents")
        print(f"  Rejection reasons: {dict(rejection_reasons)}")

        return filtered_docs


class DataCleaningPipeline:
    """Complete data cleaning pipeline."""

    def __init__(
        self,
        input_file: str,
        output_file: str,
        text_field: str = 'text'
    ):
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.text_field = text_field

        # Initialize components
        self.text_cleaner = TextCleaner()
        self.duplicate_remover = DuplicateRemover()
        self.language_filter = LanguageFilter(target_languages=['en'])
        self.quality_filter = QualityFilter()

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
        """Run the complete cleaning pipeline."""
        print(f"Starting cleaning pipeline for {self.input_file}")
        print("=" * 60)

        # Load documents
        documents = self.load_documents()
        print(f"Loaded {len(documents)} documents\n")

        # Step 1: Clean text
        print("Step 1: Cleaning text...")
        for doc in tqdm(documents):
            text = doc.get(self.text_field, '')
            cleaned = self.text_cleaner.clean_text(text)
            cleaned = self.text_cleaner.remove_boilerplate(cleaned)
            doc[self.text_field] = cleaned

        # Step 2: Remove duplicates
        print("\nStep 2: Removing duplicates...")
        documents = self.duplicate_remover.remove_duplicates(
            documents,
            text_field=self.text_field
        )

        # Step 3: Filter by language
        print("\nStep 3: Filtering by language...")
        documents = self.language_filter.filter_by_language(
            documents,
            text_field=self.text_field
        )

        # Step 4: Quality filtering
        print("\nStep 4: Quality filtering...")
        documents = self.quality_filter.filter_documents(
            documents,
            text_field=self.text_field
        )

        # Save cleaned documents
        print(f"\nSaving {len(documents)} cleaned documents to {self.output_file}")
        self.save_documents(documents)

        print("=" * 60)
        print("Cleaning pipeline completed!")


def main():
    """Example usage of data cleaning pipeline."""
    pipeline = DataCleaningPipeline(
        input_file="data/merged/merged_dataset.jsonl",
        output_file="data/cleaned/cleaned_dataset.jsonl",
        text_field="text"
    )
    pipeline.run()


if __name__ == "__main__":
    main()
