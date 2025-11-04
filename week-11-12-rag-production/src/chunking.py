"""
Chunking Strategies for RAG Systems
------------------------------------
Different strategies for splitting documents into chunks for embedding.
"""

import re
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class Chunk:
    """Represents a document chunk with metadata."""
    text: str
    start_idx: int
    end_idx: int
    metadata: Optional[dict] = None


class BaseChunker:
    """Base class for chunking strategies."""

    def chunk(self, text: str) -> List[Chunk]:
        """Split text into chunks."""
        raise NotImplementedError


class FixedSizeChunker(BaseChunker):
    """
    Fixed-size chunking with overlap.

    Simple but effective strategy where chunks have fixed size with overlap
    to maintain context at boundaries.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> List[Chunk]:
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            chunks.append(Chunk(
                text=chunk_text,
                start_idx=start,
                end_idx=end
            ))

            start = end - self.overlap

        return chunks


class SentenceChunker(BaseChunker):
    """
    Sentence-based chunking.

    Respects sentence boundaries for more coherent chunks.
    Groups sentences until reaching target size.
    """

    def __init__(self, target_size: int = 512, tolerance: int = 100):
        self.target_size = target_size
        self.tolerance = tolerance

    def chunk(self, text: str) -> List[Chunk]:
        # Simple sentence splitting (can be improved with spaCy/NLTK)
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = []
        current_size = 0
        start_idx = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            if current_size + sentence_len > self.target_size + self.tolerance:
                # Create chunk from accumulated sentences
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(Chunk(
                        text=chunk_text,
                        start_idx=start_idx,
                        end_idx=start_idx + len(chunk_text)
                    ))
                    start_idx += len(chunk_text) + 1

                current_chunk = [sentence]
                current_size = sentence_len
            else:
                current_chunk.append(sentence)
                current_size += sentence_len

        # Add remaining chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(Chunk(
                text=chunk_text,
                start_idx=start_idx,
                end_idx=start_idx + len(chunk_text)
            ))

        return chunks


class SemanticChunker(BaseChunker):
    """
    Semantic chunking based on topic coherence.

    Groups sentences by semantic similarity to create more coherent chunks.
    Uses sliding window to detect topic boundaries.
    """

    def __init__(
        self,
        min_chunk_size: int = 200,
        max_chunk_size: int = 800,
        similarity_threshold: float = 0.6
    ):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.similarity_threshold = similarity_threshold

    def chunk(self, text: str) -> List[Chunk]:
        """
        Simplified semantic chunking using paragraph breaks as boundaries.

        In a production system, you would:
        1. Embed each sentence using a sentence transformer
        2. Calculate similarity between adjacent sentences
        3. Split when similarity drops below threshold
        """
        # Split by paragraphs as proxy for semantic boundaries
        paragraphs = text.split('\n\n')

        chunks = []
        current_chunk = []
        current_size = 0
        start_idx = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_len = len(para)

            # Start new chunk if size exceeded
            if current_size + para_len > self.max_chunk_size and current_size >= self.min_chunk_size:
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append(Chunk(
                    text=chunk_text,
                    start_idx=start_idx,
                    end_idx=start_idx + len(chunk_text)
                ))
                start_idx += len(chunk_text) + 2
                current_chunk = [para]
                current_size = para_len
            else:
                current_chunk.append(para)
                current_size += para_len + 2  # +2 for \n\n

        # Add remaining chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append(Chunk(
                text=chunk_text,
                start_idx=start_idx,
                end_idx=start_idx + len(chunk_text)
            ))

        return chunks


class RecursiveChunker(BaseChunker):
    """
    Recursive chunking with hierarchical separators.

    Tries to split on major boundaries first (paragraphs),
    then sentences, then words if needed. Preserves structure better.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        # Separators in order of preference
        self.separators = ['\n\n', '\n', '. ', ' ']

    def chunk(self, text: str) -> List[Chunk]:
        return self._recursive_split(text, 0, 0)

    def _recursive_split(
        self,
        text: str,
        start_idx: int,
        separator_idx: int
    ) -> List[Chunk]:
        """Recursively split text using hierarchical separators."""

        if len(text) <= self.chunk_size:
            return [Chunk(text=text, start_idx=start_idx, end_idx=start_idx + len(text))]

        if separator_idx >= len(self.separators):
            # No more separators, force split
            return self._force_split(text, start_idx)

        separator = self.separators[separator_idx]
        splits = text.split(separator)

        if len(splits) == 1:
            # Separator not found, try next one
            return self._recursive_split(text, start_idx, separator_idx + 1)

        chunks = []
        current_chunk = []
        current_size = 0
        current_start = start_idx

        for split in splits:
            split_len = len(split)

            if current_size + split_len + len(separator) > self.chunk_size and current_chunk:
                # Create chunk
                chunk_text = separator.join(current_chunk)
                chunks.append(Chunk(
                    text=chunk_text,
                    start_idx=current_start,
                    end_idx=current_start + len(chunk_text)
                ))

                # Handle overlap
                if self.overlap > 0 and len(current_chunk) > 1:
                    current_chunk = current_chunk[-1:]
                    current_size = len(current_chunk[0])
                    current_start = current_start + len(chunk_text) - current_size
                else:
                    current_chunk = []
                    current_size = 0
                    current_start += len(chunk_text) + len(separator)

            current_chunk.append(split)
            current_size += split_len + len(separator)

        # Add remaining chunk
        if current_chunk:
            chunk_text = separator.join(current_chunk)
            chunks.append(Chunk(
                text=chunk_text,
                start_idx=current_start,
                end_idx=current_start + len(chunk_text)
            ))

        return chunks

    def _force_split(self, text: str, start_idx: int) -> List[Chunk]:
        """Force split when no separators work."""
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunks.append(Chunk(
                text=text[start:end],
                start_idx=start_idx + start,
                end_idx=start_idx + end
            ))
            start = end - self.overlap

        return chunks


def get_chunker(strategy: str = "recursive", **kwargs) -> BaseChunker:
    """Factory function to get chunker by strategy name."""
    chunkers = {
        "fixed": FixedSizeChunker,
        "sentence": SentenceChunker,
        "semantic": SemanticChunker,
        "recursive": RecursiveChunker
    }

    if strategy not in chunkers:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(chunkers.keys())}")

    return chunkers[strategy](**kwargs)
