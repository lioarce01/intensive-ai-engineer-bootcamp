"""
Dataset Collection for LLM Training

This module provides tools for collecting high-quality text data from various sources
including Wikipedia, GitHub, and web scraping.
"""

import requests
import time
from typing import List, Dict, Optional
from pathlib import Path
import json
from datasets import load_dataset
from tqdm import tqdm


class WikipediaCollector:
    """Collect and process Wikipedia articles for LLM training."""

    def __init__(self, language: str = "en", output_dir: str = "data/raw"):
        self.language = language
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_wikipedia_subset(
        self,
        num_articles: int = 10000,
        min_length: int = 500
    ) -> List[Dict[str, str]]:
        """
        Download a subset of Wikipedia articles.

        Args:
            num_articles: Number of articles to download
            min_length: Minimum article length in characters

        Returns:
            List of article dictionaries with 'title' and 'text'
        """
        print(f"Downloading {num_articles} Wikipedia articles...")

        # Use Hugging Face's Wikipedia dataset
        dataset = load_dataset(
            "wikipedia",
            f"20220301.{self.language}",
            split="train",
            streaming=True
        )

        articles = []
        for i, article in enumerate(tqdm(dataset, total=num_articles)):
            if len(articles) >= num_articles:
                break

            text = article['text']
            if len(text) >= min_length:
                articles.append({
                    'title': article['title'],
                    'text': text,
                    'url': article.get('url', ''),
                    'source': 'wikipedia'
                })

        # Save to disk
        output_file = self.output_dir / f"wikipedia_{self.language}_{num_articles}.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for article in articles:
                f.write(json.dumps(article, ensure_ascii=False) + '\n')

        print(f"Saved {len(articles)} articles to {output_file}")
        return articles


class CodeCollector:
    """Collect code samples from GitHub and other sources."""

    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_code_dataset(
        self,
        language: str = "python",
        num_samples: int = 5000
    ) -> List[Dict[str, str]]:
        """
        Download code samples from Hugging Face datasets.

        Args:
            language: Programming language
            num_samples: Number of code samples

        Returns:
            List of code sample dictionaries
        """
        print(f"Downloading {num_samples} {language} code samples...")

        # Use The Stack dataset
        dataset = load_dataset(
            "bigcode/the-stack-dedup",
            data_dir=f"data/{language}",
            split="train",
            streaming=True
        )

        samples = []
        for i, sample in enumerate(tqdm(dataset, total=num_samples)):
            if len(samples) >= num_samples:
                break

            samples.append({
                'content': sample['content'],
                'language': language,
                'source': 'the-stack',
                'max_stars_repo_path': sample.get('max_stars_repo_path', ''),
                'max_stars_count': sample.get('max_stars_count', 0)
            })

        # Save to disk
        output_file = self.output_dir / f"code_{language}_{num_samples}.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        print(f"Saved {len(samples)} code samples to {output_file}")
        return samples


class WebScraper:
    """Simple web scraper for collecting text data."""

    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Educational LLM Training Dataset Collection)'
        }

    def scrape_urls(
        self,
        urls: List[str],
        delay: float = 1.0
    ) -> List[Dict[str, str]]:
        """
        Scrape content from a list of URLs.

        Args:
            urls: List of URLs to scrape
            delay: Delay between requests in seconds

        Returns:
            List of scraped content dictionaries
        """
        print(f"Scraping {len(urls)} URLs...")

        scraped_data = []
        for url in tqdm(urls):
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                response.raise_for_status()

                # Simple text extraction (in practice, use BeautifulSoup)
                content = response.text

                scraped_data.append({
                    'url': url,
                    'content': content,
                    'source': 'web_scraping',
                    'timestamp': time.time()
                })

                time.sleep(delay)

            except Exception as e:
                print(f"Error scraping {url}: {e}")
                continue

        # Save to disk
        output_file = self.output_dir / "web_scraped.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for data in scraped_data:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')

        print(f"Saved {len(scraped_data)} scraped pages to {output_file}")
        return scraped_data


class DatasetMerger:
    """Merge and organize collected datasets."""

    def __init__(self, input_dir: str = "data/raw", output_dir: str = "data/merged"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def merge_datasets(self, output_file: str = "merged_dataset.jsonl") -> int:
        """
        Merge all JSONL files in input directory.

        Returns:
            Total number of documents merged
        """
        print("Merging datasets...")

        all_documents = []
        for file_path in self.input_dir.glob("*.jsonl"):
            print(f"Processing {file_path.name}...")
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    doc = json.loads(line)
                    all_documents.append(doc)

        # Save merged dataset
        output_path = self.output_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            for doc in all_documents:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')

        print(f"Merged {len(all_documents)} documents to {output_path}")
        return len(all_documents)

    def get_dataset_statistics(self, dataset_file: str) -> Dict:
        """Get statistics about the merged dataset."""
        file_path = self.output_dir / dataset_file

        total_docs = 0
        total_chars = 0
        sources = {}

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                total_docs += 1

                # Get text content (handle different field names)
                text = doc.get('text') or doc.get('content') or ''
                total_chars += len(text)

                # Count sources
                source = doc.get('source', 'unknown')
                sources[source] = sources.get(source, 0) + 1

        return {
            'total_documents': total_docs,
            'total_characters': total_chars,
            'avg_chars_per_doc': total_chars / total_docs if total_docs > 0 else 0,
            'sources': sources
        }


def main():
    """Example usage of dataset collection tools."""

    # Collect Wikipedia articles
    wiki_collector = WikipediaCollector()
    wiki_articles = wiki_collector.download_wikipedia_subset(
        num_articles=1000,
        min_length=500
    )

    # Collect code samples
    code_collector = CodeCollector()
    code_samples = code_collector.download_code_dataset(
        language="python",
        num_samples=500
    )

    # Merge datasets
    merger = DatasetMerger()
    total_docs = merger.merge_datasets()

    # Get statistics
    stats = merger.get_dataset_statistics("merged_dataset.jsonl")
    print("\nDataset Statistics:")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
