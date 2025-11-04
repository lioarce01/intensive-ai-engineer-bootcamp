"""
Example: Using CLIP for Image-Text Understanding

This example demonstrates:
1. Loading CLIP model
2. Zero-shot image classification
3. Image-text similarity
4. Building a visual search engine
"""

import torch
from PIL import Image
import numpy as np
from pathlib import Path
from typing import List, Dict
import clip

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


def load_model(model_name: str = "ViT-B/32", device: str = None):
    """
    Load CLIP model and preprocessing function.

    Args:
        model_name: Model architecture (ViT-B/32, ViT-B/16, ViT-L/14)
        device: Device to load model on (cuda/cpu)

    Returns:
        model, preprocess: Loaded model and preprocessing function
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading CLIP model: {model_name} on {device}")
    model, preprocess = clip.load(model_name, device=device)

    return model, preprocess, device


def zero_shot_classification(
    image_path: str,
    candidate_labels: List[str],
    model,
    preprocess,
    device: str
) -> Dict[str, float]:
    """
    Classify an image using zero-shot classification.

    Args:
        image_path: Path to image
        candidate_labels: List of possible labels
        model: CLIP model
        preprocess: CLIP preprocessing function
        device: Device

    Returns:
        Dictionary of labels and their probabilities
    """
    # Load and preprocess image
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    # Prepare text prompts
    text_prompts = [f"a photo of a {label}" for label in candidate_labels]
    text = clip.tokenize(text_prompts).to(device)

    # Get predictions
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        # Normalize features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Calculate similarity
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    # Create results dictionary
    results = {
        label: float(prob)
        for label, prob in zip(candidate_labels, similarity[0])
    }

    # Sort by probability
    results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

    return results


def compute_image_text_similarity(
    image_path: str,
    text_descriptions: List[str],
    model,
    preprocess,
    device: str
) -> List[Dict[str, any]]:
    """
    Compute similarity between an image and multiple text descriptions.

    Args:
        image_path: Path to image
        text_descriptions: List of text descriptions
        model: CLIP model
        preprocess: CLIP preprocessing function
        device: Device

    Returns:
        List of dictionaries with text and similarity scores
    """
    # Load and preprocess image
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    # Tokenize text
    text = clip.tokenize(text_descriptions).to(device)

    # Compute features
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        # Normalize
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Compute similarity
        similarity = (image_features @ text_features.T)[0]

    # Create results
    results = [
        {
            "text": text_descriptions[i],
            "similarity": float(similarity[i]),
            "percentage": float(similarity[i] * 100)
        }
        for i in range(len(text_descriptions))
    ]

    # Sort by similarity
    results.sort(key=lambda x: x["similarity"], reverse=True)

    return results


class VisualSearchEngine:
    """
    Visual search engine using CLIP embeddings.

    Index images and search using text queries or other images.
    """

    def __init__(self, model_name: str = "ViT-B/32"):
        self.model, self.preprocess, self.device = load_model(model_name)
        self.image_embeddings = []
        self.image_paths = []
        self.indexed = False

    def index_images(self, image_paths: List[str], batch_size: int = 32):
        """
        Index a collection of images.

        Args:
            image_paths: List of paths to images
            batch_size: Batch size for processing
        """
        print(f"Indexing {len(image_paths)} images...")

        embeddings = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]

            # Load and preprocess batch
            images = []
            valid_paths = []

            for path in batch_paths:
                try:
                    img = self.preprocess(Image.open(path)).unsqueeze(0)
                    images.append(img)
                    valid_paths.append(path)
                except Exception as e:
                    print(f"Error loading {path}: {e}")

            if not images:
                continue

            # Batch process
            images_batch = torch.cat(images).to(self.device)

            with torch.no_grad():
                batch_embeddings = self.model.encode_image(images_batch)
                batch_embeddings /= batch_embeddings.norm(dim=-1, keepdim=True)
                embeddings.append(batch_embeddings.cpu())

            self.image_paths.extend(valid_paths)

            print(f"Processed {len(self.image_paths)}/{len(image_paths)} images")

        # Concatenate all embeddings
        if embeddings:
            self.image_embeddings = torch.cat(embeddings)
            self.indexed = True
            print(f"✓ Indexing complete: {len(self.image_paths)} images")
        else:
            print("✗ No images indexed")

    def search_by_text(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search images using text query.

        Args:
            query: Text search query
            top_k: Number of results to return

        Returns:
            List of results with image path and similarity score
        """
        if not self.indexed:
            raise ValueError("No images indexed. Call index_images() first.")

        # Encode query
        text = clip.tokenize([query]).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.cpu()

        # Calculate similarities
        similarities = (text_features @ self.image_embeddings.T)[0]

        # Get top-k
        top_indices = similarities.argsort(descending=True)[:top_k]

        results = [
            {
                "rank": i + 1,
                "path": self.image_paths[idx],
                "similarity": float(similarities[idx]),
                "score_percentage": float(similarities[idx] * 100)
            }
            for i, idx in enumerate(top_indices)
        ]

        return results

    def search_by_image(self, image_path: str, top_k: int = 5) -> List[Dict]:
        """
        Search similar images using an image query.

        Args:
            image_path: Path to query image
            top_k: Number of results to return

        Returns:
            List of results with image path and similarity score
        """
        if not self.indexed:
            raise ValueError("No images indexed. Call index_images() first.")

        # Encode query image
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            query_features = self.model.encode_image(image)
            query_features /= query_features.norm(dim=-1, keepdim=True)
            query_features = query_features.cpu()

        # Calculate similarities
        similarities = (query_features @ self.image_embeddings.T)[0]

        # Get top-k (excluding the query image itself if it's in the index)
        top_indices = similarities.argsort(descending=True)[:top_k + 1]

        results = []
        rank = 1
        for idx in top_indices:
            # Skip if it's the query image itself
            if self.image_paths[idx] == image_path:
                continue

            results.append({
                "rank": rank,
                "path": self.image_paths[idx],
                "similarity": float(similarities[idx]),
                "score_percentage": float(similarities[idx] * 100)
            })

            rank += 1
            if len(results) >= top_k:
                break

        return results


def main():
    """Main example demonstrating CLIP usage."""

    print("=" * 60)
    print("CLIP Image-Text Understanding Examples")
    print("=" * 60)

    # Load model
    model, preprocess, device = load_model()

    # Example 1: Zero-shot classification
    print("\n📸 Example 1: Zero-Shot Image Classification")
    print("-" * 60)

    # Note: You would need actual images for this to work
    # image_path = "path/to/your/image.jpg"
    # candidate_labels = ["cat", "dog", "bird", "car", "building"]
    # results = zero_shot_classification(
    #     image_path, candidate_labels, model, preprocess, device
    # )
    #
    # print(f"Image: {image_path}")
    # print("\nClassification results:")
    # for label, prob in results.items():
    #     print(f"  {label:15s}: {prob:.2%}")

    print("To run this example, provide an image path and labels")

    # Example 2: Image-text similarity
    print("\n📝 Example 2: Image-Text Similarity")
    print("-" * 60)

    # text_descriptions = [
    #     "a photo of a sunset over the ocean",
    #     "a person walking in a park",
    #     "a cat sleeping on a couch",
    #     "a mountain landscape with trees"
    # ]
    #
    # results = compute_image_text_similarity(
    #     image_path, text_descriptions, model, preprocess, device
    # )
    #
    # print(f"Image: {image_path}")
    # print("\nSimilarity scores:")
    # for result in results:
    #     print(f"  {result['percentage']:5.1f}% - {result['text']}")

    print("To run this example, provide an image path and text descriptions")

    # Example 3: Visual search engine
    print("\n🔍 Example 3: Visual Search Engine")
    print("-" * 60)

    # search_engine = VisualSearchEngine()
    #
    # # Index images
    # image_directory = Path("path/to/image/directory")
    # image_paths = list(image_directory.glob("*.jpg"))
    # search_engine.index_images([str(p) for p in image_paths])
    #
    # # Search by text
    # query = "sunset over mountains"
    # results = search_engine.search_by_text(query, top_k=5)
    #
    # print(f"Query: '{query}'")
    # print("\nTop 5 results:")
    # for result in results:
    #     print(f"  {result['rank']}. {result['path']} ({result['score_percentage']:.1f}%)")

    print("To run this example, provide a directory of images and a search query")

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
