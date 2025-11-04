# Multimodal Models Guide

> **Comprehensive guide to building and using multimodal AI systems that understand vision, language, and audio.**

## Table of Contents
1. [Introduction to Multimodal AI](#introduction)
2. [CLIP: Vision-Language Models](#clip)
3. [Vision Transformers](#vision-transformers)
4. [Audio Processing](#audio-processing)
5. [Multimodal Fusion Techniques](#fusion-techniques)
6. [Practical Applications](#applications)

---

## Introduction to Multimodal AI

### What is Multimodal AI?

Multimodal AI systems process and understand multiple types of data (modalities) simultaneously:
- **Vision**: Images, videos
- **Language**: Text, speech
- **Audio**: Sounds, music, speech
- **Other**: Sensor data, time series, structured data

### Why Multimodal?

1. **Rich Understanding**: Humans perceive the world through multiple senses
2. **Improved Performance**: Multiple modalities provide complementary information
3. **Broader Applications**: Enable new use cases like visual question answering
4. **Robustness**: One modality can compensate when another is noisy

### Key Challenges

- **Alignment**: Different modalities have different representations
- **Fusion**: How to effectively combine information from multiple sources
- **Training**: Requires paired data across modalities
- **Scale**: Larger models and datasets needed
- **Efficiency**: Processing multiple modalities is computationally expensive

---

## CLIP: Vision-Language Models

### Architecture Overview

**CLIP (Contrastive Language-Image Pre-training)** learns to understand images and text in a unified embedding space.

```python
"""
CLIP Architecture:

Text Input → Text Encoder → Text Embedding ─┐
                                             ├→ Contrastive Loss
Image Input → Image Encoder → Image Embedding ┘

Goal: Maximize similarity for matching pairs,
      Minimize similarity for non-matching pairs
"""

# Key Components:
# 1. Image Encoder: ViT (Vision Transformer) or ResNet
# 2. Text Encoder: Transformer (similar to GPT)
# 3. Projection Heads: Map both to same embedding space
# 4. Contrastive Loss: Align matching image-text pairs
```

### How CLIP Works

```python
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Load pre-trained CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Prepare inputs
image = Image.open("cat.jpg")
texts = ["a photo of a cat", "a photo of a dog", "a photo of a bird"]

# Process inputs
inputs = processor(
    text=texts,
    images=image,
    return_tensors="pt",
    padding=True
)

# Get embeddings
with torch.no_grad():
    outputs = model(**inputs)

# Calculate similarity scores
logits_per_image = outputs.logits_per_image  # Image-text similarity
probs = logits_per_image.softmax(dim=1)  # Convert to probabilities

# Results
for text, prob in zip(texts, probs[0]):
    print(f"{text}: {prob:.4f}")
```

### Key Features

1. **Zero-Shot Classification**
   ```python
   def zero_shot_classification(image, candidate_labels, model, processor):
       """Classify image without training on these specific classes."""

       # Create text prompts
       texts = [f"a photo of a {label}" for label in candidate_labels]

       # Process inputs
       inputs = processor(text=texts, images=image, return_tensors="pt")

       # Get predictions
       outputs = model(**inputs)
       probs = outputs.logits_per_image.softmax(dim=1)

       # Return top prediction
       top_idx = probs.argmax()
       return candidate_labels[top_idx], probs[0][top_idx].item()
   ```

2. **Image-Text Retrieval**
   ```python
   def image_text_retrieval(images, texts, model, processor, top_k=5):
       """Find most relevant images for a text query."""

       # Encode all images and texts
       image_inputs = processor(images=images, return_tensors="pt")
       text_inputs = processor(text=texts, return_tensors="pt")

       with torch.no_grad():
           image_embeds = model.get_image_features(**image_inputs)
           text_embeds = model.get_text_features(**text_inputs)

       # Normalize embeddings
       image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
       text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

       # Calculate similarity
       similarity = (text_embeds @ image_embeds.T)

       # Get top-k matches
       top_k_indices = similarity.topk(k=top_k, dim=1).indices
       return top_k_indices
   ```

3. **Visual Search Engine**
   ```python
   import numpy as np
   from typing import List

   class VisualSearchEngine:
       def __init__(self, model_name="openai/clip-vit-base-patch32"):
           self.model = CLIPModel.from_pretrained(model_name)
           self.processor = CLIPProcessor.from_pretrained(model_name)
           self.image_embeddings = None
           self.image_paths = None

       def index_images(self, image_paths: List[str]):
           """Create index of image embeddings."""
           embeddings = []

           for path in image_paths:
               image = Image.open(path)
               inputs = self.processor(images=image, return_tensors="pt")

               with torch.no_grad():
                   embed = self.model.get_image_features(**inputs)
                   embed = embed / embed.norm(dim=-1, keepdim=True)
                   embeddings.append(embed.cpu().numpy())

           self.image_embeddings = np.vstack(embeddings)
           self.image_paths = image_paths

       def search(self, query: str, top_k: int = 5):
           """Search images using text query."""
           # Encode query
           inputs = self.processor(text=query, return_tensors="pt")

           with torch.no_grad():
               query_embed = self.model.get_text_features(**inputs)
               query_embed = query_embed / query_embed.norm(dim=-1, keepdim=True)
               query_embed = query_embed.cpu().numpy()

           # Calculate similarities
           similarities = (query_embed @ self.image_embeddings.T)[0]

           # Get top-k results
           top_indices = np.argsort(similarities)[::-1][:top_k]

           results = [
               {
                   "path": self.image_paths[idx],
                   "score": float(similarities[idx])
               }
               for idx in top_indices
           ]

           return results
   ```

### Fine-Tuning CLIP

```python
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class ImageTextDataset(Dataset):
    def __init__(self, image_paths, captions, processor):
        self.image_paths = image_paths
        self.captions = captions
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        caption = self.captions[idx]

        # Process image and text
        inputs = self.processor(
            text=caption,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )

        # Remove batch dimension
        return {k: v.squeeze(0) for k, v in inputs.items()}

def contrastive_loss(image_embeds, text_embeds, temperature=0.07):
    """CLIP's contrastive loss function."""
    # Normalize embeddings
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)

    # Calculate similarity matrix
    logits = (image_embeds @ text_embeds.T) / temperature

    # Labels are just the diagonal (matching pairs)
    labels = torch.arange(len(logits), device=logits.device)

    # Symmetric loss (image-to-text and text-to-image)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)

    return (loss_i2t + loss_t2i) / 2

def train_clip(model, dataloader, optimizer, epochs=10, device="cuda"):
    """Fine-tune CLIP on custom dataset."""
    model = model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for batch in dataloader:
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)

            # Calculate contrastive loss
            loss = contrastive_loss(
                outputs.image_embeds,
                outputs.text_embeds
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return model
```

---

## Vision Transformers (ViT)

### Architecture

Vision Transformers apply the transformer architecture to images by treating them as sequences of patches.

```python
"""
ViT Architecture:

Image (224x224x3)
    ↓
Patch Embedding (16x16 patches → 196 patches)
    ↓
Linear Projection + Position Embedding
    ↓
Transformer Encoder (12-24 layers)
    ↓
[CLS] Token Output
    ↓
Classification Head
"""

import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """Convert image to sequence of patch embeddings."""

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # Convolutional layer to create patch embeddings
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: (batch_size, channels, height, width)
        x = self.proj(x)  # (batch, embed_dim, n_patches^0.5, n_patches^0.5)
        x = x.flatten(2)  # (batch, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (batch, n_patches, embed_dim)
        return x

class VisionTransformer(nn.Module):
    """Vision Transformer for image classification."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1
    ):
        super().__init__()

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Position embedding
        num_patches = self.patch_embed.n_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]

        # Create patch embeddings
        x = self.patch_embed(x)  # (batch, n_patches, embed_dim)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, n_patches+1, embed_dim)

        # Add position embeddings
        x = x + self.pos_embed

        # Apply transformer
        x = self.transformer(x)

        # Classification from CLS token
        x = self.norm(x[:, 0])  # Take CLS token
        x = self.head(x)

        return x
```

### Using Pre-trained ViT

```python
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image

# Load pre-trained model
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# Prepare image
image = Image.open('cat.jpg')
inputs = processor(images=image, return_tensors="pt")

# Classify
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(-1).item()

print(f"Predicted class: {model.config.id2label[predicted_class]}")
```

---

## Audio Processing

### Whisper for Speech Recognition

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

class AudioTranscriber:
    """Transcribe audio using Whisper."""

    def __init__(self, model_size="base"):
        model_name = f"openai/whisper-{model_size}"
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)

    def transcribe(self, audio_path, language="en"):
        """Transcribe audio file to text."""
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)

        # Process audio
        inputs = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        )

        # Generate transcription
        with torch.no_grad():
            predicted_ids = self.model.generate(
                inputs.input_features,
                language=language,
                task="transcribe"
            )

        # Decode transcription
        transcription = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]

        return transcription

    def translate(self, audio_path, target_language="en"):
        """Translate audio to target language."""
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)

        # Process audio
        inputs = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        )

        # Generate translation
        with torch.no_grad():
            predicted_ids = self.model.generate(
                inputs.input_features,
                task="translate",
                language=target_language
            )

        # Decode translation
        translation = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]

        return translation
```

### Audio Feature Extraction

```python
import librosa
import numpy as np

class AudioFeatureExtractor:
    """Extract features from audio for ML tasks."""

    @staticmethod
    def extract_mel_spectrogram(audio_path, n_mels=128):
        """Extract mel spectrogram."""
        audio, sr = librosa.load(audio_path, sr=22050)
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=n_mels
        )
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db

    @staticmethod
    def extract_mfcc(audio_path, n_mfcc=40):
        """Extract MFCC features."""
        audio, sr = librosa.load(audio_path, sr=22050)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        return mfcc

    @staticmethod
    def extract_chroma(audio_path):
        """Extract chroma features."""
        audio, sr = librosa.load(audio_path, sr=22050)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        return chroma
```

---

## Multimodal Fusion Techniques

### 1. Early Fusion (Feature-Level)

Combine features from different modalities before processing:

```python
class EarlyFusion(nn.Module):
    """Concatenate features early and process together."""

    def __init__(self, vision_dim, text_dim, hidden_dim, num_classes):
        super().__init__()

        # Combined feature dimension
        combined_dim = vision_dim + text_dim

        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, vision_features, text_features):
        # Concatenate features
        combined = torch.cat([vision_features, text_features], dim=-1)

        # Process combined features
        output = self.fusion(combined)
        return output
```

### 2. Late Fusion (Decision-Level)

Process each modality separately, then combine predictions:

```python
class LateFusion(nn.Module):
    """Process modalities separately, combine predictions."""

    def __init__(self, vision_dim, text_dim, hidden_dim, num_classes):
        super().__init__()

        # Separate networks for each modality
        self.vision_net = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

        self.text_net = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, vision_features, text_features):
        # Get predictions from each modality
        vision_logits = self.vision_net(vision_features)
        text_logits = self.text_net(text_features)

        # Average predictions (can also use weighted average)
        combined_logits = (vision_logits + text_logits) / 2
        return combined_logits
```

### 3. Cross-Modal Attention

Use attention to align information across modalities:

```python
class CrossModalAttention(nn.Module):
    """Cross-attention between vision and text."""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, query_modality, key_value_modality):
        """
        Args:
            query_modality: Features from one modality (e.g., text)
            key_value_modality: Features from another modality (e.g., vision)
        """
        # Cross-attention
        attn_output, attn_weights = self.multihead_attn(
            query=query_modality,
            key=key_value_modality,
            value=key_value_modality
        )

        # Residual connection and normalization
        output = self.norm(query_modality + attn_output)

        return output, attn_weights

class BimodalTransformer(nn.Module):
    """Transformer with cross-modal attention between vision and text."""

    def __init__(self, dim=768, num_heads=12, num_layers=6):
        super().__init__()

        # Cross-attention layers
        self.vision_to_text_layers = nn.ModuleList([
            CrossModalAttention(dim, num_heads)
            for _ in range(num_layers)
        ])

        self.text_to_vision_layers = nn.ModuleList([
            CrossModalAttention(dim, num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, vision_features, text_features):
        """
        Args:
            vision_features: (batch, seq_len_v, dim)
            text_features: (batch, seq_len_t, dim)
        """
        v_feats = vision_features
        t_feats = text_features

        for v2t_layer, t2v_layer in zip(
            self.vision_to_text_layers,
            self.text_to_vision_layers
        ):
            # Text attends to vision
            t_feats, _ = v2t_layer(t_feats, v_feats)

            # Vision attends to text
            v_feats, _ = t2v_layer(v_feats, t_feats)

        return v_feats, t_feats
```

---

## Practical Applications

### Visual Question Answering (VQA)

```python
from transformers import ViltProcessor, ViltForQuestionAnswering

class VQASystem:
    """Visual Question Answering system."""

    def __init__(self):
        self.processor = ViltProcessor.from_pretrained(
            "dandelin/vilt-b32-finetuned-vqa"
        )
        self.model = ViltForQuestionAnswering.from_pretrained(
            "dandelin/vilt-b32-finetuned-vqa"
        )

    def answer_question(self, image, question):
        """Answer a question about an image."""
        # Prepare inputs
        encoding = self.processor(image, question, return_tensors="pt")

        # Get answer
        with torch.no_grad():
            outputs = self.model(**encoding)
            logits = outputs.logits
            idx = logits.argmax(-1).item()

        # Decode answer
        answer = self.model.config.id2label[idx]
        return answer
```

### Image Captioning

```python
from transformers import BlipProcessor, BlipForConditionalGeneration

class ImageCaptioner:
    """Generate captions for images."""

    def __init__(self):
        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )

    def generate_caption(self, image, max_length=50):
        """Generate a caption for an image."""
        # Prepare inputs
        inputs = self.processor(image, return_tensors="pt")

        # Generate caption
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=5
            )

        # Decode caption
        caption = self.processor.decode(output[0], skip_special_tokens=True)
        return caption

    def conditional_caption(self, image, text_prompt):
        """Generate caption conditioned on a text prompt."""
        inputs = self.processor(image, text_prompt, return_tensors="pt")

        with torch.no_grad():
            output = self.model.generate(**inputs)

        caption = self.processor.decode(output[0], skip_special_tokens=True)
        return caption
```

---

## Best Practices

### 1. Data Preparation
- Ensure modalities are properly aligned (synchronized timestamps, matching pairs)
- Normalize features appropriately for each modality
- Handle missing modalities gracefully

### 2. Model Design
- Start simple (early/late fusion) before trying complex architectures
- Use pre-trained encoders when possible
- Consider computational costs of processing multiple modalities

### 3. Training
- Balance loss contributions from different modalities
- Use separate learning rates for different components
- Monitor performance on each modality independently

### 4. Evaluation
- Test on diverse datasets
- Evaluate cross-modal retrieval performance
- Check for modality bias (over-reliance on one modality)

### 5. Deployment
- Optimize inference for each modality separately
- Cache embeddings when possible
- Consider edge cases where one modality is unavailable

---

## Resources

- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [ViT Paper](https://arxiv.org/abs/2010.11929)
- [Whisper Paper](https://arxiv.org/abs/2212.04356)
- [Hugging Face Multimodal](https://huggingface.co/tasks/visual-question-answering)
- [OpenAI CLIP GitHub](https://github.com/openai/CLIP)

---

**Next**: [Responsible AI Guide](./02-responsible-ai.md)
