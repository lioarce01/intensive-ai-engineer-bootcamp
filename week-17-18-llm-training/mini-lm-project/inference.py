"""
Inference Script for Mini-LM

Generate text using trained Mini-LM model.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import argparse
from typing import List, Optional

from model import MiniLM, ModelConfig
from tokenizers import Tokenizer


class TextGenerator:
    """Text generation with Mini-LM."""

    def __init__(
        self,
        model: MiniLM,
        tokenizer: Tokenizer,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.0
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens (None to disable)
            top_p: Nucleus sampling threshold (None to disable)
            repetition_penalty: Penalty for repeating tokens

        Returns:
            Generated text
        """
        # Tokenize prompt
        encoding = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([encoding.ids], dtype=torch.long).to(self.device)

        # Generate tokens
        for _ in range(max_new_tokens):
            # Get model predictions
            outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :]  # Last token logits

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(input_ids[0].tolist()):
                    logits[0, token_id] /= repetition_penalty

            # Apply temperature
            logits = logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Stop if EOS token
            if next_token.item() == self.tokenizer.token_to_id("</s>"):
                break

            # Stop if sequence too long
            if input_ids.size(1) >= self.model.config.max_position_embeddings:
                break

        # Decode generated text
        generated_ids = input_ids[0].tolist()
        generated_text = self.tokenizer.decode(generated_ids)

        return generated_text

    def generate_batch(
        self,
        prompts: List[str],
        **generation_kwargs
    ) -> List[str]:
        """Generate text for multiple prompts."""
        return [self.generate(prompt, **generation_kwargs) for prompt in prompts]


def load_model_from_checkpoint(checkpoint_path: str, device: str = 'cuda') -> MiniLM:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config from checkpoint or use saved config
    if 'config' in checkpoint:
        config = checkpoint['config']
        model_config = ModelConfig()  # Use defaults and override if needed
    else:
        raise ValueError("No config found in checkpoint")

    # Create model
    from model import create_mini_lm
    model = create_mini_lm(config.model_name if hasattr(config, 'model_name') else 'mini-50m')

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


def interactive_generation(generator: TextGenerator):
    """Interactive text generation loop."""
    print("\n" + "=" * 60)
    print("Interactive Text Generation")
    print("Type 'quit' to exit")
    print("=" * 60 + "\n")

    while True:
        prompt = input("Prompt: ").strip()

        if prompt.lower() in ['quit', 'exit', 'q']:
            break

        if not prompt:
            continue

        print("\nGenerating...")
        generated = generator.generate(
            prompt,
            max_new_tokens=100,
            temperature=0.8,
            top_k=50,
            top_p=0.9
        )

        print(f"\nGenerated text:\n{generated}\n")
        print("-" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate text with Mini-LM")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--tokenizer', type=str, required=True,
                        help='Path to tokenizer')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Text prompt (if not provided, enter interactive mode)')
    parser.add_argument('--max-tokens', type=int, default=100,
                        help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=50,
                        help='Top-k sampling')
    parser.add_argument('--top-p', type=float, default=0.9,
                        help='Top-p (nucleus) sampling')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model_from_checkpoint(args.checkpoint, args.device)

    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer}...")
    tokenizer = Tokenizer.from_file(args.tokenizer)

    # Create generator
    generator = TextGenerator(model, tokenizer, args.device)

    # Generate
    if args.prompt:
        # Single generation
        generated = generator.generate(
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        print(f"\nGenerated text:\n{generated}")
    else:
        # Interactive mode
        interactive_generation(generator)


if __name__ == "__main__":
    main()
