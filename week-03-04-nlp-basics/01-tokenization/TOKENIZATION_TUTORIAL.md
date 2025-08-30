# Interactive Tokenization Tutorial

This tutorial provides hands-on experience with the three most important tokenization algorithms used in modern NLP: **BPE**, **WordPiece**, and **SentencePiece**.

## Overview

Learn how tokenization works in practice with real examples from GPT-2, BERT, and T5 models. The tutorial focuses on engineering understanding rather than theory, showing you exactly how these algorithms split text, handle out-of-vocabulary words, and create token sequences.

## Files Included

1. **`tokenization_tutorial.ipynb`** - Interactive Jupyter notebook with comprehensive examples
2. **`tokenization_simple.py`** - Standalone Python script that runs on all platforms
4. **`requirements_tokenization.txt`** - Required Python packages

## Quick Start

### Option 1: Jupyter Notebook (Recommended)
```bash
# Navigate to the week folder
cd week-03-04-nlp-basics

# Install dependencies
pip install -r requirements_tokenization.txt

# Start Jupyter and open the notebook
jupyter notebook tokenization_tutorial.ipynb
```

### Option 2: Python Script
```bash
# Navigate to the week folder
cd week-03-04-nlp-basics

# Install dependencies
pip install torch transformers

# Run the full tutorial
python tokenization_simple.py

# Or run in interactive mode
python tokenization_simple.py --interactive
```

## What You'll Learn

### Core Concepts
- How BPE (Byte-Pair Encoding) works in GPT models
- How WordPiece tokenization works in BERT models  
- How SentencePiece works in T5 and multilingual models
- Vocabulary building processes for each algorithm
- Handling of out-of-vocabulary (OOV) words

### Practical Skills
- Using Hugging Face tokenizers in practice
- Comparing tokenization outputs across algorithms
- Understanding token IDs and decoding
- Batch processing with padding and truncation
- Exploring tokenizer vocabularies
- Building a simple BPE tokenizer from scratch

## Tutorial Structure

### Part 1: Fundamentals
- Introduction to tokenization concepts
- Loading and using pre-trained tokenizers
- Basic tokenization examples

### Part 2: BPE Deep Dive (GPT-2)
- How BPE merges character pairs iteratively
- Space handling with special symbols
- Character-level fallback for unknown words

### Part 3: WordPiece Analysis (BERT)
- Greedy longest-match approach
- Subword continuation markers (##)
- Special tokens: [CLS], [SEP], [UNK], etc.

### Part 4: SentencePiece Exploration (T5)
- Language-agnostic Unicode handling
- Word boundary markers (▁)
- Unigram language model approach

### Part 5: Comparative Analysis
- Side-by-side tokenization comparisons
- Token count analysis and visualization
- Performance trade-offs

### Part 6: OOV Handling
- Testing with challenging words
- Comparing fallback strategies
- Information preservation vs. compression

### Part 7: Building BPE from Scratch
- Implementing the core BPE algorithm
- Understanding merge operations
- Training on custom text

### Part 8: Practical Applications
- Batch processing techniques
- Sequence length considerations
- Memory and efficiency tips

## Key Examples

The tutorial includes analysis of these text examples:

**Basic Examples:**
- "Hello, world!"
- "The transformer architecture revolutionized NLP"
- "Machine learning models need to understand text"

**Challenging Cases:**
- "COVID-19 pandemic started in 2020"
- "GPT-3.5-turbo and GPT-4 are powerful models"  
- "supercalifragilisticexpialidocious"
- URLs, emails, and technical terms

## Learning Outcomes

After completing this tutorial, you will:

1. **Understand** how modern tokenizers work under the hood
2. **Compare** the strengths and weaknesses of each algorithm
3. **Apply** tokenizers effectively in your NLP projects
4. **Debug** tokenization issues in real applications
5. **Choose** the right tokenizer for your specific use case

## Best Practices Covered

- Always use the same tokenizer for training and inference
- Handle special tokens appropriately
- Consider sequence length limits
- Test on edge cases and domain-specific text
- Understand vocabulary size vs. sequence length trade-offs
- Use batch processing for efficiency
- Validate tokenization pipelines thoroughly

## Technical Requirements

- Python 3.8+ (tested with 3.13.5)
- PyTorch 2.0+ (tested with 2.8.0)
- Transformers 4.21+ (tested with 4.56.0)
- Basic familiarity with Python and NLP concepts

## Troubleshooting

### Windows Unicode Issues
If you see Unicode encoding errors on Windows, use `tokenization_simple.py` instead of the full demo script.

### Memory Issues
The tutorial downloads small models (GPT-2, BERT-base, T5-small). If you have memory constraints, the examples will still work but may load slower.

### Import Errors
Make sure all required packages are installed:
```bash
pip install torch transformers tokenizers matplotlib pandas
```

## Next Steps

After completing this tutorial:

1. **Experiment** with domain-specific text in your field
2. **Train** custom tokenizers on your own data
3. **Explore** other algorithms like Unigram and SentencePiece variants
4. **Learn** about multilingual tokenization challenges
5. **Study** how tokenization affects model performance

## Contributing

This tutorial is designed for the AI Intensive Bootcamp weeks 3-4, focusing on practical NLP engineering skills. If you find issues or have suggestions for improvements, please provide feedback.

---

**Happy tokenizing!** 🚀

Remember: Understanding tokenization is crucial for working with any modern NLP model. Take your time with the examples and experiment with your own text!