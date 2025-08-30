# 01-Tokenization: Understanding How Text Becomes Tokens

This section covers the fundamental tokenization algorithms used in modern NLP: BPE, WordPiece, and SentencePiece.

## 📋 What You'll Learn

- How BPE (Byte-Pair Encoding) works in GPT models
- How WordPiece tokenization works in BERT models  
- How SentencePiece works in T5 and multilingual models
- Out-of-vocabulary (OOV) word handling
- Building a simple BPE tokenizer from scratch

## 📁 Files

- **`tokenization_tutorial.ipynb`** - Interactive Jupyter notebook
- **`tokenization_simple.py`** - Cross-platform Python script
- **`TOKENIZATION_TUTORIAL.md`** - Complete documentation
- **`requirements_tokenization.txt`** - Dependencies

## 🚀 Quick Start

```bash
cd 01-tokenization
pip install -r requirements_tokenization.txt
python tokenization_simple.py
```

## 📚 Key Concepts

- **BPE**: Character-level fallback, great for OOV handling
- **WordPiece**: Greedy longest-match, good vocab/length balance  
- **SentencePiece**: Language-agnostic, Unicode-friendly

---
**Status**: ✅ Complete - Ready for hands-on learning