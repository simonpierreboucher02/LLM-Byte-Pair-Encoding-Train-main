# LLM Byte Pair Encoding Training

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Made with Love](https://img.shields.io/badge/Made%20with-Love-red.svg)](https://github.com/simonpierreboucher02)
[![Stars](https://img.shields.io/github/stars/simonpierreboucher02/LLM-Byte-Pair-Encoding-Train-main?style=social)](https://github.com/simonpierreboucher02/LLM-Byte-Pair-Encoding-Train-main)
[![Forks](https://img.shields.io/github/forks/simonpierreboucher02/LLM-Byte-Pair-Encoding-Train-main?style=social)](https://github.com/simonpierreboucher02/LLM-Byte-Pair-Encoding-Train-main)

A complete implementation of Byte Pair Encoding (BPE) tokenizer training and GPT model training from scratch using PyTorch and the Hugging Face tokenizers library.

## Overview

This repository contains a full pipeline for training a custom BPE tokenizer and a GPT-style language model from scratch. The implementation includes:

- **Custom BPE Tokenizer Training**: Using Hugging Face's tokenizers library
- **GPT Model Implementation**: A complete transformer-based language model
- **Training Pipeline**: End-to-end training with validation and model saving
- **Text Generation**: Inference capabilities for generating text

## Features

- 🎯 **BPE Tokenizer**: Custom vocabulary training with configurable size
- 🤖 **GPT Architecture**: Multi-head attention, feed-forward networks, and layer normalization
- 📊 **Training Monitoring**: Real-time loss tracking for both training and validation
- 💾 **Model Persistence**: Save and load trained models
- 🚀 **Accelerated Training**: Uses Hugging Face Accelerate for distributed training
- 📝 **Text Generation**: Generate text from trained models

## 📈 Repository Stats

![Repository Size](https://img.shields.io/github/repo-size/simonpierreboucher02/LLM-Byte-Pair-Encoding-Train-main)
![Lines of Code](https://img.shields.io/tokei/lines/github/simonpierreboucher02/LLM-Byte-Pair-Encoding-Train-main)
![Last Commit](https://img.shields.io/github/last-commit/simonpierreboucher02/LLM-Byte-Pair-Encoding-Train-main)
![Issues](https://img.shields.io/github/issues/simonpierreboucher02/LLM-Byte-Pair-Encoding-Train-main)
![Pull Requests](https://img.shields.io/github/issues-pr/simonpierreboucher02/LLM-Byte-Pair-Encoding-Train-main)

## Project Structure

```
LLM-Byte-Pair-Encoding-Train-main/
├── model.py              # GPT model implementation
├── tokenizer_utils.py    # BPE tokenizer training and utilities
├── training_utils.py     # Training helper functions
├── train.py             # Main training script
└── README.md            # This file
```

## Requirements

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.0%2B-green)](https://huggingface.co/transformers/)
[![Tokenizers](https://img.shields.io/badge/Tokenizers-0.13%2B-orange)](https://huggingface.co/docs/tokenizers/)
[![Accelerate](https://img.shields.io/badge/Accelerate-0.20%2B-purple)](https://huggingface.co/docs/accelerate/)

Install the required dependencies:

```bash
pip install torch torchvision torchaudio
pip install transformers tokenizers
pip install accelerate
```

## Usage

### 1. Prepare Your Data

Place your training text file (e.g., `pg2554.txt`) in the project directory. The script expects a text file with UTF-8 encoding.

### 2. Train the Model

Run the training script:

```bash
python train.py
```

The script will:
1. Load and preprocess your text data
2. Train a BPE tokenizer with a vocabulary size of 30,000 tokens
3. Tokenize the text data
4. Train a GPT model with the following default hyperparameters:
   - Batch size: 64
   - Block size: 256
   - Embedding dimension: 384
   - Number of layers: 6
   - Number of attention heads: 6
   - Learning rate: 3e-4
   - Dropout: 0.2

### 3. Model Output

The trained model will be saved to `./scratchGPT/model.pt`, and the script will generate a sample text using the trained model.

## Model Architecture

[![Model Type](https://img.shields.io/badge/Model-GPT--style-brightgreen)](https://arxiv.org/abs/1706.03762)
[![Architecture](https://img.shields.io/badge/Architecture-Transformer-blue)](https://arxiv.org/abs/1706.03762)
[![Attention](https://img.shields.io/badge/Attention-Multi--Head-orange)](https://arxiv.org/abs/1706.03762)
[![Embedding](https://img.shields.io/badge/Embedding-384d-yellow)](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)

The GPT model implementation includes:

- **Token Embeddings**: Convert token IDs to dense vectors
- **Positional Embeddings**: Add positional information
- **Multi-Head Attention**: Self-attention mechanism with multiple heads
- **Feed-Forward Networks**: Two-layer MLPs with ReLU activation
- **Layer Normalization**: Applied before attention and feed-forward layers
- **Residual Connections**: Skip connections for better gradient flow

## Configuration

You can modify the hyperparameters in `train.py`:

```python
batch_size = 64          # Training batch size
block_size = 256         # Context window size
max_iters = 200          # Number of training iterations
learning_rate = 3e-4     # Learning rate
n_embed = 384           # Embedding dimension
n_head = 6              # Number of attention heads
n_layer = 6             # Number of transformer layers
dropout = 0.2           # Dropout rate
```

## Tokenizer Configuration

The BPE tokenizer is configured with:
- Vocabulary size: 30,000 tokens
- Special tokens: `<s>`, `<pad>`, `</s>`, `<unk>`, `<mask>`
- Pre-tokenizer: Whitespace-based
- Post-processor: BERT-style processing
- Decoder: Byte-level decoding

## Training Process

1. **Data Preparation**: Text is loaded and split into training (90%) and validation (10%) sets
2. **Tokenization**: Text is tokenized using the trained BPE tokenizer
3. **Batch Generation**: Random batches are sampled for training
4. **Forward Pass**: Model processes input sequences and computes loss
5. **Backward Pass**: Gradients are computed and parameters are updated
6. **Evaluation**: Loss is estimated on validation data periodically
7. **Model Saving**: Trained model is saved for later use

## Text Generation

After training, the model can generate text by:
1. Starting with a context token
2. Predicting the next token using the trained model
3. Appending the predicted token to the context
4. Repeating until the desired number of tokens is generated

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this implementation.

## License

This project is open source and available under the MIT License.

## Author

**Simon Pierre Boucher**

[![GitHub](https://img.shields.io/badge/GitHub-simonpierreboucher02-black?style=for-the-badge&logo=github)](https://github.com/simonpierreboucher02)
[![Profile Views](https://komarev.com/ghpvc/?username=simonpierreboucher02&color=brightgreen)](https://github.com/simonpierreboucher02)
[![Followers](https://img.shields.io/github/followers/simonpierreboucher02?style=social)](https://github.com/simonpierreboucher02)

- GitHub: [@simonpierreboucher02](https://github.com/simonpierreboucher02)

## Acknowledgments

- Hugging Face for the excellent tokenizers library
- PyTorch team for the deep learning framework
- The transformer architecture paper by Vaswani et al. 