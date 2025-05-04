# Transformer from Scratch for Text Classification

Implementing a Transformer encoder in PyTorch for sentiment analysis (SST-2 dataset). No high-level libraries—just pure PyTorch!

## 🛠️ Tech Stack
- **PyTorch** (Custom `nn.Module` implementations)
- **Hugging Face** `datasets` and `tokenizers` (for data loading)
- **Seqeval** (for evaluation)

## 🧩 Key Components
1. **Multi-Head Attention**: Scaled dot-product attention with masking.
2. **Positional Encoding**: Sinusoidal embeddings for sequence order.
3. **Transformer Block**: Residual connections + LayerNorm.
4. **Encoder**: Stacks Transformer blocks for classification.

## 📊 Results
| Metric       | Score  |
|--------------|--------|
| Accuracy     | 85%    |
| Training Time| ~1h/epoch (CPU) |

## 🚀 How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/transformer-from-scratch.git
