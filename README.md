# Vision Transformer (ViT) Implementation from Scratch

PyTorch implementation of the Vision Transformer architecture built from scratch without external transformer/attention libraries.

## Implementation Details

- **Patch Embedding (`PatchEmbedding`)**: Convolutional projection splitting $224 \times 224$ images into $16 \times 16$ spatial patches ($196$ tokens) with positional encoding.
- **Multi-Head Self-Attention (`MHSA`)**: Custom scaled dot-product attention formulation $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$.
- **Transformer Encoder Block**: Stacked residual self-attention and MLP normalization blocks (`LayerNorm`).
- **Hierarchical & Breed Classification (`notebooks/vit_from_scratch.ipynb`)**: Evaluated on `OxfordIIITPet` for fine-grained classification. Includes hierarchical grouping (binary Cat vs. Dog separation vs multi-class breed recognition).
