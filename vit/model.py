"""ViT model components, extracted from notebooks/vit_from_scratch.ipynb."""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Split an image into non-overlapping patches and project each to embed_dim."""

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        x = self.proj(x)       # (B, D, H/P, W/P)
        x = x.flatten(2)       # (B, D, N)
        x = x.transpose(1, 2)  # (B, N, D)
        return x


class ViTEmbedding(nn.Module):
    """Patch embedding + learnable CLS token + learnable positional embedding."""

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=192, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches  # 196

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        # Initialize weights (important for stable training)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Scaled dot-product multi-head self-attention, written by hand."""

    def __init__(self, embed_dim=192, num_heads=3, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, D = x.shape

        # Project and split into Q, K, V: each (B, heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Weighted sum over values, then concatenate heads
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, D)

        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class MLP(nn.Module):
    """Two-layer feed-forward network with GELU."""

    def __init__(self, in_features=192, hidden_features=None, out_features=None, drop=0.1):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        out_features = out_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Stochastic depth: randomly drop the whole residual branch per sample."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # B, 1, 1
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class TransformerBlock(nn.Module):
    """Pre-norm encoder block: x + MHSA(LN(x)), then x + MLP(LN(x))."""

    def __init__(self, embed_dim=192, num_heads=3, mlp_ratio=4.0, drop=0.1, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, attn_drop=drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_features = int(embed_dim * mlp_ratio)
        self.mlp = MLP(in_features=embed_dim, hidden_features=hidden_features, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ViT(nn.Module):
    """ViT-Tiny: 12 layers, 3 heads, embed_dim 192 (~5.53M params with 37 classes)."""

    def __init__(
        self, img_size=224, patch_size=16, in_channels=3, num_classes=37,
        embed_dim=192, num_heads=3, num_layers=12, mlp_ratio=4.0,
        drop=0.1, drop_path_max=0.1,
    ):
        super().__init__()
        self.embedding = ViTEmbedding(
            img_size=img_size, patch_size=patch_size, in_channels=in_channels,
            embed_dim=embed_dim, dropout=drop,
        )

        # Stochastic depth rate increases linearly with depth
        drop_path_rates = torch.linspace(0, drop_path_max, num_layers).tolist()
        self.blocks = nn.Sequential(*[
            TransformerBlock(
                embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop, drop_path=drop_path_rates[i],
            )
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)  # only sees the CLS token

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.embedding(x)  # (B, 197, D)
        x = self.blocks(x)     # (B, 197, D)
        x = self.norm(x)
        cls = x[:, 0]          # (B, D) CLS token only
        return self.head(cls)  # (B, num_classes)
