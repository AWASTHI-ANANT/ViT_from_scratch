"""Shape and sanity checks mirroring the verification cells in the notebook.

Run with:  pytest -q
"""

import torch

from vit import (
    MLP,
    DropPath,
    MultiHeadSelfAttention,
    PatchEmbedding,
    TransformerBlock,
    ViT,
    ViTEmbedding,
)


def test_patch_embedding_shape():
    out = PatchEmbedding()(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 196, 192)


def test_vit_embedding_adds_cls_token():
    out = ViTEmbedding()(torch.randn(4, 3, 224, 224))
    assert out.shape == (4, 197, 192)


def test_mhsa_preserves_shape():
    out = MultiHeadSelfAttention()(torch.randn(1, 197, 192))
    assert out.shape == (1, 197, 192)


def test_mhsa_rejects_indivisible_heads():
    try:
        MultiHeadSelfAttention(embed_dim=192, num_heads=5)
    except AssertionError:
        return
    raise AssertionError("expected AssertionError for embed_dim % num_heads != 0")


def test_mlp_preserves_shape():
    out = MLP(in_features=192)(torch.randn(1, 196, 192))
    assert out.shape == (1, 196, 192)


def test_transformer_block_preserves_shape():
    out = TransformerBlock(embed_dim=192, num_heads=3)(torch.randn(1, 196, 192))
    assert out.shape == (1, 196, 192)


def test_drop_path_is_identity_in_eval():
    layer = DropPath(0.5).eval()
    x = torch.randn(8, 197, 192)
    assert torch.equal(layer(x), x)


def test_drop_path_zeroes_whole_samples_in_train():
    torch.manual_seed(0)
    layer = DropPath(0.5).train()
    out = layer(torch.ones(64, 4, 4))
    per_sample = out.flatten(1)
    # every sample is either fully dropped (0) or fully kept and rescaled (1 / keep_prob)
    assert all(row.unique().numel() == 1 for row in per_sample)
    assert set(per_sample[:, 0].unique().tolist()) <= {0.0, 2.0}


def test_vit_logits_shape_and_param_count():
    model = ViT(num_classes=37)
    logits = model(torch.randn(4, 3, 224, 224))
    assert logits.shape == (4, 37)
    total_params = sum(p.numel() for p in model.parameters())
    assert round(total_params / 1e6, 2) == 5.53


def test_vit_backward_pass():
    model = ViT(num_classes=37, num_layers=2)
    loss = model(torch.randn(2, 3, 224, 224)).sum()
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
