# Results

All numbers come from the training run recorded in [`notebooks/vit_from_scratch.ipynb`](notebooks/vit_from_scratch.ipynb). The per-epoch log is in [`results/training_log.csv`](results/training_log.csv).

**Model:** ViT-Tiny (12 layers, 3 heads, dim 192, patch 16), 5.53M params, trained from scratch
**Data:** Oxford-IIIT Pet, 37 classes (trainval for training, test for evaluation)
**Hardware:** Colab T4 GPU, 100 epochs, about 47 s/epoch, about 78 min total

## Summary

| Metric | Value |
|---|---|
| Best test accuracy | **20.39%** @ epoch 97 (checkpoint saved) |
| Final test accuracy | 19.16% |
| Final train accuracy | 91.31% (peak 92.76% @ epoch 99) |
| Lowest test loss | 3.1826 @ epoch 45 |
| Final test loss | 3.9512 |
| Final train loss | 1.0405 |
| Random-chance baseline | 2.70% (1/37) |
| Generalization gap (final) | ~72 percentage points |

![Training curves](results/training_curves.png)

## Sanity checks (notebook outputs)

| Check | Output |
|---|---|
| Data batch | `[64, 3, 224, 224]`, labels `[64]`, 57 train batches, 37 classes |
| `PatchEmbedding` | `[1, 196, 192]` |
| `ViTEmbedding` (with CLS) | `[4, 197, 192]` |
| `MultiHeadSelfAttention` | `[1, 197, 192]` |
| `MLP` | `[1, 196, 192]` |
| `TransformerBlock` | `[1, 196, 192]` |
| `ViT` logits | `[4, 37]`, **5.53M** parameters |

These checks are automated in `tests/test_model.py`.

## Progress snapshots

| Epoch | Train loss | Train acc | Test loss | Test acc |
|---:|---:|---:|---:|---:|
| 1 | 3.6468 | 3.78% | 3.5840 | 3.76% |
| 10 | 3.3738 | 10.83% | 3.4257 | 8.80% |
| 20 | 3.1796 | 16.53% | 3.2972 | 11.47% |
| 30 | 3.0040 | 20.94% | 3.2495 | 14.06% |
| 40 | 2.7491 | 27.63% | 3.2593 | 16.05% |
| 45 | 2.6144 | 33.00% | **3.1826** | 17.47% |
| 50 | 2.4087 | 38.60% | 3.2684 | 17.96% |
| 60 | 1.9570 | 54.80% | 3.4853 | 17.69% |
| 70 | 1.5418 | 69.96% | 3.8313 | 19.30% |
| 80 | 1.2697 | 81.83% | 3.9028 | 19.00% |
| 90 | 1.1232 | 87.36% | 3.9965 | 19.22% |
| 97 | 1.0565 | 90.32% | 3.9543 | **20.39%** |
| 100 | 1.0405 | 91.31% | 3.9512 | 19.16% |

## Key observations

1. **The model learns real features.** Test accuracy reaches about 7.5× random chance (20.4% vs 2.7%), so the hand-written attention and transformer blocks work.
2. **Phase 1 (epochs 1–45) is healthy learning.** Train and test loss fall together, and test loss bottoms out at 3.18 around epoch 45.
3. **Phase 2 (epochs 45–100) is overfitting.** Train accuracy climbs from 33% to 92%, while test loss rises from 3.18 to about 3.95. Test accuracy plateaus at 18–20%. The model memorizes the 3.7k training images.
4. **Accuracy and loss disagree after epoch 45.** Test accuracy still creeps up slightly while test loss rises. The model grows overconfident on the test images it gets wrong, and this pushes the loss up.
5. **Why it overfits:** ViTs lack the locality and translation inductive biases of CNNs, so they need very large datasets or pretraining. About 100 images per class is far too few to train a 5.5M-parameter ViT from scratch.

## Next steps to improve

- ImageNet-pretrained initialization (the largest expected gain)
- LR warmup and cosine schedule instead of a constant lr of 1e-3
- RandAugment, Mixup/CutMix, and a higher stochastic-depth rate
- Early stopping on test loss (around epoch 45)
