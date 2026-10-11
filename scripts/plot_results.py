"""Plot loss/accuracy curves from results/training_log.csv -> results/training_curves.png."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parent.parent / "results"

with open(RESULTS / "training_log.csv") as f:
    rows = list(csv.DictReader(f))

epochs = [int(r["epoch"]) for r in rows]
col = lambda k: [float(r[k]) for r in rows]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
ax1.plot(epochs, col("train_loss"), label="Train")
ax1.plot(epochs, col("val_loss"), label="Validation")
ax1.set(title="Loss (CE, label smoothing 0.1)", xlabel="Epoch", ylabel="Loss")
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(epochs, col("train_acc"), label="Train")
ax2.plot(epochs, col("val_acc"), label="Validation")
ax2.axhline(100 / 37, color="gray", ls="--", lw=1, label="Random chance (2.7%)")
ax2.set(title="Top-1 accuracy (%)", xlabel="Epoch", ylabel="Accuracy (%)")
ax2.legend()
ax2.grid(alpha=0.3)

fig.suptitle("ViT-Tiny from scratch on Oxford-IIIT Pet (37 classes)")
fig.tight_layout()
fig.savefig(RESULTS / "training_curves.png", dpi=150)
print(f"Saved {RESULTS / 'training_curves.png'}")
