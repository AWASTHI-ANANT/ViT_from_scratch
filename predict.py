"""Run inference with a trained checkpoint on one or more images.

Usage:
    python predict.py path/to/image.jpg [more.jpg ...] --checkpoint best_vit_tiny.pth
"""

import argparse

import torch
from PIL import Image

from vit import ViT
from vit.data import val_transforms

# Oxford-IIIT Pet class order (torchvision OxfordIIITPet.classes)
CLASSES = [
    "Abyssinian", "American Bulldog", "American Pit Bull Terrier", "Basset Hound", "Beagle",
    "Bengal", "Birman", "Bombay", "Boxer", "British Shorthair", "Chihuahua", "Egyptian Mau",
    "English Cocker Spaniel", "English Setter", "German Shorthaired", "Great Pyrenees",
    "Havanese", "Japanese Chin", "Keeshond", "Leonberger", "Maine Coon", "Miniature Pinscher",
    "Newfoundland", "Persian", "Pomeranian", "Pug", "Ragdoll", "Russian Blue", "Saint Bernard",
    "Samoyed", "Scottish Terrier", "Shiba Inu", "Siamese", "Sphynx", "Staffordshire Bull Terrier",
    "Wheaten Terrier", "Yorkshire Terrier",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+")
    parser.add_argument("--checkpoint", default="best_vit_tiny.pth")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViT(num_classes=len(CLASSES))
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device).eval()

    for path in args.images:
        x = val_transforms(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = model(x).softmax(dim=-1)[0]
        top = probs.topk(args.top_k)
        print(f"\n{path}")
        for p, idx in zip(top.values.tolist(), top.indices.tolist()):
            print(f"  {CLASSES[idx]:<28} {p * 100:5.1f}%")


if __name__ == "__main__":
    main()
