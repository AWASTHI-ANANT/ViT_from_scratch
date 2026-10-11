"""Train ViT-Tiny from scratch on Oxford-IIIT Pet (same setup as the notebook).

Usage:
    python train.py --epochs 100 --lr 1e-3
"""

import argparse
import csv
import time

import torch
import torch.nn as nn
import torch.optim as optim

from vit import ViT
from vit.data import get_dataloaders


def run_epoch(model, loader, criterion, device, use_amp, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            with torch.amp.autocast(device.type, enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            if training:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)  # unscale before gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()
            correct += outputs.argmax(1).eq(targets).sum().item()
            total += targets.size(0)

    return total_loss / len(loader), 100. * correct / total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--checkpoint", default="best_vit_tiny.pth")
    parser.add_argument("--log", default="results/training_log.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Training on device: {device}")

    train_loader, val_loader, classes = get_dataloaders(args.data_root, args.batch_size)
    model = ViT(num_classes=len(classes)).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_val_acc = 0.0
    with open(args.log, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "time_s", "train_loss", "train_acc", "val_loss", "val_acc"])

        for epoch in range(1, args.epochs + 1):
            start = time.time()
            train_loss, train_acc = run_epoch(model, train_loader, criterion, device, use_amp, optimizer, scaler)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, device, use_amp)
            elapsed = time.time() - start

            print(f"Epoch [{epoch}/{args.epochs}] | Time: {elapsed:.0f}s")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%\n")
            writer.writerow([epoch, f"{elapsed:.0f}", f"{train_loss:.4f}", f"{train_acc:.2f}",
                             f"{val_loss:.4f}", f"{val_acc:.2f}"])
            f.flush()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), args.checkpoint)
                print("--> Saved new best checkpoint!\n")

    print(f"Training Complete. Best Validation Accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
