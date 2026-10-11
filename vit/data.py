"""Oxford-IIIT Pet data pipeline, matching the notebook."""

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224
BATCH_SIZE = 64

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def get_dataloaders(root="./data", batch_size=BATCH_SIZE, num_workers=2):
    """Return (train_loader, val_loader, class_names). Downloads ~800MB on first run."""
    train_dataset = OxfordIIITPet(root=root, split="trainval", download=True, transform=train_transforms)
    val_dataset = OxfordIIITPet(root=root, split="test", target_types="category", download=True,
                                transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, train_dataset.classes
