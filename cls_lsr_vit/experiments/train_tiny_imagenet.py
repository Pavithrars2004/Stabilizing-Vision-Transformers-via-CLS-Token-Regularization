import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as T
from tqdm import tqdm
import timm


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------
    # MODEL
    # -------------------------
    model = timm.create_model(
        "deit_small_patch16_224",
        pretrained=True,
        num_classes=200
    ).to(device)

    # -------------------------
    # DATA
    # -------------------------
    transform_train = T.Compose([
        T.Resize(224),
        T.RandomCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])

    transform_test = T.Compose([
        T.Resize(224),
        T.ToTensor(),
    ])

    train_ds = datasets.ImageFolder(
        "./data/tiny-imagenet-200/train", transform=transform_train
    )
    val_ds = datasets.ImageFolder(
        "./data/tiny-imagenet-200/val", transform=transform_test
    )

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=4)

    # -------------------------
    # OPTIMIZER
    # -------------------------
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()

    # -------------------------
    # CHECKPOINT SETUP
    # -------------------------
    os.makedirs("experiments/checkpoints", exist_ok=True)
    best_acc = 0.0

    # -------------------------
    # TRAINING LOOP
    # -------------------------
    for epoch in range(args.epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch}")

        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())

        acc = evaluate(model, val_loader)
        print(f"Epoch {epoch} | Tiny-ImageNet Acc: {acc:.2f}%")

        # 🔑 SAVE BEST MODEL
        if acc > best_acc:
            best_acc = acc
            torch.save(
                model.state_dict(),
                "experiments/checkpoints/tiny_imagenet_baseline_best.pth"
            )
            print(f"*** Saved BEST checkpoint: {best_acc:.2f}% ***")

    print(f"\nFinal Best Accuracy: {best_acc:.2f}%")


def evaluate(model, loader):
    model.eval()
    device = next(model.parameters()).device
    correct, total = 0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()
    train(args)
