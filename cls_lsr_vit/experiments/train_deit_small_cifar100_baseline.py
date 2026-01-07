import argparse
import os
import json

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as T
from tqdm import tqdm
import timm


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = timm.create_model(
        "deit_small_patch16_224",
        pretrained=True,
        num_classes=100,
    ).to(device)

    # -------------------------
    # DATA
    # -------------------------
    transform_train = T.Compose([
        T.Resize(224),
        T.RandomCrop(224, padding=4),
        T.RandomHorizontalFlip(),
        T.RandAugment(2, 9),
        T.ToTensor(),
        T.RandomErasing(p=0.25),
    ])

    transform_test = T.Compose([
        T.Resize(224),
        T.ToTensor(),
    ])

    train_ds = datasets.CIFAR100("./data", train=True, download=True, transform=transform_train)
    test_ds = datasets.CIFAR100("./data", train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4)

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    os.makedirs("experiments/checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    best_acc = 0.0
    loss_log = []

    # -------------------------
    # TRAINING
    # -------------------------
    for epoch in range(args.epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"[Baseline] Epoch {epoch}")

        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            loss_log.append(loss.item())
            loop.set_postfix(loss=loss.item())

        acc = evaluate(model, test_loader)
        print(f"Epoch {epoch} | Test Acc: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(
                model.state_dict(),
                "experiments/checkpoints/deit_small_baseline_best.pth",
            )
            print(f"*** Saved BEST Baseline: {best_acc:.2f}% ***")

    # -------------------------
    # SAVE LOSS CURVE
    # -------------------------
    with open("results/baseline_losses.json", "w") as f:
        json.dump(loss_log, f)


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
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    train(args)
