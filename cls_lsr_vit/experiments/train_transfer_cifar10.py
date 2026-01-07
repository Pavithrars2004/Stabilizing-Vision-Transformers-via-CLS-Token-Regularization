import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as T
from tqdm import tqdm

from models.deit_small_cls_lsr import DeiTSmall_CLS_LSR
from utils.ema import EMA


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------
    # MODEL (CLS-LSR OFF DURING TRANSFER)
    # -------------------------
    model = DeiTSmall_CLS_LSR(
        lambda_stab=0.0,   # CLS-LSR disabled for transfer
        num_classes=10
    ).to(device)

    # -------------------------
    # LOAD PRETRAINED WEIGHTS (REMOVE HEAD)
    # -------------------------
    ckpt = torch.load(args.pretrained, map_location="cpu")

    # Handle EMA / wrapped checkpoints
    if "model" in ckpt:
        ckpt = ckpt["model"]
    elif "ema" in ckpt:
        ckpt = ckpt["ema"]

    # Remove CIFAR-100 classifier head
    for k in list(ckpt.keys()):
        if "head.weight" in k or "head.bias" in k:
            del ckpt[k]

    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    # -------------------------
    # FREEZE BACKBONE (FIRST 2 EPOCHS)
    # -------------------------
    for name, param in model.named_parameters():
        if "head" not in name:
            param.requires_grad = False

    # -------------------------
    # DATA
    # -------------------------
    transform_train = T.Compose([
        T.Resize(224),
        T.RandomCrop(224, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])

    transform_test = T.Compose([
        T.Resize(224),
        T.ToTensor(),
    ])

    train_ds = datasets.CIFAR10(
        "./data", train=True, download=True, transform=transform_train
    )
    test_ds = datasets.CIFAR10(
        "./data", train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_ds, batch_size=128, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        test_ds, batch_size=128, shuffle=False, num_workers=4
    )

    # -------------------------
    # OPTIMIZATION
    # -------------------------
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,          # low LR = better transfer
        weight_decay=0.05
    )

    criterion = nn.CrossEntropyLoss()
    ema = EMA(model, decay=0.9999)

    # -------------------------
    # TRAINING
    # -------------------------
    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch}")

        # Unfreeze backbone after warm-up
        if epoch == 2:
            for param in model.parameters():
                param.requires_grad = True

        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()

            logits = model(imgs, None)  # CLS-LSR inactive
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()
            ema.update(model)

            loop.set_postfix(loss=loss.item())

        acc = evaluate(ema.clone_model(model), test_loader)
        print(f"Epoch {epoch} | CIFAR-10 Acc: {acc:.2f}%")

        best_acc = max(best_acc, acc)

    print(f"\nFinal CIFAR-10 Accuracy: {best_acc:.2f}%")


def evaluate(model, loader):
    model.eval()
    device = next(model.parameters()).device
    correct, total = 0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs, None).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    train(args)
