import argparse
import os
import json

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as T
from timm.data.mixup import Mixup
from tqdm import tqdm

from models.deit_small_cls_lsr import DeiTSmall_CLS_LSR
from utils.ema import EMA


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DeiTSmall_CLS_LSR(
        lambda_stab=args.lambda_stab,
        num_classes=100
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

    mixup_fn = Mixup(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        prob=1.0,
        switch_prob=0.5,
        label_smoothing=0.1,
        num_classes=100,
    )

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    ema = EMA(model, decay=0.9999)

    os.makedirs("experiments/checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    best_acc = 0.0
    loss_log = []

    # -------------------------
    # TRAINING
    # -------------------------
    for epoch in range(args.epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"[CLS-LSR] Epoch {epoch}")

        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            imgs, labels = mixup_fn(imgs, labels)

            optimizer.zero_grad()
            loss, ce, stab, _ = model(imgs, labels)
            loss.backward()
            optimizer.step()

            ema.update(model)
            loss_log.append(loss.item())
            loop.set_postfix(loss=loss.item())

        acc = evaluate(ema.ema_model, test_loader)
        print(f"Epoch {epoch} | EMA Acc: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(
                ema.ema_model.state_dict(),
                "experiments/checkpoints/deit_small_clslsr_best.pth",
            )
            print(f"*** Saved BEST CLS-LSR: {best_acc:.2f}% ***")

    # -------------------------
    # SAVE LOSS CURVE
    # -------------------------
    with open("results/cls_lsr_losses.json", "w") as f:
        json.dump(loss_log, f)


def evaluate(model, loader):
    model.eval()
    device = next(model.parameters()).device
    correct, total = 0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs, None)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lambda_stab", type=float, default=0.1)
    args = parser.parse_args()
    train(args)
