import argparse
import os
import torch
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
    # MODEL (CLS-LSR ENABLED)
    # -------------------------
    model = DeiTSmall_CLS_LSR(
        lambda_stab=args.lambda_stab,
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
        "./data/tiny-imagenet-200/train",
        transform=transform_train
    )

    val_ds = datasets.ImageFolder(
        "./data/tiny-imagenet-200/val",
        transform=transform_test
    )

    train_loader = DataLoader(
        train_ds, batch_size=128, shuffle=True,
        num_workers=4, pin_memory=True
    )

    val_loader = DataLoader(
        val_ds, batch_size=128, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # -------------------------
    # OPTIMIZATION
    # -------------------------
    optimizer = optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.05
    )

    ema = EMA(model, decay=0.9999)

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
        loop = tqdm(train_loader, desc=f"[CLS-LSR] Epoch {epoch}")

        for imgs, labels in loop:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            # IMPORTANT:
            # labels are INTEGER → model handles CE internally
            loss, ce, stab, _ = model(imgs, labels)

            loss.backward()
            optimizer.step()
            ema.update(model)

            loop.set_postfix(
                loss=f"{loss.item():.3f}",
                ce=f"{ce.item():.3f}",
                stab=f"{stab.item():.3f}"
            )

        acc = evaluate(ema.ema_model, val_loader)
        print(f"Epoch {epoch} | Tiny-ImageNet Acc: {acc:.2f}%")

        # 🔑 SAVE BEST
        if acc > best_acc:
            best_acc = acc
            torch.save(
                ema.ema_model.state_dict(),
                "experiments/checkpoints/tiny_imagenet_clslsr_best.pth"
            )
            print(f"*** Saved BEST CLS-LSR: {best_acc:.2f}% ***")

    print(f"\nFinal Best CLS-LSR Accuracy: {best_acc:.2f}%")


def evaluate(model, loader):
    model.eval()
    device = next(model.parameters()).device
    correct, total = 0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(imgs, None)
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lambda_stab", type=float, default=0.1)
    args = parser.parse_args()

    train(args)
