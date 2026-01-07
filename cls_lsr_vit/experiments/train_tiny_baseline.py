import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as T
from tqdm import tqdm

# ✅ Correct model import
from models.deit_small_cls_lsr import DeiTSmall_CLS_LSR


def train(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------
    # BASELINE MODEL (CLS-LSR DISABLED)
    # -------------------------------------------------
    model = DeiTSmall_CLS_LSR(
        lambda_stab=0.0,   # 🔴 LSR OFF → baseline
        num_classes=100
    ).to(device)

    # -------------------------------------------------
    # DATA
    # -------------------------------------------------
    transform_train = T.Compose([
        T.Resize(224),
        T.RandomCrop(224, padding=4),
        T.RandomHorizontalFlip(),
        T.RandAugment(2, 9),
        T.ToTensor(),
        T.RandomErasing(p=0.25)
    ])

    transform_test = T.Compose([
        T.Resize(224),
        T.ToTensor(),
    ])

    train_ds = datasets.CIFAR100(
        root="./data", train=True, download=True, transform=transform_train
    )
    test_ds = datasets.CIFAR100(
        root="./data", train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_ds, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True
    )

    # -------------------------------------------------
    # OPTIMIZER
    # -------------------------------------------------
    optimizer = optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=0.05
    )

    best_acc = 0.0
    os.makedirs("experiments/checkpoints", exist_ok=True)

    # -------------------------------------------------
    # TRAIN LOOP
    # -------------------------------------------------
    for epoch in range(args.epochs):

        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch}")

        for imgs, labels in loop:
            imgs = imgs.to(device)
            labels = labels.to(device)

            # 🔧 FIX: convert labels → one-hot (CLS-LSR expects this)
            labels_oh = torch.zeros(
                labels.size(0), 100, device=device
            )
            labels_oh.scatter_(1, labels.unsqueeze(1), 1.0)

            optimizer.zero_grad()

            loss, _, _, _ = model(imgs, labels_oh)
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())

        # -------------------------------------------------
        # EVALUATION
        # -------------------------------------------------
        acc = evaluate(model, test_loader)
        print(f"Epoch {epoch} - Test Acc: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(
                model.state_dict(),
                "experiments/checkpoints/deit_small_baseline_best.pth"
            )
            print(f"*** Saved BEST baseline: {best_acc:.2f}% ***")


def evaluate(model, loader):

    model.eval()
    correct, total = 0, 0
    device = next(model.parameters()).device

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs, None)
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    train(args)
