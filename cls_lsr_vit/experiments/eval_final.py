import argparse
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as T

from models.deit_small_cls_lsr import DeiTSmall_CLS_LSR


def load_model(path, device):
    ckpt = torch.load(path, map_location=device)

    if args.no_ema:
        print(">> Loading RAW model weights (no EMA)")
        model = load_raw_model(args.model, device)
    else:
        print(">> Loading EMA weights")
        model = load_ema_model(args.model, device)

    # EMA checkpoint
    if "ema_state_dict" in ckpt:
        print(">> Loading EMA weights")
        model.load_state_dict(ckpt["ema_state_dict"])

    # Normal checkpoint
    elif "model_state_dict" in ckpt:
        print(">> Loading standard model weights")
        model.load_state_dict(ckpt["model_state_dict"])

    # Raw state_dict (your cls_lsr_best.pth)
    else:
        print(">> Loading raw model weights")
        model.load_state_dict(ckpt)

    model.to(device)
    model.eval()
    return model


def evaluate(model, loader):
    correct, total = 0, 0
    device = next(model.parameters()).device

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs, None)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = T.Compose([
        T.Resize(224),
        T.ToTensor()
    ])

    test_ds = datasets.CIFAR100(
        "./data",
        train=False,
        download=True,
        transform=transform
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=128,
        shuffle=False,
        num_workers=4
    )

    model = load_model(args.model, device)
    acc = evaluate(model, test_loader)

    print(f"\nFinal Test Accuracy: {acc:.2f}%")

def load_raw_model(ckpt_path, device):
    model = DeiTSmall_CLS_LSR(num_classes=100).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)

    # Case 1: full training checkpoint
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)

    # Case 2: EMA-only checkpoint (your case)
    elif isinstance(ckpt, dict) and "ema_state_dict" in ckpt:
        model.load_state_dict(ckpt["ema_state_dict"], strict=False)

    # Case 3: plain state_dict
    else:
        model.load_state_dict(ckpt, strict=False)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--no-ema", action="store_true", help="Evaluate without EMA weights")
    args = parser.parse_args()

    main(args)
