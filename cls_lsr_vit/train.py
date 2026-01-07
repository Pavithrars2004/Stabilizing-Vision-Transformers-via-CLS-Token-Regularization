import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.datasets as datasets

from timm.data.mixup import Mixup
from timm.utils import ModelEma
from tqdm import tqdm

from models.vit_cls_lsr import ViT_CLS_LSR
from models.vit_baseline import ViT_Baseline


# -------------------------------------------------------------
# DATA LOADER
# -------------------------------------------------------------
def get_dataloaders(dataset_name, batch_size=128):

    transform_train = T.Compose([
        T.Resize(224),
        T.RandomCrop(224, padding=4),
        T.RandomHorizontalFlip(),
        T.RandAugment(num_ops=2, magnitude=9),
        T.ToTensor(),
        T.RandomErasing(p=0.25, value='random')
    ])

    transform_test = T.Compose([
        T.Resize(224),
        T.ToTensor()
    ])

    if dataset_name == "cifar10":
        train_dataset = datasets.CIFAR10("./datasets", train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10("./datasets", train=False, download=True, transform=transform_test)
        num_classes = 10

    else:
        train_dataset = datasets.CIFAR100("./datasets", train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR100("./datasets", train=False, download=True, transform=transform_test)
        num_classes = 100

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader, num_classes


# -------------------------------------------------------------
# TRAINING LOOP
# -------------------------------------------------------------
def train_model(model_type, model_name, dataset):

    train_loader, test_loader, num_classes = get_dataloaders(dataset, batch_size=128)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------
    # SELECT MODEL
    # -----------------------
    if model_type == "baseline":
        model = ViT_Baseline(model_name=model_name).to(device)
        print("Training BASELINE model...")
    else:
        model = ViT_CLS_LSR(model_name=model_name, num_classes=num_classes, lambda_stab=0.05).to(device)
        print("Training CLS-LSR model...")

    # -----------------------
    # MIXUP SOFT LABELS
    # -----------------------
    mixup_fn = Mixup(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        prob=1.0,
        switch_prob=0.5,
        label_smoothing=0.1,
        num_classes=num_classes,
    )

    # -----------------------
    # OPTIMIZER + EMA
    # -----------------------
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    model_ema = ModelEma(model, decay=0.9999)

    epochs = 1

    # Create checkpoint directory
    os.makedirs("checkpoints", exist_ok=True)
    best_ema_acc = 0.0

    # ---------------------------------------------------------
    # TRAIN
    # ---------------------------------------------------------
    for epoch in range(epochs):

        model.train()
        loop = tqdm(train_loader)
        running_loss = 0

        for imgs, labels in loop:

            imgs, labels = imgs.to(device), labels.to(device)

            # create mixed batch
            mixed_imgs, mixed_labels = mixup_fn(imgs, labels)

            mixed_imgs = mixed_imgs.to(device)
            mixed_labels = mixed_labels.to(device)

            optimizer.zero_grad()

            # TRAINING FORWARD
            loss, ce_loss, stab_loss, logits = model(
                x_clean=imgs,
                x_mix=mixed_imgs,
                targets=mixed_labels
            )

            loss.backward()
            optimizer.step()
            model_ema.update(model)

            running_loss += loss.item()
            loop.set_description(
                f"Epoch {epoch} | Loss: {running_loss/len(loop):.4f}"
            )

        # -----------------------
        # VALIDATION ON EMA
        # -----------------------
        acc = evaluate(model_ema.ema, test_loader)
        print(f"\nEpoch {epoch} - EMA Test Accuracy: {acc:.2f}%\n")

        # -----------------------
        # SAVE CHECKPOINT
        # -----------------------
        ckpt_path = f"checkpoints/{model_name}_clslsr_e{epoch}.pth"

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": model_ema.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "ema_accuracy": acc
        }, ckpt_path)

        # Save BEST model
        if acc > best_ema_acc:
            best_ema_acc = acc
            torch.save(model_ema.ema.state_dict(), f"checkpoints/{model_name}_clslsr_best.pth")
            print(f"**** Saved BEST checkpoint: {best_ema_acc:.2f}% ****")


# -------------------------------------------------------------
# ACCURACY EVAL
# -------------------------------------------------------------
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    device = next(model.parameters()).device

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            logits = model(imgs, None)
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
if __name__ == "__main__":
    train_model(
        model_type="cls_lsr",
        model_name="deit_tiny_patch16_224",
        dataset="cifar100"
    )
