import os
import torch
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.optim import AdamW
from timm.data.mixup import Mixup
from tqdm import tqdm

from models.deit_small_cls_lsr import DeiTSmall_CLS_LSR
from utils.ema import EMA

# ------------------------
# Config
# ------------------------
EPOCHS = 50
BATCH_SIZE = 128
LR = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------
# Data
# ------------------------
train_tf = T.Compose([
    T.RandomResizedCrop(224, scale=(0.6, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandAugment(2, 9),
    T.ToTensor(),
    T.RandomErasing(p=0.25)
])

val_tf = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor()
])

train_ds = ImageFolder("data/tiny-imagenet-200/train", transform=train_tf)
val_ds = ImageFolder("data/tiny-imagenet-200/val", transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ------------------------
# Mixup
# ------------------------
mixup = Mixup(
    mixup_alpha=0.8,
    cutmix_alpha=1.0,
    prob=1.0,
    switch_prob=0.5,
    label_smoothing=0.1,
    num_classes=200
)

# ------------------------
# Model + EMA
# ------------------------
model = DeiTSmall_CLS_LSR(num_classes=200, lambda_stab=0.1).to(DEVICE)
ema = EMA(model, decay=0.9999)
opt = AdamW(model.parameters(), lr=LR, weight_decay=0.05)

best_acc = 0
os.makedirs("experiments/checkpoints", exist_ok=True)

# ------------------------
# Training
# ------------------------
for epoch in range(EPOCHS):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for x, y in pbar:
        x, y = x.to(DEVICE), y.to(DEVICE)
        x, y = mixup(x, y)

        loss, _, _, _ = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        ema.update(model)
        pbar.set_postfix(loss=loss.item())

    # ---- Eval EMA ----
    model_ema = ema.clone_model(model)
    model_ema.eval()

    correct = total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model_ema(x, None).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = 100 * correct / total
    print(f"Epoch {epoch} | EMA Acc: {acc:.2f}%")

    if acc > best_acc:
        best_acc = acc
        torch.save(
            ema.ema_model.state_dict(),
            "experiments/checkpoints/deit_small_clslsr_tiny_best.pth"
        )
        print("✔ Saved BEST CLS-LSR")

print("CLS-LSR finished")
