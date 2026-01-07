import torch
import torchvision.datasets as datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader
import timm

device = "cuda" if torch.cuda.is_available() else "cpu"

model = timm.create_model(
    "deit_small_patch16_224",
    pretrained=False,
    num_classes=200
).to(device)

model.load_state_dict(
    torch.load("experiments/checkpoints/tiny_imagenet_baseline_best.pth")
)
model.eval()

transform = T.Compose([
    T.Resize(224),
    T.ToTensor(),
])

val_ds = datasets.ImageFolder(
    "./data/tiny-imagenet-200/val",
    transform=transform
)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

correct, total = 0, 0

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

acc = 100.0 * correct / total
print(f"\nFinal Tiny-ImageNet Accuracy: {acc:.2f}%")
