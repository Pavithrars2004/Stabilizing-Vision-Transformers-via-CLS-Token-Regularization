
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.deit_small_cls_lsr import DeiTSmall_CLS_LSR

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- DATA ----------------
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])

dataset = datasets.CIFAR100("./data", train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=128, shuffle=False)

# ---------------- MODEL ----------------
model = DeiTSmall_CLS_LSR(lambda_stab=0.1, num_classes=100).to(device)
ckpt = torch.load("experiments/checkpoints/deit_small_clslsr_best.pth", map_location="cpu")
model.load_state_dict(ckpt, strict=False)
model.eval()

# ---------------- CLS COLLECTION ----------------
cls_tokens = []

with torch.no_grad():
    for i, (imgs, _) in enumerate(loader):
        imgs = imgs.to(device)
        cls = model.get_cls_token(imgs)
        cls_tokens.append(cls.mean(dim=0))

        if i == 20:   # only few batches enough
            break

# ---------------- COSINE SIMILARITY ----------------
sims = []
for i in range(1, len(cls_tokens)):
    sim = F.cosine_similarity(cls_tokens[i], cls_tokens[i-1], dim=0)
    sims.append(sim.item())

np.savetxt("results/cls_similarity_clslsr.txt", sims)

print("CLS similarity saved!")
