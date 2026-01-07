import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.deit_small_meanpool import DeiTSmall_MeanPool

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DeiTSmall_MeanPool(num_classes=100).to(device)

    transform = T.Compose([
        T.Resize(224),
        T.RandomHorizontalFlip(),
        T.ToTensor()
    ])

    train_ds = datasets.CIFAR100("./data", train=True, download=True, transform=transform)
    test_ds  = datasets.CIFAR100("./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=128)

    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()
        for x,y in tqdm(train_loader):
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        acc = evaluate(model, test_loader)
        print(f"[MeanPool] Epoch {epoch} Acc: {acc:.2f}%")

def evaluate(model, loader):
    model.eval()
    correct,total = 0,0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.cuda(), y.cuda()
            pred = model(x).argmax(1)
            correct += (pred==y).sum().item()
            total += y.size(0)
    return 100*correct/total

if __name__ == "__main__":
    main()
