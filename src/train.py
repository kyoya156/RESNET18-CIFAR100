import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from model import ResNet18
from utils import save_model, compute_accuracy

def get_cifar100_loaders(batch_size=128, num_workers=4):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),        # standard CIFAR augmentation
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_set = torchvision.datasets.CIFAR100(root='./data', train=True,
                                               download=True, transform=train_transform)
    test_set  = torchvision.datasets.CIFAR100(root='./data', train=False,
                                               download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True,  num_workers=num_workers,
                                               pin_memory=True)
    test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=batch_size,
                                               shuffle=False, num_workers=num_workers,
                                               pin_memory=True)
    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        correct    += outputs.argmax(1).eq(labels).sum().item()
        total      += inputs.size(0)

    return total_loss / total, 100.0 * correct / total


def main():
    #  Config 
    batch_size  = 64
    num_epochs  = 100
    lr          = 0.1
    save_path   = 'resnet18.pth'

    # Enforce GPU usage
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. GPU is required for training.")
    device = torch.device('cuda')
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Data 
    train_loader, test_loader = get_cifar100_loaders(batch_size=batch_size)

    # Model 
    model = ResNet18(num_classes=100).to(device)

    # Loss / Optimiser / Scheduler 
    criterion = nn.CrossEntropyLoss()
    # SGD + momentum + weight decay consistently outperforms Adam on CIFAR ResNets
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=0.9, weight_decay=5e-4)
    # Cosine annealing: smoothly decays LR → 0 over num_epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training loop 
    best_acc = 0.0
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader,
                                                criterion, optimizer, device)
        test_acc = compute_accuracy(model, test_loader, device) * 100
        scheduler.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}/{num_epochs} | "
                  f"Train loss {train_loss:.3f}  acc {train_acc:.1f}% | "
                  f"Test acc {test_acc:.1f}%")

        # Save best checkpoint
        if test_acc > best_acc:
            best_acc = test_acc
            save_model(model, save_path)

    print(f"\nTraining complete. Best test accuracy: {best_acc:.2f}%")
    print(f"Best model saved to {save_path}")


if __name__ == '__main__':
    main()