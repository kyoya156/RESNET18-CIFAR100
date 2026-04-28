import torch
import torchvision
import torchvision.transforms as transforms

from model import ResNet18
from utils import load_model, compute_accuracy

def get_test_loader(batch_size=100, num_workers=4):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_set = torchvision.datasets.CIFAR100(root='./data', train=False,
                                              download=True, transform=transform)
    return torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                       shuffle=False, num_workers=num_workers,
                                       pin_memory=True)


def main(model_path='resnet18_cifar100.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Build model, then load weights — consistent with utils.load_model signature
    model = ResNet18(num_classes=100).to(device)
    model = load_model(model, model_path, device=device)

    test_loader = get_test_loader()

    accuracy = compute_accuracy(model, test_loader, device) * 100
    print(f"Test accuracy: {accuracy:.2f}%")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='resnet18_cifar100.pth',
                        help='Path to saved model weights')
    args = parser.parse_args()
    main(model_path=args.model_path)