import torch


def save_model(model, filepath):
    """Save model weights to filepath."""
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


def load_model(model, filepath, device='cuda'):
    """Load model weights from filepath. map_location ensures CPU-trained
    models load on GPU and vice-versa without crashing."""
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.eval()
    return model


def compute_accuracy(model, dataloader, device):
    """Return top-1 accuracy (0–1) of model over the full dataloader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total if total > 0 else 0.0