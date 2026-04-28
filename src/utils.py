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


def compute_accuracy(model, dataloader, device, topk=(1, 5)):
    """Return top-k accuracies (0–1) of model over the full dataloader.
    Returns a dict: {1: top1_acc, 5: top5_acc}
    """
    model.eval()
    correct = {k: 0 for k in topk}
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            max_k = max(topk)
            # Get top-k predicted class indices: shape (batch, max_k)
            _, pred = outputs.topk(max_k, dim=1, largest=True, sorted=True)
            # pred.T is (max_k, batch); compare against labels (1, batch)
            pred = pred.t()
            correct_mat = pred.eq(labels.unsqueeze(0).expand_as(pred))

            for k in topk:
                correct[k] += correct_mat[:k].any(dim=0).sum().item()
            total += labels.size(0)

    return {k: correct[k] / total if total > 0 else 0.0 for k in topk}