import torch
import torchvision
import torchvision.transforms as transforms

import os
from PIL import Image
import matplotlib.pyplot as plt

from model import ResNet18
from utils import load_model, compute_accuracy

# CIFAR-100 class names (the 100-class labels used throughout this script)
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]

# Shared test transform
TEST_TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])


def get_test_loader(batch_size=100, num_workers=4):
    test_set = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=TEST_TRANSFORM
    )
    return torch.utils.data.DataLoader(
        test_set, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )


def load_images_from_folder(test_images_dir='./test_images'):
    test_images_list = sorted([
        f for f in os.listdir(test_images_dir)
        if f.endswith(('.jpg', '.png'))
    ])
    print(f'Found {len(test_images_list)} test images:')
    for img in test_images_list:
        print(f'  - {img}')
    return test_images_list


def predict_image(model, image_path, device, transform):
    """
    Predict class for a single image.
    Returns: (predicted_idx, confidence, all_probs)
    """
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    return predicted_idx.item(), confidence.item(), probabilities[0].cpu()


def test_model_on_custom_images(
    test_images_list,
    loaded_model,
    test_images_dir='./test_images',
    device='cuda',
    test_transform=TEST_TRANSFORM,
):
    print('=' * 70)
    print('TESTING ON CUSTOM IMAGES')
    print('=' * 70 + '\n')

    results = []
    for img_file in test_images_list:
        img_path = os.path.join(test_images_dir, img_file)
        pred_idx, confidence, all_probs = predict_image(
            loaded_model, img_path, device, test_transform
        )
        pred_class = CIFAR100_CLASSES[pred_idx]
        results.append({
            'filename': img_file,
            'predicted_class': pred_class,
            'confidence': confidence,
            'all_probs': all_probs,
        })
        print(
            f'Image: {img_file:<20} | '
            f'Predicted: {pred_class:<20} | '
            f'Confidence: {confidence * 100:.2f}%'
        )

    return results


def visualize(results, test_images_dir='./test_images', save_path='./predictions.png'):
    """Visualize up to 20 predictions in a 4×5 grid with labels in dedicated rows."""
    n = min(len(results), 20)
    cols = 5
    img_rows = (n + cols - 1) // cols  # number of image rows

    # Alternate: image row (height 4) + label row (height 0.6)
    height_ratios = [4, 0.6] * img_rows
    fig, axes = plt.subplots(
        img_rows * 2, cols,
        figsize=(20, img_rows * 5),
        gridspec_kw={'height_ratios': height_ratios, 'hspace': 0.05, 'wspace': 0.05}
    )

    for i in range(n):
        img_row = (i // cols) * 2       # even rows = images
        lbl_row = img_row + 1           # odd rows  = labels
        col = i % cols

        # ── Image ──────────────────────────────────────────────────────────
        ax_img = axes[img_row, col]
        img_path = os.path.join(test_images_dir, results[i]['filename'])
        ax_img.imshow(Image.open(img_path).convert('RGB'))
        ax_img.axis('off')

        # ── Label ──────────────────────────────────────────────────────────
        ax_lbl = axes[lbl_row, col]
        pred_class = results[i]['predicted_class']
        confidence = results[i]['confidence']
        color = 'green' if confidence > 0.8 else 'orange'
        ax_lbl.text(
            0.5, 0.5,
            f"Pred: {pred_class}\nConf: {confidence * 100:.1f}%",
            transform=ax_lbl.transAxes,
            ha='center', va='center',
            fontsize=9, fontweight='bold', color=color
        )
        ax_lbl.axis('off')

    # Hide any unused axes
    for i in range(n, img_rows * cols):
        img_row = (i // cols) * 2
        axes[img_row,     i % cols].axis('off')
        axes[img_row + 1, i % cols].axis('off')

    plt.suptitle(
        'ResNet18 CIFAR-100 Predictions on Test Images',
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f'Plot saved to {save_path}')
    plt.show()
    print('Visualization complete!')


def main(model_path='resnet18.pth',test_images_dir='data\\test_images'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = ResNet18(num_classes=100).to(device)
    model = load_model(model, model_path, device=device)

    # Evaluate on the full CIFAR-100 test set
    test_loader = get_test_loader()
    accs = compute_accuracy(model, test_loader, device)
    print(f"Test accuracy (top-1): {accs[1] * 100:.2f}%")
    print(f"Test accuracy (top-5): {accs[5] * 100:.2f}%")

    # Run inference on custom images
    test_images = load_images_from_folder(test_images_dir)

    results = test_model_on_custom_images(
        test_images_list=test_images,
        loaded_model=model,
        test_images_dir=test_images_dir,
        device=device,
        test_transform=TEST_TRANSFORM,
    )

    visualize(results, test_images_dir=test_images_dir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path', type=str, default='resnet18.pth',
        help='Path to saved model weights'
    )
    args = parser.parse_args()
    main(model_path=args.model_path)