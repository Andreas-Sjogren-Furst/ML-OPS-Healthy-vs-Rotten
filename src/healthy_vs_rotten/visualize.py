"""Visualize random samples of healthy and rotten produce.

    Returns:
        _type_: None
"""
from pathlib import Path
import random
import matplotlib.pyplot as plt
import typer
from PIL import Image
import numpy as np

from healthy_vs_rotten.data import FruitVegDataset


def show_images(images, titles, processed=False, figsize=(15, 8)):
    """Helper function to display images in a grid."""
    # Calculate number of rows and columns needed
    n = len(images)
    ncols = 4  # Show 4 images per row
    nrows = (n + ncols - 1) // ncols  # Ceiling division to get enough rows

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.ravel()

    for idx, (img, title) in enumerate(zip(images, titles)):
        if processed:
            # Convert processed tensor back to displayable image
            img = img.permute(1, 2, 0).numpy()
            # Denormalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)

        axes[idx].imshow(img)
        axes[idx].set_title(title)
        axes[idx].axis("off")

    # Hide empty subplots
    for idx in range(len(images), len(axes)):
        axes[idx].axis("off")
        axes[idx].set_visible(False)

    plt.tight_layout()
    return fig


def visualize(
    data_dir: Path = Path("data/processed"),
    num_samples: int = 3,
    save_dir: Path = Path("visualizations"),
    random_seed: int = 42,
) -> None:
    """Visualize random samples of healthy and rotten produce."""
    random.seed(random_seed)
    save_dir.mkdir(exist_ok=True)

    # Get dataset
    dataset = FruitVegDataset(data_dir / "train")

    # Separate healthy and rotten indices
    healthy_idx = [i for i, (_, label) in enumerate(dataset.samples) if label == 1]
    rotten_idx = [i for i, (_, label) in enumerate(dataset.samples) if label == 0]

    # Randomly sample from each class
    healthy_samples = random.sample(healthy_idx, num_samples)
    rotten_samples = random.sample(rotten_idx, num_samples)

    # Get raw images
    raw_images = []
    titles = []
    for idx in healthy_samples + rotten_samples:
        img_path, label = dataset.samples[idx]
        img = Image.open(img_path).convert("RGB")
        raw_images.append(img)
        # Extract fruit/vegetable type from the parent's parent directory
        produce_type = img_path.parent.parent.name  # Going up two levels to get the actual type
        status = "Healthy" if label == 1 else "Rotten"
        titles.append(f"{status}\n{produce_type}")

    # Visualize raw images
    fig_raw = show_images(raw_images, titles, processed=False, figsize=(15, 8))
    fig_raw.suptitle("Raw Images", y=1.02, fontsize=14)
    fig_raw.savefig(save_dir / "raw_samples.png", bbox_inches="tight", dpi=300)

    # Get processed images
    processed_images = []
    for idx in healthy_samples + rotten_samples:
        img, _ = dataset[idx]  # This will apply the transformations
        processed_images.append(img)

    # Visualize processed images
    fig_processed = show_images(processed_images, titles, processed=True, figsize=(15, 8))
    fig_processed.suptitle("Processed Images (Resized + Normalized)", y=1.02, fontsize=14)
    fig_processed.savefig(save_dir / "processed_samples.png", bbox_inches="tight", dpi=300)

    print(f"Visualizations saved to {save_dir}")

    # Show plots if running in interactive environment
    plt.show()


if __name__ == "__main__":
    typer.run(visualize)
