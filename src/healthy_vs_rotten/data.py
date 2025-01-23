"""Data handling utilities for the fruit classification project."""
from pathlib import Path
import shutil
from typing import Tuple
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import typer
from loguru import logger


class FruitVegDataset(Dataset):
    """Dataset for healthy vs rotten fruit/vegetable classification."""

    # Common image formats to support
    SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')

    @classmethod
    def get_image_files(cls, path: Path) -> list:
        """Get all supported image files in a directory."""
        files = []
        for img_format in cls.SUPPORTED_FORMATS:
            files.extend(path.rglob(f"*{img_format}"))
        return files

    def __init__(self, data_path: Path, transform=None) -> None:
        self.data_path = data_path
        self.transform = transform if transform else self._get_default_transforms()

        # Get all image paths and labels
        self.samples = []
        for img_path in self.get_image_files(data_path):
            # Label is 1 for healthy, 0 for rotten
            label = 1 if "healthy" in str(img_path).lower() else 0
            self.samples.append((img_path, label))

        if not self.samples:
            logger.warning(f"No images found in {data_path}. Supported formats: {self.SUPPORTED_FORMATS}")
            logger.info("Please check your data directory structure and image formats.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[index]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

    @staticmethod
    def _get_default_transforms():
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def preprocess(self, output_folder: Path) -> None:
        """Organize data into train/val/test splits."""
        logger.info("Preprocessing data...")
        # Create output directories
        splits = ["train", "val", "test"]
        classes = ["healthy", "rotten"]

        for split in splits:
            for cls in classes:
                (output_folder / split / cls).mkdir(parents=True, exist_ok=True)

        # Split data (80/10/10)
        random.seed(133)
        random.shuffle(self.samples)

        n_samples = len(self.samples)
        train_idx = int(0.8 * n_samples)
        val_idx = int(0.9 * n_samples)

        splits_dict = {
            "train": self.samples[:train_idx],
            "val": self.samples[train_idx:val_idx],
            "test": self.samples[val_idx:],
        }

        # Copy files to new structure
        for split, samples in splits_dict.items():
            for img_path, label in samples:
                cls = "healthy" if label == 1 else "rotten"
                dest = output_folder / split / cls / img_path.name
                shutil.copy2(img_path, dest)
        logger.info("Preprocessing complete.")


def preprocess(
    raw_data_path: Path = typer.Argument("data/raw/Fruit And Vegetable Diseases Dataset"),
    output_folder: Path = typer.Argument("data/processed"),
) -> None:
    """Preprocess the raw data and save it to the output folder."""
    dataset = FruitVegDataset(raw_data_path)
    dataset.preprocess(output_folder)
    logger.info(f'Saved succesfully to folder "{output_folder}"')


if __name__ == "__main__":
    # do python src/healthy_vs_rotten/data.py [raw data folder] [processed data folder]
    typer.run(preprocess)
