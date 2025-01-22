"""
This script loads a trained model and predicts whether the fruit/vegetable is healthy or rotten for a list of images.
"""

from io import BytesIO
import torch
from torchvision import transforms
from PIL import Image
import typer
from torch.utils.data import DataLoader, Dataset
from typing import List
from omegaconf import DictConfig
import hydra

from healthy_vs_rotten.model import FruitClassifier

app = typer.Typer()


class ImageDataset(Dataset):
    """Dataset for loading images from in-memory bytes (instead of file paths)."""
    def __init__(self, images: List[bytes], transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_bytes = self.images[idx]
        # Wrap the raw bytes in a BytesIO buffer
        with BytesIO(image_bytes) as buffer:
            image = Image.open(buffer).convert("RGB")

        if self.transform:
            image = self.transform(image)
        return image

def load_config(config_path: str, config_name: str = "config") -> DictConfig:
    """
    Load Hydra config from a given path and config name.
    This function can be called once at startup.
    """
    with hydra.initialize_config_dir(version_base=None, config_dir=config_path):
        cfg = hydra.compose(config_name=config_name)
    return cfg

def load_model(cfg: DictConfig, model_path: str) -> FruitClassifier:
    """
    Load the trained model from a file.
    
    :param cfg: Hydra DictConfig object
    :param model_path: Path to your saved .pt or .pth file
    :return: An instance of your PyTorch model
    """
    model: FruitClassifier = hydra.utils.instantiate(cfg.model)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model


def preprocess_images(images: List[bytes]) -> DataLoader:
    """
    Preprocess the input images (in bytes) and create a DataLoader.

    :param images: list of image bytes
    :return: A DataLoader for these images
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    dataset = ImageDataset(images, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    return dataloader


def run_inference(model: torch.nn.Module, dataloader: DataLoader) -> torch.Tensor:
    """
    Run prediction on a batch of images using the loaded model.

    :param model: A PyTorch model (FruitClassifier)
    :param dataloader: DataLoader with images
    :return: A tensor of concatenated model outputs
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in dataloader:
            output = model(batch)
            predictions.append(output)

    predictions_tensor = torch.cat(predictions, dim=0)
    return predictions_tensor
