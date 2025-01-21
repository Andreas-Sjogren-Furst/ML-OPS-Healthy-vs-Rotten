"""
This script loads a trained model and predicts whether the fruit/vegetable is healthy or rotten for a list of images.
"""

from pathlib import Path
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
    """Dataset for loading images."""

    def __init__(self, image_paths: List[str], transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


def load_model(model_path: str, cfg: DictConfig) -> FruitClassifier:
    """Load the trained model from a file."""
    model = hydra.utils.instantiate(cfg.model)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model


def preprocess_images(image_paths: List[str]) -> DataLoader:
    """Preprocess the input images and create a dataloader."""
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = ImageDataset(image_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    return dataloader


def predict(model: torch.nn.Module, dataloader: DataLoader) -> torch.Tensor:
    """Run prediction for a given model and dataloader.

    Args:
        model: model to use for prediction
        dataloader: dataloader with batches

    Returns:
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model
    """
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            output = model(batch)
            predictions.append(output)
    predictions_tensor = torch.cat(predictions, dim=0)
    print(f"Predictions: {predictions_tensor}")
    return predictions_tensor


@app.command()
def predict_images(
    cfg: DictConfig,
    image_paths: List[str] = None,
    model_path: str = None,
):
    """Predict whether the fruit/vegetable is healthy or rotten for a list of images."""
    # Set default values for `image_paths` and `model_path` if not provided
    if image_paths is None:
        image_paths = ["data/processed/test/healthy/2.jpg", "data/processed/test/rotten/2.jpg"]
    if model_path is None:
        model_path = "models/best_model.pt"

    # Load the model
    model = load_model(model_path, cfg)

    # Preprocess the images and create a dataloader
    dataloader = preprocess_images(image_paths)

    # Make predictions
    predictions = predict(model, dataloader)

    # Print the results
    for image_path, prediction in zip(image_paths, predictions):
        prediction_value = torch.sigmoid(prediction).item()
        print(f"Prediction for {image_path}: {prediction_value}")
        if prediction_value > 0.5:  # strictly greater
            print(f"The produce in the image {image_path} is healthy.")
        else:
            print(f"The produce in the image {image_path} is rotten.")


project_root = Path(__file__).resolve().parents[2]
CONFIG_PATH = str(project_root / "configs")  # Fixed constant naming


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig):
    print("Starting the prediction script...")
    predict_images(cfg)


if __name__ == "__main__":
    main()
