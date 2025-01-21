import torch
from torchvision import transforms
from PIL import Image
import typer
from torch.utils.data import DataLoader, Dataset
from typing import List

from healthy_vs_rotten.model import FruitClassifier, ModelParams

app = typer.Typer()


class ImageDataset(Dataset):
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


def load_model(model_path: str, params: ModelParams) -> FruitClassifier:
    """Load the trained model from a file."""
    model = FruitClassifier(params)
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
    return torch.cat(predictions, dim=0)


@app.command()
def predict_images(image_paths: List[str], model_path: str):
    """Predict whether the fruit/vegetable is healthy or rotten for a list of images."""
    # Define your model parameters
    params = ModelParams(pretrained_model_name="microsoft/resnet-50", hidden_dim=512, dropout_rate=0.2)

    # Load the model
    model = load_model(model_path, params)

    # Preprocess the images and create a dataloader
    dataloader = preprocess_images(image_paths)

    # Make predictions
    predictions = predict(model, dataloader)

    # Print the results
    for image_path, prediction in zip(image_paths, predictions):
        prediction = torch.sigmoid(prediction).item()
        if prediction > 0.5:
            print(f"The produce in the image {image_path} is healthy.")
        else:
            print(f"The produce in the image {image_path} is rotten.")


if __name__ == "__main__":
    app()
