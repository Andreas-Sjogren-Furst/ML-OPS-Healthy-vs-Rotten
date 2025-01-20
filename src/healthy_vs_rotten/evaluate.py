from pathlib import Path
import torch
from torch.utils.data import DataLoader
import typer
from sklearn.metrics import classification_report
from loguru import logger
from data import FruitVegDataset
from model import FruitClassifier


def evaluate(
    model_path: Path = Path("models/best_model.pt"),
    data_dir: Path = Path("data/processed"),
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """Evaluate the model on the test set."""
    # Load test dataset
    test_dataset = FruitVegDataset(data_dir / "test")
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Load model
    model = FruitClassifier().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Evaluation
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images).squeeze()
            predictions = (outputs > 0).float()

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())

    logger.info("Evaluation complete.")
    # Print metrics
    print("\nTest Set Evaluation:")
    print(classification_report(all_labels, all_preds, target_names=["Rotten", "Healthy"]))


if __name__ == "__main__":
    typer.run(evaluate)
