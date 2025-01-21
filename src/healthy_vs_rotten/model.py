"""
This module contains the definition of the FruitClassifier model and its associated parameters.
"""

from dataclasses import dataclass
from pathlib import Path
import torch
from torch import nn
from transformers import ResNetModel
from omegaconf import DictConfig
import hydra
from loguru import logger


@dataclass
class ModelParams:
    """Parameters for the FruitClassifier model."""

    pretrained_model_name: str = "microsoft/resnet-50"
    hidden_dim: int = 512
    dropout_rate: float = 0.2


class FruitClassifier(nn.Module):
    """Binary classifier for healthy/rotten fruit classification."""

    def __init__(self, params: ModelParams):
        super().__init__()

        # Load pretrained model
        self.backbone = ResNetModel.from_pretrained(params.pretrained_model_name)

        # Get the correct input size for the classifier
        # ResNet-50's last layer outputs 2048 features
        self.classifier = nn.Sequential(
            nn.Linear(2048, params.hidden_dim),
            nn.ReLU(),
            nn.Dropout(params.dropout_rate),
            nn.Linear(params.hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Tensor of shape (batch_size, 1) containing logits
        """
        # Get features from backbone
        features = self.backbone(x).last_hidden_state

        # Global average pooling to get a feature vector
        features = features.mean(dim=[2, 3])

        # Pass through classifier
        logits = self.classifier(features)
        return logits


project_root = Path(__file__).resolve().parents[2]
CONFIG_PATH = str(project_root / "configs")  # Fixed constant naming


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def print_model_info(cfg: DictConfig) -> None:
    """Print information about the FruitClassifier

    Args:
        cfg (DictConfig): Hydra configuration object
    """
    # Initialize model
    params = ModelParams(**cfg.model)
    model = FruitClassifier(params)
    logger.info("Model initialized.")
    # Print model information
    print("\n" + "=" * 50)
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")


if __name__ == "__main__":
    print_model_info()
