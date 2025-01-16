from dataclasses import dataclass
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn
from transformers import ResNetModel

@dataclass
class ModelParams:
    pretrained_model_name: str = "microsoft/resnet-50"
    hidden_dim: int = 512
    dropout_rate: float = 0.2

@dataclass
class Config:
    model_params: ModelParams

class FruitClassifier(nn.Module):
    """Binary classifier for healthy/rotten fruit classification."""

    def __init__(self, params: ModelParams) -> None:
        super().__init__()
        
        # Load pretrained model
        self.backbone = ResNetModel.from_pretrained(params.pretrained_model_name)
        
        # Get the correct input size for the classifier
        # ResNet-50's last layer outputs 2048 features
        self.classifier = nn.Sequential(
            nn.Linear(2048, params.hidden_dim),
            nn.ReLU(),
            nn.Dropout(params.dropout_rate),
            nn.Linear(params.hidden_dim, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get features from backbone
        # The last_hidden_state has shape (batch_size, num_channels, height, width)
        features = self.backbone(x).last_hidden_state
        
        # Global average pooling to get a feature vector
        features = features.mean(dim=[2, 3])
        
        # Pass through classifier
        return self.classifier(features)

project_root = Path(__file__).resolve().parents[2]  # Adjust as needed
config_path = str(project_root / "configs/model_experiments")

@hydra.main(config_path=config_path, config_name="model1.yaml")
def main(cfg):
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Instantiate the model using Hydra
    model = instantiate(cfg.model)
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")


if __name__ == "__main__":
    main()