from torch import nn
from transformers import ResNetModel
from omegaconf import DictConfig
import hydra
import torch
from pathlib import Path
from loguru import logger

class FruitClassifier(nn.Module):
    """Binary classifier for healthy/rotten fruit classification."""
    
    def __init__(
        self, 
        pretrained_model_name: str = "microsoft/resnet-50",
        classifier: DictConfig = None
    ):
        super().__init__()
        
        # Load pretrained model
        self.backbone = ResNetModel.from_pretrained(pretrained_model_name)
        
        # Get classifier configuration or use defaults
        self.hidden_size = classifier.hidden_size if classifier else 512
        self.dropout = classifier.dropout if classifier else 0.2
        
        # Get the correct input size for the classifier
        # ResNet-50's last layer outputs 2048 features
        self.classifier = nn.Sequential(
            nn.Linear(2048, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 1)
        )
        
    def forward(self, x):
        # Get features from backbone
        # The last_hidden_state has shape (batch_size, num_channels, height, width)
        features = self.backbone(x).last_hidden_state
        
        # Global average pooling to get a feature vector
        # Average over height and width dimensions
        features = features.mean([-2, -1])  # Result shape: (batch_size, 2048)
        
        # Classify
        logits = self.classifier(features)
        return logits

project_root = Path(__file__).resolve().parents[2]  # Adjust as needed
config_path = str(project_root / "configs")

@hydra.main(version_base=None, config_path=config_path, config_name="config")
def print_model_info(cfg: DictConfig) -> None:
    # Initialize model
    model = hydra.utils.instantiate(cfg.model)
    logger.info("Model initialized.")
    # Print model information
    print("\n" + "="*50)
    print("FruitClassifier Model Information")
    print("="*50)
    
    print(f"\nBackbone: ResNet")
    print(f"Pretrained model: {cfg.model.pretrained_model_name}")
    
    print("\nClassifier Configuration:")
    print(f"  Input Size: 2048")
    print(f"  Hidden Size: {model.hidden_size}")
    print(f"  Dropout Rate: {model.dropout}")
    print(f"  Output Size: 1")
    
    # Calculate parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\nModel Parameters:")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Non-trainable Parameters: {total_params - trainable_params:,}")
    
    # Test forward pass
    print("\nTesting forward pass:")
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print("="*50 + "\n")

if __name__ == "__main__":
    print_model_info()