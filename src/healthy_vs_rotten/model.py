from torch import nn
from transformers import ResNetModel, ResNetConfig

class FruitClassifier(nn.Module):
    """Binary classifier for healthy/rotten fruit classification."""
    
    def __init__(self, pretrained_model_name: str = "microsoft/resnet-50"):
        super().__init__()
        
        # Load pretrained model
        self.backbone = ResNetModel.from_pretrained(pretrained_model_name)
        
        # Get the correct input size for the classifier
        # ResNet-50's last layer outputs 2048 features
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1)
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