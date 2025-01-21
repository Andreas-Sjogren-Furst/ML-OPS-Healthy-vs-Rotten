import torch
from healthy_vs_rotten.model import FruitClassifier, ModelParams


def test_fruit_classifier():
    """
    Test the FruitClassifier model's forward pass.
    """
    params = ModelParams(pretrained_model_name="microsoft/resnet-50", hidden_dim=512, dropout_rate=0.2)
    model = FruitClassifier(params)
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    assert output.shape == (1, 1)
