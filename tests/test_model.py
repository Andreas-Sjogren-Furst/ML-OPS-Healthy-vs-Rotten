from src.healthy_vs_rotten.model import FruitClassifier
import pytest
from omegaconf import OmegaConf
import torch


@pytest.mark.parametrize("batch_size", [32, 64])
def test_model(batch_size: int):
    cfg = OmegaConf.load("configs/model/default.yaml")
    model = FruitClassifier(cfg["pretrained_model_name"], cfg["classifier"])
    x = torch.randn(batch_size, 3, 28, 28)
    y = model(x)
    assert y.shape == (batch_size, 1)
