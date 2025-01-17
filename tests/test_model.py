from src.healthy_vs_rotten.model import FruitClassifier, ModelParams
import pytest
from omegaconf import OmegaConf
import torch

@pytest.mark.parametrize("batch_size", [32])
def test_model(batch_size : int):
    cfg = OmegaConf.load("configs/model_experiments/model1.yaml")
    cfg = OmegaConf.to_container(cfg, resolve=True)["model_params"]
    params = ModelParams(**cfg)
    model = FruitClassifier(params)
    x = torch.randn(batch_size, 3, 28, 28)
    y = model(x)
    assert y.shape == (batch_size, 1)




    

