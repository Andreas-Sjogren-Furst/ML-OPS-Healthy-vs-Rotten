"""Script to quantize the fruit classifier model to int8."""

import torch
import copy
from pathlib import Path
import hydra
from omegaconf import DictConfig
from torch.ao.quantization import get_default_qat_qconfig, prepare_qat

from healthy_vs_rotten.model import FruitClassifier
from healthy_vs_rotten.data import FruitVegDataset

project_root = Path(__file__).resolve().parents[2]
CONFIG_PATH = str(project_root / "configs")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def quantize(cfg: DictConfig) -> None:
    """Quantize the model to int8 using QAT."""
    # Load model
    model = hydra.utils.instantiate(cfg.model)
    model_path = Path(cfg.paths.output_dir) / "best_model.pt"
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.train()  # Set to train mode for QAT
    
    # Set qconfig for QAT
    model.qconfig = get_default_qat_qconfig('fbgemm')
    
    # Prepare model for QAT
    model_prepared = prepare_qat(model)
    
    # Load calibration data
    data_dir = Path(cfg.paths.data_dir)
    cal_dataset = FruitVegDataset(data_dir / "val")
    cal_loader = torch.utils.data.DataLoader(cal_dataset, batch_size=32)

    # Perform quantization aware training
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model_prepared.parameters(), lr=0.0001)
    
    print("Starting quantization aware training...")
    for epoch in range(3):  # Few epochs for fine-tuning
        for batch_idx, (data, target) in enumerate(cal_loader):
            optimizer.zero_grad()
            output = model_prepared(data).squeeze()
            loss = criterion(output, target.float())
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

    # Convert to quantized model
    model_prepared.eval()
    model_int8 = torch.ao.quantization.convert(model_prepared)

    # Save quantized model
    quantized_path = Path(cfg.paths.output_dir) / "quantized_model.pt"
    torch.save({
        'state_dict': model_int8.state_dict(),
        'config': cfg.model,
    }, quantized_path)
    
    print(f"Saved quantized model to {quantized_path}")

    # Print model sizes
    fp32_size = Path(model_path).stat().st_size / (1024 * 1024)
    int8_size = Path(quantized_path).stat().st_size / (1024 * 1024)
    print(f"\nModel size comparison:")
    print(f"FP32 model: {fp32_size:.2f} MB")
    print(f"INT8 model: {int8_size:.2f} MB")
    print(f"Size reduction: {(1 - int8_size/fp32_size) * 100:.1f}%")


if __name__ == "__main__":
    quantize()