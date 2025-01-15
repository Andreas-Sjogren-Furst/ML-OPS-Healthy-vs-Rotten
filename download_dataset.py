import os
import kaggle
from pathlib import Path

# Create data/raw directory if it doesn't exist
data_dir = Path("data/raw")
data_dir.mkdir(parents=True, exist_ok=True)

# Download dataset
dataset_name = "muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten"
kaggle.api.dataset_download_files(
    dataset_name,
    path=data_dir,
    unzip=True
)

print("Dataset downloaded and extracted to data/raw/")