"""Download kaggle dataset"""

import kaggle
from pathlib import Path
from loguru import logger

# Create data/raw directory if it doesn't exist
data_dir = Path("data/raw")
data_dir.mkdir(parents=True, exist_ok=True)
logger.info("Succesfully created directory")

# Download dataset
DATASET_NAME = "muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten"
logger.info(f"Downloading dataset {DATASET_NAME}...")
kaggle.api.dataset_download_files(DATASET_NAME, path=data_dir, unzip=True)
logger.info("Succesfully downloaded dataset")
logger.info(f"Saved to {data_dir}")
