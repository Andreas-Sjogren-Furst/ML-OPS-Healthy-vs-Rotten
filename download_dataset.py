import kaggle
from pathlib import Path
from loguru import logger

# Create data/raw directory if it doesn't exist
data_dir = Path("data/raw")
data_dir.mkdir(parents=True, exist_ok=True)
logger.info("Succesfully created directory")

# Download dataset
dataset_name = "muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten"
logger.info(f"Downloading dataset {dataset_name}...")
kaggle.api.dataset_download_files(dataset_name, path=data_dir, unzip=True)
logger.info("Succesfully downloaded dataset")
logger.info(f"Saved to {data_dir}")
