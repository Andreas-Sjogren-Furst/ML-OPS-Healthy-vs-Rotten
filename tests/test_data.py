"""Tests for the data module."""
import os.path
import shutil
import pytest
from tests import _PATH_TEST_DATA
from src.healthy_vs_rotten.data import FruitVegDataset


@pytest.mark.skipif(
    not os.path.exists(_PATH_TEST_DATA / "raw/Fruit And Vegetable Diseases Dataset"),
    reason="Dataset 'Fruit And Vegetable Diseases Dataset' not found",
)
def test_data():
    """Test the MyDataset class."""
    dataset = FruitVegDataset(_PATH_TEST_DATA / "raw/Fruit And Vegetable Diseases Dataset")
    assert isinstance(dataset, FruitVegDataset)

    processed_path = _PATH_TEST_DATA / "processed"

    os.makedirs(processed_path, exist_ok=True)

    dataset.preprocess(processed_path)

    splits = ["train", "val", "test"]
    classes = ["healthy", "rotten"]

    for split in splits:
        for cls in classes:
            split_cls_dir = processed_path / split / cls
            # Assert that the directory exists
            assert split_cls_dir.exists(), f"{split_cls_dir} does not exist"

    # Verify that each directory contains files (to check if files were copied correctly)

    for cls in classes:
        numb_classes = 0
        for split in splits:
            split_cls_dir = processed_path / split / cls
            files = list(split_cls_dir.glob("*.jpg"))  # Assuming files are jpg
            numb_classes += len(files)
        assert numb_classes > 0, f"No images found in {split_cls_dir}"

    shutil.rmtree(processed_path)

    assert not os.path.exists(processed_path)

    print("Test completed successfully. Directories and files have been validated.")
