from torch.utils.data import preprocess
from healthy_vs_rotten.data import FruitVegDataset
import pytest
import os.path
from tests import _PATH_TEST_DATA, _PATH_DATA
from pathlib import Path
import shutil



@pytest.mark.skipif(not os.path.exists("data/raw/Fruit And Vegetable Diseases Dataset"))
def test_data():
    preprocess()



def test_my_dataset():
    """Test the MyDataset class."""
    dataset = FruitVegDataset("data/raw")
    assert isinstance(dataset, FruitVegDataset)



def create_test_dataset(testdataset_path: Path = _PATH_TEST_DATA, dataset_path: Path = _PATH_DATA) -> None:
    """
    Create a mock test dataset with small images for unit testing.
    """
    # Ensure the dataset directory exists
    (testdataset_path / 'raw').mkdir(parents=True, exist_ok=True)
    (testdataset_path / 'processed').mkdir(parents=True, exist_ok=True)

    # Create mock images
    healthy_image_path = dataset_path / 'raw' / 'Fruit And Vegetable Diseases Dataset' / 'Bellpepper_Healthy' / 'freshPepper(1).jpg'
    rotten_image_path = dataset_path / 'raw' / 'Fruit And Vegetable Diseases Dataset' / 'Bellpepper_Rotten' / 'rottenPepper(1).jpg'

    shutil.copy(healthy_image_path, testdataset_path / 'raw')
    shutil.copy(rotten_image_path, testdataset_path / 'raw')

    print(f"Test dataset created at {testdataset_path}")

