"""Test the predict_model module."""
from src.healthy_vs_rotten.predict_model import preprocess_images

from io import BytesIO
from PIL import Image
from torch.utils.data import DataLoader


def generate_image_bytes():
    """Generate mock image bytes for testing."""
    img = Image.new('RGB', (100, 100), color=(73, 109, 137))
    byte_io = BytesIO()
    img.save(byte_io, 'JPEG')
    byte_io.seek(0)
    return byte_io.read()

def test_preprocess_images():
    """Test the preprocess_images function."""
    # Generate mock image bytes
    image_bytes = [generate_image_bytes() for _ in range(5)]  # 5 mock images

    # Call the preprocess_images function
    dataloader = preprocess_images(image_bytes)

    # Check if the returned object is a DataLoader
    assert isinstance(dataloader, DataLoader)
