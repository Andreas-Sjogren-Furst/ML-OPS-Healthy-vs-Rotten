
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from healthy_vs_rotten.api import app  # Adjust the import path if necessary

# Initialize TestClient
client = TestClient(app)

# Mocked configurations and model for testing
mock_config = {
    "data": {"batch_size": 8},
    "model": {"input_size": 224}
}

mock_model = "mocked_model_object"


@pytest.fixture(autouse=True)
def mock_startup():
    """
    Mock the startup event to load dummy config and model.
    """
    with patch("healthy_vs_rotten.predict_model.load_config", return_value=mock_config):
        with patch("healthy_vs_rotten.predict_model.load_model", return_value=mock_model):
            yield


def test_read_root():
    """
    Test the root endpoint.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Healthy vs. Rotten API!"}


def test_predict_images_no_files():
    """
    Test the /predict endpoint with no files provided.
    """
    response = client.post("/predict", files=[])
    assert response.status_code == 422


