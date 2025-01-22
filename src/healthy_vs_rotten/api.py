"""
FastAPI application for Healthy vs. Rotten image classification.
[... rest of the module docstring ...]
"""
from pathlib import Path
import uvicorn
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    status
)
from pydantic import BaseModel
from typing import List
import torch

from healthy_vs_rotten.predict_model import (
    load_config,
    load_model,
    preprocess_images,
    run_inference
)

project_root = Path(__file__).resolve().parents[2]
CONFIG_DIR = str(project_root / "configs")
MODEL_PATH = "models/best_model.pt"

class PredictionResponse(BaseModel):
    """
    Response model for image predictions.
    
    Attributes:
        filename: Name of the processed image file
        score: Confidence score of the prediction
        label: Classification label (healthy/rotten)
    """
    filename: str
    score: float
    label: str

app = FastAPI()
CONFIG = None
MODEL = None

@app.on_event("startup")
def on_startup():
    """
    Initialize configuration and model on startup.
    Loads the model and configuration files for inference.
    """
    global CONFIG, MODEL
    CONFIG = load_config(config_path=CONFIG_DIR, config_name="config")
    MODEL = load_model(CONFIG, MODEL_PATH)
    print("[Startup] Model loaded successfully!")

@app.get("/")
async def read_root():
    """Return welcome message for the API root endpoint."""
    return {"message": "Welcome to the Healthy vs. Rotten API!"}

@app.post("/predict", response_model=List[PredictionResponse])
async def predict_images(files: List[UploadFile] = File(...)):
    """
    Accept multiple image files as form-data, run inference, and return predictions.
    [... rest of the docstring ...]
    """
    try:
        if not files:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No files were provided"
            )

        for file in files:
            if not file.content_type.startswith('image/'):
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail=f"File {file.filename} is not an image"
                )

        try:
            images_bytes = [await f.read() for f in files]
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error reading image files: {str(e)}"
            ) from e

        try:
            dataloader = preprocess_images(images_bytes)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Error preprocessing images: {str(e)}"
            ) from e

        if MODEL is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not initialized. Please try again later."
            )

        try:
            preds = run_inference(MODEL, dataloader)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error during inference: {str(e)}"
            ) from e

        responses = []
        for f, pred in zip(files, preds):
            score = torch.sigmoid(pred).item()
            label = "healthy" if score > 0.5 else "rotten"
            responses.append(
                PredictionResponse(filename=f.filename or "unknown", score=score, label=label)
            )

        return responses

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        ) from e

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
