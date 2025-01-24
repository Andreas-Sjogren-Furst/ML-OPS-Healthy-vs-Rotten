"""
FastAPI application for Healthy vs. Rotten image classification.
[... rest of the module docstring ...]
"""

import os
from pathlib import Path
from google.cloud import storage
import time
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

import uvicorn
from fastapi import FastAPI, Response, UploadFile, File, HTTPException, status
from pydantic import BaseModel
from typing import List
import torch
from google.cloud import monitoring_v3
from google.protobuf.timestamp_pb2 import Timestamp
from google.auth import default

from healthy_vs_rotten.predict_model import load_config, load_model, preprocess_images, run_inference

project_root = Path(__file__).resolve().parents[2]
CONFIG_DIR = str(project_root / "configs")
#MODEL_PATH = "models/best_model.pt"
CONTAINER_MODEL_PATH = "./tmp/best_model.pt"  # Cloud Run allows writing to /tmp
BUCKET_NAME = "ml-ops-healthy-vs-rotten-data"
MODEL_BLOB_PATH = "models/best_model.pt"

def download_model_from_gcs(bucket_name: str, blob_path : str, local_path: str ="./models/best_model.pt"):
    """
    Download model file from Google Cloud Storage to a local path.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(local_path)
    print(f"[Startup] Model downloaded from GCS: gs://{bucket_name}/{blob_path}")
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

# Prometheus metrics
PREDICTION_REQUESTS = Counter(
    'prediction_requests_total',
    'Number of prediction requests received'
)

PREDICTION_SUCCESSES = Counter(
    'prediction_successes_total',
    'Number of successful predictions'
)

PREDICTION_FAILURES = Counter(
    'prediction_failures_total',
    'Number of failed predictions'
)

PREDICTION_DURATION = Histogram(
    'prediction_duration_seconds',
    'Time spent processing prediction requests',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
)

BATCH_SIZE = Histogram(
    'prediction_batch_size',
    'Size of prediction batches',
    buckets=[1, 2, 5, 10, 20, 50]
)

MODEL_VERSION = Gauge(
    'model_version_info',
    'Information about the currently loaded model version'
)






def write_custom_metric(metric_type: str, value: float, labels: dict = None):
    """
    Write a custom metric to Google Cloud Monitoring using the 'global' resource type.
    Args:
        metric_type (str): Full name of the custom metric (e.g., "custom.googleapis.com/prediction_requests_total").
        value (float): The value to record for the metric.
        labels (dict): Optional labels to add to the metric (e.g., {"env": "local", "model": "v1"}).
    """
    try:
        # Get default credentials and project
        credentials, project_id = default()
        client = monitoring_v3.MetricServiceClient(credentials=credentials)
        project_name = f"projects/{project_id}"

        # Construct the TimeSeries object
        series = monitoring_v3.TimeSeries()
        series.metric.type = metric_type
        if labels:
            series.metric.labels.update(labels)

        # Use 'global' resource type for all metrics
        series.resource.type = "global"
        series.resource.labels["project_id"] = project_id

        # Construct the Point object
        now = time.time()
        timestamp = Timestamp()
        timestamp.seconds = int(now)
        timestamp.nanos = int((now - int(now)) * 1e9)

        point = monitoring_v3.Point()
        point.value.double_value = value
        point.interval.end_time = timestamp  # Set the end time to now

        # Add the point to the TimeSeries
        series.points.append(point)

        # Send the TimeSeries to Google Cloud Monitoring
        client.create_time_series(name=project_name, time_series=[series])
        print(f"[DEBUG] Metric '{metric_type}' written successfully with value: {value}")

    except Exception as e:
        print(f"[ERROR] Failed to write metric '{metric_type}': {e}")






@app.on_event("startup")
def on_startup():
    """
    Initialize configuration and model on startup.
    Loads the model and configuration files for inference.
    """
    global CONFIG, MODEL
    if not Path(CONTAINER_MODEL_PATH).exists():
        if not Path("./tmp").exists():
            Path("./tmp").mkdir()
        download_model_from_gcs(bucket_name=BUCKET_NAME, blob_path=MODEL_BLOB_PATH, local_path=CONTAINER_MODEL_PATH)
    CONFIG = load_config(config_path=CONFIG_DIR, config_name="config")
    MODEL = load_model(CONFIG, CONTAINER_MODEL_PATH)
    MODEL_VERSION.set(1)  # Set to appropriate version number
    print("[Startup] Model loaded successfully!")


@app.get("/")
async def read_root():
    """Return welcome message for the API root endpoint."""
    return {"message": "Welcome to the Healthy vs. Rotten API!"}

@app.get("/metrics")
async def metrics():
    """Endpoint for Prometheus metrics."""
    
    write_custom_metric(
        metric_type="custom.googleapis.com/prediction_requests_total",
        value=PREDICTION_REQUESTS._value.get(),
        labels={"env": os.getenv("ENVIRONMENT", "local")}
    )
    write_custom_metric(
        metric_type="custom.googleapis.com/prediction_successes_total",
        value=PREDICTION_SUCCESSES._value.get(),
        labels={"env": os.getenv("ENVIRONMENT", "local")}
    )
    write_custom_metric(
        metric_type="custom.googleapis.com/prediction_failures_total",
        value=PREDICTION_FAILURES._value.get(),
        labels={"env": os.getenv("ENVIRONMENT", "local")}
    )
    write_custom_metric(
        metric_type="custom.googleapis.com/prediction_duration_seconds",
        value=PREDICTION_DURATION._sum.get(),
        labels={"env": os.getenv("ENVIRONMENT", "local")}
    )
    write_custom_metric(
        metric_type="custom.googleapis.com/prediction_batch_size",
        value=BATCH_SIZE._sum.get(),
        labels={"env": os.getenv("ENVIRONMENT", "local")}
    )
    write_custom_metric(
        metric_type="custom.googleapis.com/model_version_info",
        value=MODEL_VERSION._value.get(),
        labels={"env": os.getenv("ENVIRONMENT", "local")}
    )

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=List[PredictionResponse])
async def predict_images(files: List[UploadFile] = File(...)):
    """
    Accept multiple image files as form-data, run inference, and return predictions.
    [... rest of the docstring ...]
    """
    PREDICTION_REQUESTS.inc()
    BATCH_SIZE.observe(len(files))
    
    start_time = time.time()
    try:
        if not files:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files were provided")

        for file in files:
            if not file.content_type.startswith("image/"):
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=f"File {file.filename} is not an image"
                )

        try:
            images_bytes = [await f.read() for f in files]
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=f"Error reading image files: {str(e)}"
            ) from e

        try:
            dataloader = preprocess_images(images_bytes)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Error preprocessing images: {str(e)}"
            ) from e

        if MODEL is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not initialized. Please try again later."
            )

        try:
            preds = run_inference(MODEL, dataloader)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error during inference: {str(e)}"
            ) from e

        responses = []
        for f, pred in zip(files, preds):
            score = torch.sigmoid(pred).item()
            label = "healthy" if score > 0.5 else "rotten"
            responses.append(PredictionResponse(filename=f.filename or "unknown", score=score, label=label))

        PREDICTION_SUCCESSES.inc()
        return responses

    except HTTPException:
        PREDICTION_FAILURES.inc()
        raise
    except Exception as e:
        PREDICTION_FAILURES.inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Unexpected error: {str(e)}"
        ) from e
    finally:
        PREDICTION_DURATION.observe(time.time() - start_time)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
