# Fruit and Vegetable Disease Detection
## Project Description

### Goal
The objective of the machine-learning model is to be capable of classifying fruits as healthy or rotten based on image data. 
The main goal of the project is to implement the model together with the technologies, frameworks and theories learned throughout the course.

### Framework
We are going to be using hugging face transformers as our third-party package. 
The framework contains pre-trained models for classification, etc, which we are going to apply in the project. 

### Dataset
The dataset is found on Kaggle and only consists of one dataset so we will have to split it into a train and test set. (https://www.kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten/data)
The dataset consists of 28 different classes of 14 different fruits and vegetables. For each fruit and vegetable, there are two classes: healthy and rotten.
There is around 29300 images of both rotten and healthy of the 14 fruit and vegetables. The dataset takes up around 5 GB of memory.

### Models
We are utilizing the Hugging Face framework to obtain a pre-trained model for our project. 
The model we are going to use is called the resnet-50 model(https://huggingface.co/microsoft/resnet-50). ResNet-50, a widely used convolutional neural network, is well-suited for image classification tasks due to its deep residual learning architecture. This model will be fine-tuned on our dataset to ensure optimal performance for the binary classification task.
By utilising the hugging face framework we can fast and efficiently implement a model thereby allowing more time on the DevOps part of the project. 
Hopefully, by creating a well-implemented project we can easily evaluate and compare other models from hugging face for our problem also.


# Setup Guide

A machine learning project for detecting diseases in fruits and vegetables using the [Kaggle Fruits and Vegetables Disease Dataset](https://www.kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten).

## Repository Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Andreas-Sjogren-Furst/ML-OPS-Healthy-vs-Rotten.git
   cd ML-OPS-Healthy-vs-Rotten
   ```

2. Create virtual environment:
   ```bash
   # Using venv
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   .\venv\Scripts\activate   # Windows
   
   # OR using conda
   conda create -n fruit-disease python=3.11.*
   conda activate fruit-disease
   ```

## Initial Setup

1. Install base requirements:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements_dev.txt # For developers
   pip install -e .
   ```

2. Set up Kaggle credentials:
   - Create a Kaggle account
   - Download API token from kaggle.com/settings
   - Configure credentials:
     ```powershell
     New-Item -Path "$env:USERPROFILE\.kaggle" -ItemType Directory -Force
     Move-Item -Path "$env:USERPROFILE\Downloads\kaggle.json" -Destination "$env:USERPROFILE\.kaggle\kaggle.json"
     ```

3. Configure Weights & Biases:
   Add to `.env`:
   - `WANDB_API_KEY`
   - `WANDB_ENTITY` (from URL: wandb.ai/<entity>/<project>)
   - `WANDB_PROJECT`

## Development Workflow

1. Download and preprocess data:
   ```bash
   python download_dataset.py
   invoke preprocess-data
   ```

2. Train model:
   ```bash
   invoke train
   ```

3. Run tests:
   ```bash
   invoke test
   pre-commit run --all-files  # Before committing
   ```

## API Usage

### Production Endpoints
- Swagger: [https://ml-healthy-vs-rotten-api-63364934645.europe-west1.run.app/docs/docs](https://ml-healthy-vs-rotten-api-63364934645.europe-west1.run.app/docs)
- Redoc: [https://ml-healthy-vs-rotten-api-63364934645.europe-west1.run.app/docs/redoc](https://ml-healthy-vs-rotten-api-63364934645.europe-west1.run.app/redoc)

### Local Development

Note: Requires gcloud authentication or manually placed `best_model.pt` in `/tmp` folder.

```bash
invoke serve  # With Invoke
# OR
uvicorn --reload --port 8000 healthy_vs_rotten.api:app  # Without Invoke
```

### Run streamlit interface

If you want to test a deployed model run:

```bash
streamlit run src/healthy_vs_rotten/frontend.py
```

This will start a localhost to a streamlit frontend application. You will be able to drag images which then will be predicted either healthy/rotten.

## Docker

Prerequisites:
- Docker Desktop running
- gcloud authentication or `best_model.pt` in `/tmp`

```bash
# Build
docker build -f dockerfiles/api.dockerfile . -t api:latest

# Run
docker run --rm --name experimentapi api:latest  # Basic
docker run --rm -p 8000:8000 --name experimentapi api:latest  # Expose locally
```

## Data Version Control (DVC)

### Initial Setup
```bash
pip install dvc-gs
dvc remote add -d remote_storage gs://<bucket-name>
dvc remote modify remote_storage version_aware true
git add .dvc/config
git commit -m "Configure DVC remote storage"
```

### Data Management
```bash
dvc add <file_or_folder>  # Track data
git add <file_or_folder>.dvc
git commit -m "Track data with DVC"

dvc push --no-run-cache  # Upload to storage
dvc pull --no-run-cache  # Download from storage
gsutil ls gs://<bucket-name>  # Verify storage
```


## Project structure

The directory structure of the project looks like this:
```txt
├───configs
│   ├───model
│   └───training
├───data
│   ├───processed
│   │   ├───test
│   │   │   ├───healthy
│   │   │   └───rotten
│   │   ├───train
│   │   │   ├───healthy
│   │   │   └───rotten
│   │   └───val
│   │       ├───healthy
│   │       └───rotten
│   └───raw
│       └───Fruit And Vegetable Diseases Dataset
│           └───[Fruit]__[Healthy/Rotten]
├───docs
├───models
├───notebooks
├───reports
├───src
│   └───healthy_vs_rotten
├───tests
└───visualizations
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
