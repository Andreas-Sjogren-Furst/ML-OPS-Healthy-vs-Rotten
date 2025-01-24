# healthy_vs_rotten
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


# Dataset Setup

This project uses the [Fruits and Vegetables Disease Dataset](https://www.kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten) from Kaggle.

## Download Instructions (Windows)

1. Install requirements:
   ```powershell
   pip install kaggle
   ```

2. Set up Kaggle credentials (unsure whether this is necessary):
   - Create a Kaggle account if you don't have one
   - Go to kaggle.com/account
   - Click "Create New API Token" in the API section
   - Move the downloaded `kaggle.json` to your `.kaggle` folder:
     ```powershell
     New-Item -Path "$env:USERPROFILE\.kaggle" -ItemType Directory -Force
     Move-Item -Path "$env:USERPROFILE\Downloads\kaggle.json" -Destination "$env:USERPROFILE\.kaggle\kaggle.json"
     ```

3. Run the download script:
   ```powershell
   python download_dataset.py
   ```

The dataset will be downloaded to `data/raw/`.


## Wandb Setup

Add the following secrets to your repository's .env file:
1. Add `WANDB_API_KEY`
2. Add `WANDB_ENTITY`
3. Add `WANDB_PROJECT`

### How to Get Entity and Project

You can find your entity and project in the URL:
```
https://wandb.ai/<entity>/<project>
```

# Project Automation with Invoke

This project uses [Invoke](https://www.pyinvoke.org/) for task automation.

## Setup Commands

Ensure you have invoke installed:
```bash
pip install invoke
```

- **Create Environment**: Create a Conda environment.
  ```bash
  invoke create-environment
  ```
- **Install Requirements**: Install project dependencies.
  ```bash
  invoke requirements
  ```
- **Install Dev Requirements**: Install development dependencies.
  ```bash
  invoke dev-requirements
  ```

## Project Commands

- **Preprocess Data**: Process raw data to create processed datasets.
  ```bash
  invoke preprocess-data --raw-data-folder=<raw> --processed-data-folder=<processed>
  ```
- **Train Model**: Train the machine learning model.
  ```bash
  invoke train
  ```
- **Run Tests**: Run tests and generate a coverage report.
  ```bash
  invoke test
  ```

- **Run api locally**: Run the api locally
  ```bash
  invoke serve
  ```
Requires you are authenticated with gcloud and have the correct service account to download the best model. However, if not, you can also create your own tmp folder and place the best_model.pt file in there.

## Workflow

1. Create environment: `invoke create-environment`
2. Install dependencies: `invoke requirements`
3. Preprocess data: `invoke preprocess-data`
4. Train model: `invoke train`
5. Test code: `invoke test`
6. Run api locally: `invoke serve` (required gcloud to download model weights, or place them locally in the /tmp folder)


# Google Cloud Storage with DVC

## Setup (already done in our repo)

1. **Install DVC Google Cloud Extension**  
   ```bash
   pip install dvc-gs
   ```

2. **Configure Remote Storage** 
   ```bash
   dvc remote add -d remote_storage gs://<bucket-name>
   dvc remote modify remote_storage version_aware true
   git add .dvc/config
   git commit -m "Configure DVC remote storage"
   ```

## Workflow with DVC. 

1. **Add Data**  
   ```bash
   dvc add <file_or_folder>
   git add <file_or_folder>.dvc
   git commit -m "Track data with DVC"
   ```

2. **Push Data**  
   ```bash
   dvc push --no-run-cache
   ```

3. **Pull Data**  
   ```bash
   dvc pull --no-run-cache
   ```

4. **Verify Storage**  
   ```bash
   gsutil ls gs://<bucket-name>
   ```

This ensures your data is tracked, versioned, and stored in Google Cloud.



# API Documentation:
## Production Endpoint: 
* Swagger: [https://ml-healthy-vs-rotten-api-63364934645.europe-west1.run.app/docs/docs](https://ml-healthy-vs-rotten-api-63364934645.europe-west1.run.app/docs/docs)


* Redoc: [https://ml-healthy-vs-rotten-api-63364934645.europe-west1.run.app/docs/redoc](https://ml-healthy-vs-rotten-api-63364934645.europe-west1.run.app/docs/redoc)

## Local Endpoint:
1. Authencicate with Gcloud. 
2. It will download model weights from google cloud storage. If not authenticated, place the model weights in the /tmp folder named "best_model.pt"

* local deveopment
 ```invoke serve```

* How to run without invoke: 
 ```uvicorn --reload --port 8000 healthy_vs_rotten.api:app```



# Docker Setup
Remeber to open docker desktop. 
Remeber Docker file needs gcloud service account to download model weights, or you can place this manually in the /tmp folder named "best_model.pt"

## How to build Docker
Example of how to build the docker image for the API:
 ``` docker build -f dockerfiles/api.dockerfile . -t api:latest ```

## How to run Docker
Example of how to run the docker image for the API:
 ``` docker run --rm --name experimentapi api:latest ```

Example of exposing the api publicly at localhost: 
   ``` docker run --rm -p 8000:8000 --name experimentapi api:latest ```


## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).