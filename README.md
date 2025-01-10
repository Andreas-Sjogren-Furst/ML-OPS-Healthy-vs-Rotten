# healthy_vs_rotten
## Project Description

### Goal
The goal of the machine-learning model is to be capable of classifying fruits as healthy or rotten based on image data. 
The main goal of the project is to implement the model together with the technologies and theories learned throughout the course.

### Framework
For our third-party package, we are going to be using hugging face transformers. 
The framework contains pre-trained models for classification etc which we could apply in our project. 

### Dataset
The dataset is found on Kaggle and only consists of one dataset so we will have to split it into a train and test set. (https://www.kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten/data)
The dataset consists of 28 different classes of 14 different fruits and vegetables. For each fruit and vegetable, there are two classes: healthy and rotten.
There is around 29300 images of both rotten and healthy of the 14 fruit and vegetables. The dataset takes up around 5 GB of memory.

### Models
We are utilizing the Hugging Face framework to obtain a pre-trained model for our project. 
The model we are going to use is called the resnet-50 model(https://huggingface.co/microsoft/resnet-50). ResNet-50, a widely used convolutional neural network, is well-suited for image classification tasks due to its deep residual learning architecture. This model will be fine-tuned on our dataset to ensure optimal performance for the binary classification task.
By utilising the hugging face framework we can fast and efficiently implement a model thereby allowing more time on the DevOps part of the project. 
Hopefully, by creating a well-implemented project we can easily evaluate and compare other models from hugging face for our problem also. 

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
