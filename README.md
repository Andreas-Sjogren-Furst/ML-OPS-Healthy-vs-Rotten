# healthy_vs_rotten
## Project Description
This project focuses on developing a machine learning model capable of classifying fruits as healthy or rotten based on image data. We will be using the following dataset for both training and testing our model:

- Fruit and Vegetable Disease (https://www.kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten/data)



We are going to be using huggingface transformers as our framework and specifically expect to use a resnet-50 model(https://huggingface.co/microsoft/resnet-50).



When you have come up with an idea, write a project description. The description is the delivery for today and should be at least 300 words. Try to answer the following questions in the description:

Overall goal of the project
What framework are you going to use, and you do you intend to include the framework into your project?
What data are you going to run on (initially, may change)
What models do you expect to use


Data set: https://www.kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten/data?fbclid=IwZXh0bgNhZW0CMTEAAR2U3lpnkUaODqRrqhH7s9b1xtbiXHJo30SUkUsRnGED05tyG6nFjjcHNEE_aem_ZGETeN_5JNM60wXPqtClXw

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
