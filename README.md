# MLflow Hyperparameter Tuning with Bayesian Search and Optuna

This repository demonstrates an end-to-end implementation of hyperparameter tuning using Bayesian optimization with Optuna and MLflow for experiment tracking.

## Overview

The project provides a framework for optimizing machine learning models through systematic hyperparameter tuning using Bayesian search techniques. It leverages:

- **MLflow** for experiment tracking, model versioning, and artifact management
- **Optuna** for efficient hyperparameter optimization with Bayesian search
- **Scikit-learn** for model training and evaluation

## Key Features

- Hyperparameter tuning using Bayesian optimization
- Experiment tracking and model versioning with MLflow
- Model evaluation and metrics visualization
- Support for multiple optimization objectives
- Integration with popular ML frameworks

## Getting Started

### Prerequisites

This project requires Python 3 and the following packages (included in requirements.txt):
- mlflow
- optuna
- scikit-learn
- pandas
- numpy
- matplotlib

### Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Usage

1. Configure your experiment parameters in the configuration file
2. Run hyperparameter optimization as specified in one of the Jupyter notebook:
   ```
   notebooks/mlflow_experiments_bayesiansearch_hp_tuning.ipynb
   ```
   This notebook implements Bayesian optimization with Optuna to tune RandomForestClassifier hyperparameters including:
   - n_estimators
   - max_features
   - max_depth
   - max_samples
   - bootstrap
   - min_samples_split
   - min_samples_leaf
3. Track experiments using MLflow:
   ```
   mlflow ui --host 0.0.0.0 --port 5000
   ```
   Access the MLflow UI at http://localhost:5000/

## Workflow

1. Data preprocessing and feature engineering
2. Define model architecture and tunable parameters
3. Run Optuna optimization with Bayesian search
4. Track all experiments with MLflow
5. Select the best model based on evaluation metrics
6. Export and save the optimized model

## Additional Information

This project includes integration with other tools from the requirements.txt:
- Pandas and NumPy for data manipulation
- Matplotlib for visualization

## License

MIT License

## Acknowledgments

- [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) for model training and evaluation
- MLflow and Optuna open-source communities