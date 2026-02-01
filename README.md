# Insurance Enrollment Prediction 

A comprehensive machine learning pipeline for predicting employee insurance enrollment based on demographic and employment data.

##  Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Results](#results)
- [Technologies Used](#technologies-used)

##  Overview

This project implements an end-to-end machine learning solution to predict whether employees will enroll in a voluntary insurance product. The pipeline includes data processing, multiple ML models, and hyperparameter tuning optimization.

** Best Model**: Gradient Boosting with **100% Accuracy**.

##  Features

- ✅ **Data Processing**: Automated EDA, feature engineering (age groups, salary bins, interaction features).
- ✅ **Multiple Algorithms**: XOR-style comparison of 5 models including XGBoost and LightGBM.
- ✅ **GridSearchCV**: Exhaustive hyperparameter tuning for optimization.
- ✅ **REST API**: FastAPI server for real-time inference.
- ✅ **Explainability**: Feature importance analysis and SHAP-ready structure.

##  Project Structure

```
insurance-enrollment-prediction/
├── data/                           # Data directory
│   └── employee_data.csv          # Dataset
├── src/                           # Source code (Processing, Training, Eval)
├── models/                        # Saved models (PKL files)
├── notebooks/                     # Generated Visualizations
├── api/                           # FastAPI application
├── main.py                       # CLI Entry point
└── report.md                     # Detailed Analysis Report
```

##  Installation

1. **Clone & Setup**:
   ```bash
   cd insurance-enrollment-prediction
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

##  Quick Start

Run the complete pipeline (Process -> Train -> Tune -> Evaluate):

```bash
python main.py
```

##  Usage

### CLI Modes
- **Train & Tune**: `python main.py --mode train --tune`
- **Train with OHE**: `python main.py --mode train --encoding ohe`
- **Predict**: `python main.py --mode predict --input data/test.csv`

### REST API
Start the server and access Swagger UI at `http://localhost:8000/docs`:

```bash
uvicorn api.main:app --reload
```

##  Results

We achieved perfect classification on the test set.

![ROC Analysis](notebooks/roc_comparison_all_models.png)

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|-------|----------|-----------|--------|----------|---------|
| **Gradient Boosting** | **100.00%** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| XGBoost | 99.95% | 0.9992 | 1.0000 | 0.9996 | 1.0000 |
| LightGBM | 99.95% | 0.9992 | 1.0000 | 0.9996 | 1.0000 |
| Random Forest | 99.85% | 0.9992 | 0.9984 | 0.9988 | 1.0000 |
| Logistic Regression | 88.00% | 0.8866 | 0.9239 | 0.9048 | 0.9310 |

See [report.md](report.md) for detailed analysis and visualizations.

##  Generalized Inference Script

Use `src/inference.py` to train new models with custom parameters or predict using saved models.

### 1. Train a New Model
```bash
python src/inference.py --mode train \
    --model random_forest \
    --encoding ohe \
    --params '{"n_estimators": 200, "max_depth": 10}' \
    --input data/employee_data.csv
```

### 2. Predict with Saved Model
```bash
python src/inference.py --mode load \
    --model_path models/random_forest_ohe_tuned.pkl \
    --input data/test_sample.csv \
    --output results.csv
```

##  Jupyter Notebook

Explore the full interactive pipeline in `notebooks/full_pipeline.ipynb`. This notebook provides a deep dive into:
-  **EDA**: Visualizations of distributions and correlations.
-  **Feature Engineering**: Creation of derived features.
-  **Model Tuning**: Step-by-step GridSearchCV (configured for stability).
-  **Evaluation**: ROC and Precision-Recall curves.

```bash
jupyter notebook notebooks/full_pipeline.ipynb
```

> **Note for macOS Users**: The notebook and training scripts are configured with `n_jobs=1` (sequential execution) to ensure stability and avoid multiprocessing errors common with Python 3.14/Pandas on macOS.

##  Experiment Tracking (MLflow)

This project uses **MLflow** to track metrics, hyperparameters, and models.

1. **Run experiment**:
   ```bash
   python main.py --tune
   ```
   (MLflow logging is automatic for Random Forest and XGBoost)

2. **View Dashboard**:
   ```bash
   mlflow ui
   ```
   Open `http://127.0.0.1:5000` to visualize run comparisons.

## � Tech Stack

- **Core**: Python 3.8+, Pandas, NumPy
- **ML**: Scikit-Learn, XGBoost, LightGBM
- **Tracking**: MLflow
- **API**: FastAPI, Uvicorn, Pydantic
- **Viz**: Matplotlib, Seaborn
