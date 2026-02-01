"""
FastAPI Application for Insurance Enrollment Prediction
Provides REST API endpoints for model predictions
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import pickle
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.schemas import (
    EmployeeInput, PredictionOutput, BatchPredictionInput, 
    BatchPredictionOutput, ModelInfo, HealthCheck
)

# Initialize FastAPI app
app = FastAPI(
    title="Insurance Enrollment Prediction API",
    description="ML-powered API for predicting employee insurance enrollment",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and preprocessors
model = None
scaler = None
label_encoders = None
feature_names = None
model_metrics = None
MODEL_NAME = "XGBoost"


def load_model_and_preprocessors():
    """Load trained model and preprocessing objects"""
    global model, scaler, label_encoders, feature_names, model_metrics
    
    try:
        # Load model
        model_path = Path(__file__).parent.parent / 'models' / 'xgboost.pkl'
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load scaler
        scaler_path = Path(__file__).parent.parent / 'data' / 'scaler.pkl'
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Load label encoders
        encoders_path = Path(__file__).parent.parent / 'data' / 'label_encoders.pkl'
        with open(encoders_path, 'rb') as f:
            label_encoders = pickle.load(f)
        
        # Load feature names
        X_train_path = Path(__file__).parent.parent / 'data' / 'X_train.csv'
        X_train = pd.read_csv(X_train_path)
        feature_names = X_train.columns.tolist()
        
        # Load model metrics
        metrics_path = Path(__file__).parent.parent / 'models' / 'xgboost_metrics.pkl'
        try:
            with open(metrics_path, 'rb') as f:
                model_metrics = pickle.load(f)
        except FileNotFoundError:
            model_metrics = {"note": "Metrics file not found"}
        
        print("✓ Model and preprocessors loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


# Load model on startup
@app.on_event("startup")
async def startup_event():
    """Load model when API starts"""
    load_model_and_preprocessors()


def preprocess_input(employee_data: EmployeeInput) -> pd.DataFrame:
    """
    Preprocess input data to match training format
    
    Args:
        employee_data: Employee input data
        
    Returns:
        Preprocessed DataFrame ready for prediction
    """
    # Convert to dictionary
    data_dict = employee_data.dict()
    
    # Create DataFrame
    df = pd.DataFrame([data_dict])
    
    # Feature engineering (matching training pipeline)
    # Age groups
    df['age_group'] = pd.cut(
        df['age'], 
        bins=[0, 30, 40, 50, 100], 
        labels=['Young', 'Middle', 'Senior', 'Veteran']
    )
    
    # Salary bins
    df['salary_bin'] = pd.cut(
        df['salary'],
        bins=[0, 50000, 70000, 90000, float('inf')],
        labels=['Low', 'Medium', 'High', 'Very High']
    )
    
    # Tenure categories
    df['tenure_category'] = pd.cut(
        df['tenure_years'],
        bins=[-1, 2, 5, 10, 100],
        labels=['New', 'Intermediate', 'Experienced', 'Veteran']
    )
    
    # Binary encoding for has_dependents
    df['has_dependents_binary'] = (df['has_dependents'] == 'Yes').astype(int)
    
    # Interaction feature
    df['salary_per_tenure'] = df['salary'] / (df['tenure_years'] + 1)
    
    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if col in label_encoders:
            # Handle unseen categories
            try:
                df[col] = label_encoders[col].transform(df[col])
            except ValueError:
                # Use most frequent class for unseen categories
                df[col] = label_encoders[col].transform([label_encoders[col].classes_[0]])[0]
    
    # Scale numerical features
    numerical_cols = ['age', 'salary', 'tenure_years', 'salary_per_tenure']
    df[numerical_cols] = scaler.transform(df[numerical_cols])
    
    # Ensure correct column order
    df = df[feature_names]
    
    return df


def get_confidence_level(probability: float) -> str:
    """
    Determine confidence level based on probability
    
    Args:
        probability: Prediction probability
        
    Returns:
        Confidence level: Low, Medium, or High
    """
    if probability < 0.3 or probability > 0.7:
        return "High"
    elif probability < 0.4 or probability > 0.6:
        return "Medium"
    else:
        return "Low"


@app.get("/", response_model=HealthCheck)
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "version": "1.0.0"
    }


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "version": "1.0.0"
    }


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": MODEL_NAME,
        "model_type": type(model).__name__,
        "features": feature_names,
        "performance_metrics": model_metrics if model_metrics else {},
        "version": "1.0.0"
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict_enrollment(employee: EmployeeInput):
    """
    Predict insurance enrollment for a single employee
    
    Args:
        employee: Employee data
        
    Returns:
        Prediction with probability and confidence
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Preprocess input
        X = preprocess_input(employee)
        
        # Make prediction
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0, 1]
        
        # Determine enrollment status
        enrollment_status = "Enrolled" if prediction == 1 else "Not Enrolled"
        confidence = get_confidence_level(probability)
        
        return {
            "enrolled": int(prediction),
            "probability": float(probability),
            "enrollment_status": enrollment_status,
            "confidence": confidence
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionOutput)
async def predict_batch(batch_input: BatchPredictionInput):
    """
    Predict insurance enrollment for multiple employees
    
    Args:
        batch_input: List of employee data
        
    Returns:
        Batch predictions with summary statistics
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        predictions = []
        enrolled_count = 0
        not_enrolled_count = 0
        
        for employee in batch_input.employees:
            # Preprocess input
            X = preprocess_input(employee)
            
            # Make prediction
            prediction = model.predict(X)[0]
            probability = model.predict_proba(X)[0, 1]
            
            # Determine enrollment status
            enrollment_status = "Enrolled" if prediction == 1 else "Not Enrolled"
            confidence = get_confidence_level(probability)
            
            predictions.append({
                "enrolled": int(prediction),
                "probability": float(probability),
                "enrollment_status": enrollment_status,
                "confidence": confidence
            })
            
            if prediction == 1:
                enrolled_count += 1
            else:
                not_enrolled_count += 1
        
        return {
            "predictions": predictions,
            "total_count": len(predictions),
            "enrolled_count": enrolled_count,
            "not_enrolled_count": not_enrolled_count
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
