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


# Global variables for preprocessors (shared across models)
scaler = None
label_encoders = None
feature_names = None

# Model management
class ModelManager:
    def __init__(self):
        self.models_dir = Path(__file__).parent.parent / 'models'
        self.loaded_models = {}
        self.loaded_metrics = {}
        self.default_model = "xgboost_label_tuned"
    
    def list_models(self):
        """List all available model names based on files in models directory"""
        if not self.models_dir.exists():
            return []
        
        # Look for .pkl files that are not metrics files
        files = self.models_dir.glob("*.pkl")
        model_names = [
            f.stem for f in files 
            if not f.name.endswith("_metrics.pkl")
        ]
        return sorted(model_names)

    def get_model(self, model_name: str = None):
        """
        Get a model by name, loading it if necessary.
        
        Args:
            model_name: Name of the model (without .pkl extension)
            
        Returns:
            Tuple of (model, metrics)
        """
        if model_name is None:
            model_name = self.default_model
            
        if model_name in self.loaded_models:
            return self.loaded_models[model_name], self.loaded_metrics.get(model_name, {})
            
        # Check if model file exists
        model_path = self.models_dir / f"{model_name}.pkl"
        if not model_path.exists():
            # Try matching close names or fuzzy match if we wanted, for now just strict
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found. Available models: {self.list_models()}")
            
        try:
            # Load model
            print(f"Loading model: {model_name}...")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Load metrics (optional)
            metrics_path = self.models_dir / f"{model_name}_metrics.pkl"
            metrics = {}
            if metrics_path.exists():
                try:
                    with open(metrics_path, 'rb') as f:
                        metrics = pickle.load(f)
                except (EOFError, Exception) as e:
                    print(f"Warning: Could not load metrics for {model_name}: {e}")
                    metrics = {"note": "Metrics not available"}
            
            # Cache
            self.loaded_models[model_name] = model
            self.loaded_metrics[model_name] = metrics
            
            return model, metrics
            
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

model_manager = ModelManager()

def load_preprocessors():
    """Load shared preprocessing objects"""
    global scaler, label_encoders, feature_names
    
    try:
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
        
        print("✓ Shared preprocessors loaded successfully!")
        
    except Exception as e:
        print(f"Error loading preprocessors: {e}")
        # Build paths for detail message
        paths_checked = f"Scaler: {scaler_path}, Encoders: {encoders_path}"
        raise RuntimeError(f"Failed to load preprocessors. {e}. Paths: {paths_checked}")


# Load resources on startup
@app.on_event("startup")
async def startup_event():
    """Initialize API resources"""
    load_preprocessors()
    # Pre-load default model
    try:
        model_manager.get_model()
        print(f"✓ Default model '{model_manager.default_model}' loaded.")
    except Exception as e:
        print(f"Warning: Default model could not be loaded on startup: {e}")


def preprocess_input(employee_data: EmployeeInput, model_feature_names: list = None) -> pd.DataFrame:
    """
    Preprocess input data to match training format
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
                if len(label_encoders[col].classes_) > 0:
                    df[col] = label_encoders[col].transform([label_encoders[col].classes_[0]])[0]
    
    # Scale numerical features
    numerical_cols = ['age', 'salary', 'tenure_years', 'salary_per_tenure']
    df[numerical_cols] = scaler.transform(df[numerical_cols])
    
    # Ensure correct column order using global feature names if not provided
    cols_to_use = model_feature_names if model_feature_names else feature_names
    
    # Add missing columns with 0 if needed (robustness)
    for col in cols_to_use:
        if col not in df.columns:
            df[col] = 0
            
    df = df[cols_to_use]
    
    return df


def get_confidence_level(probability: float) -> str:
    """Determine confidence level based on probability"""
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
        "model_loaded": len(model_manager.loaded_models) > 0,
        "version": "1.1.0"
    }


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": len(model_manager.loaded_models) > 0,
        "version": "1.1.0"
    }

@app.get("/models")
async def list_models():
    """List all available models"""
    return {
        "models": model_manager.list_models(),
        "default": model_manager.default_model,
        "loaded": list(model_manager.loaded_models.keys())
    }

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info(model: str = None):
    """
    Get information about a model
    
    Args:
        model: Optional model name (defaults to active/default model)
    """
    model_obj, metrics = model_manager.get_model(model)
    model_name = model if model else model_manager.default_model
    
    return {
        "model_name": model_name,
        "model_type": type(model_obj).__name__,
        "features": feature_names,
        "performance_metrics": metrics if metrics else {},
        "version": "1.0.0"
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict_enrollment(employee: EmployeeInput, model: str = None):
    """
    Predict insurance enrollment for a single employee
    
    Args:
        employee: Employee data
        model: Optional model name to use for prediction
    """
    try:
        # Get model
        model_obj, _ = model_manager.get_model(model)
        
        # Preprocess input
        X = preprocess_input(employee, feature_names)
        
        # Make prediction
        prediction = model_obj.predict(X)[0]
        try:
            probability = model_obj.predict_proba(X)[0, 1]
        except AttributeError:
            # Some models might not support predict_proba, or handle differently
            probability = float(prediction) # Fallback
        
        # Determine enrollment status
        enrollment_status = "Enrolled" if prediction == 1 else "Not Enrolled"
        confidence = get_confidence_level(probability)
        
        return {
            "enrolled": int(prediction),
            "probability": float(probability),
            "enrollment_status": enrollment_status,
            "confidence": confidence
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionOutput)
async def predict_batch(batch_input: BatchPredictionInput, model: str = None):
    """
    Predict insurance enrollment for multiple employees
    
    Args:
        batch_input: List of employee data
        model: Optional model name to use for prediction
    """
    try:
        # Get model
        model_obj, _ = model_manager.get_model(model)
        
        predictions = []
        enrolled_count = 0
        not_enrolled_count = 0
        
        for employee in batch_input.employees:
            # Preprocess input
            # Note: For efficiency in a real production system, we'd vectorize this 
            # instead of looping, but reusing the single preprocess function is safer for consistency here
            X = preprocess_input(employee, feature_names)
            
            # Make prediction
            prediction = model_obj.predict(X)[0]
            try:
                probability = model_obj.predict_proba(X)[0, 1]
            except AttributeError:
                probability = float(prediction)
            
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
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
