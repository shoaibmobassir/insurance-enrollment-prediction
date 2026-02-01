"""
Pydantic Schemas for API Request/Response Validation
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class EmployeeInput(BaseModel):
    """Schema for single employee prediction input"""
    age: int = Field(..., ge=18, le=100, description="Employee age")
    gender: str = Field(..., description="Gender: Male, Female, or Other")
    marital_status: str = Field(..., description="Marital status: Single, Married, or Divorced")
    salary: float = Field(..., gt=0, description="Annual salary")
    employment_type: str = Field(..., description="Employment type: Full-time, Part-time, or Contract")
    region: str = Field(..., description="Region: Northeast, South, Midwest, or West")
    has_dependents: str = Field(..., description="Has dependents: Yes or No")
    tenure_years: float = Field(..., ge=0, description="Years of tenure")
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 35,
                "gender": "Male",
                "marital_status": "Married",
                "salary": 75000.0,
                "employment_type": "Full-time",
                "region": "West",
                "has_dependents": "Yes",
                "tenure_years": 5.5
            }
        }


class PredictionOutput(BaseModel):
    """Schema for prediction output"""
    enrolled: int = Field(..., description="Predicted enrollment: 0 or 1")
    probability: float = Field(..., ge=0, le=1, description="Probability of enrollment")
    enrollment_status: str = Field(..., description="Human-readable enrollment status")
    confidence: str = Field(..., description="Confidence level: Low, Medium, or High")


class BatchPredictionInput(BaseModel):
    """Schema for batch prediction input"""
    employees: List[EmployeeInput]


class BatchPredictionOutput(BaseModel):
    """Schema for batch prediction output"""
    predictions: List[PredictionOutput]
    total_count: int
    enrolled_count: int
    not_enrolled_count: int


class ModelInfo(BaseModel):
    """Schema for model information"""
    model_name: str
    model_type: str
    features: List[str]
    performance_metrics: dict
    version: str


class HealthCheck(BaseModel):
    """Schema for health check response"""
    status: str
    model_loaded: bool
    version: str
