"""
API Request/Response Schemas
Pydantic models for API validation.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from datetime import datetime


class PredictionRequest(BaseModel):
    """Request schema for prediction endpoint."""

    # Features based on the best model configuration
    Shooter_Outcome: str = Field(..., description="Outcome of the shooter")
    Shooter_Injury: str = Field(..., description="Shooter injury status")
    Situation: str = Field(..., description="Type of situation")
    Victim_School_Affiliation: str = Field(..., description="Victim's school affiliation")
    Victim_Gender: str = Field(..., description="Victim's gender")
    Victim_Race: str = Field(..., description="Victim's race")
    Shooter_Died: str = Field(..., description="Whether shooter died")
    Shooter_Gender: str = Field(..., description="Shooter's gender")
    Shooter_Race: str = Field(..., description="Shooter's race")
    School_Level: str = Field(..., description="School education level")
    Weapon_Type: str = Field(..., description="Type of weapon used")
    Targets: str = Field(..., description="Targeted or random")
    Victim_Injury: str = Field(..., description="Victim injury type")
    City: str = Field(..., description="City location")
    Media_Attention: str = Field(..., description="Media coverage")
    Day: int = Field(..., ge=1, le=31, description="Day of month")
    Year: int = Field(..., ge=1966, le=2030, description="Year of incident")
    Shooter_Killed: str = Field(..., description="Shooter killed flag")
    Reliability: str = Field(..., description="Data reliability")
    Duration_min: float = Field(..., ge=0, description="Duration in minutes")
    LAT: float = Field(..., ge=-90, le=90, description="Latitude")
    LNG: float = Field(..., ge=-180, le=180, description="Longitude")

    class Config:
        schema_extra = {
            "example": {
                "Shooter_Outcome": "Apprehended/Killed",
                "Shooter_Injury": "Fatal",
                "Situation": "Conflict_Related",
                "Victim_School_Affiliation": "Student/Former_Student",
                "Victim_Gender": "Male",
                "Victim_Race": "White",
                "Shooter_Died": "Yes",
                "Shooter_Gender": "Male",
                "Shooter_Race": "White",
                "School_Level": "High_School",
                "Weapon_Type": "Handgun",
                "Targets": "Victims_Targeted",
                "Victim_Injury": "Fatal",
                "City": "Anytown",
                "Media_Attention": "Yes",
                "Day": 15,
                "Year": 2023,
                "Shooter_Killed": "Yes",
                "Reliability": "High",
                "Duration_min": 10.0,
                "LAT": 40.7128,
                "LNG": -74.0060,
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint."""

    prediction: int = Field(..., description="Binary prediction (0 or 1)")
    has_victims: bool = Field(..., description="Whether incident has victims")
    probability_no_victims: Optional[float] = Field(None, description="Probability of no victims")
    probability_has_victims: Optional[float] = Field(None, description="Probability of victims")
    confidence: Optional[float] = Field(None, description="Prediction confidence")
    model_version: str = Field(..., description="Model version used")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

    class Config:
        schema_extra = {
            "example": {
                "prediction": 1,
                "has_victims": True,
                "probability_no_victims": 0.15,
                "probability_has_victims": 0.85,
                "confidence": 0.85,
                "model_version": "1.0.0",
                "timestamp": "2024-01-15T10:30:00",
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions."""

    instances: List[PredictionRequest] = Field(..., description="List of prediction instances")

    class Config:
        schema_extra = {
            "example": {
                "instances": [
                    # Example instance would go here
                ]
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""

    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_count: int = Field(..., description="Total number of predictions")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""

    status: str = Field(..., description="API status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_version: Optional[str] = Field(None, description="Loaded model version")
    uptime_seconds: Optional[float] = Field(None, description="API uptime in seconds")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ModelInfoResponse(BaseModel):
    """Response schema for model info endpoint."""

    model_type: str = Field(..., description="Type of model")
    version: str = Field(..., description="Model version")
    trained_date: str = Field(..., description="Date model was trained")
    n_features: int = Field(..., description="Number of features")
    feature_names: List[str] = Field(..., description="List of feature names")


class ErrorResponse(BaseModel):
    """Response schema for errors."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class TokenResponse(BaseModel):
    """Response schema for authentication token."""

    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in minutes")
