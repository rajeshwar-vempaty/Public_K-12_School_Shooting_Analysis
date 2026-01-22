"""
Tests for FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.app import app
from src.api.auth import create_access_token


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def auth_token():
    """Create authentication token for testing."""
    token = create_access_token(data={"sub": "test_user"})
    return f"Bearer {token}"


def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


def test_predict_without_auth(client):
    """Test prediction endpoint without authentication."""
    payload = {
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
        "City": "Test City",
        "Media_Attention": "Yes",
        "Day": 15,
        "Year": 2023,
        "Shooter_Killed": "Yes",
        "Reliability": "High",
        "Duration_min": 10.0,
        "LAT": 40.7128,
        "LNG": -74.0060,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 403  # Should fail without auth


def test_metrics_endpoint(client):
    """Test Prometheus metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
