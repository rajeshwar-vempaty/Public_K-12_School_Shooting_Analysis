# API Documentation

## Base URL
```
http://localhost:8000
```

## Authentication

The API uses JWT (JSON Web Token) authentication. Include the token in the `Authorization` header:

```
Authorization: Bearer <your_token>
```

### Get Authentication Token

**Endpoint:** `POST /auth/token`

**Parameters:**
- `username` (string): Your username
- `password` (string): Your password

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 60
}
```

## Endpoints

### 1. Root Endpoint

**GET** `/`

Returns API information.

**Response:**
```json
{
  "message": "School Shooting Incident Prediction API",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health"
}
```

### 2. Health Check

**GET** `/health`

Check API and model status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1.0.0",
  "uptime_seconds": 3600.5,
  "timestamp": "2024-01-15T10:30:00"
}
```

### 3. Model Information

**GET** `/model/info`

Get details about the loaded model. **Requires authentication.**

**Headers:**
```
Authorization: Bearer <token>
```

**Response:**
```json
{
  "model_type": "SVC",
  "version": "1.0.0",
  "trained_date": "2024-01-15T08:00:00",
  "n_features": 22,
  "feature_names": ["Shooter_Outcome", "Shooter_Injury", ...]
}
```

### 4. Single Prediction

**POST** `/predict`

Make prediction on a single incident. **Requires authentication.**

**Headers:**
```
Authorization: Bearer <token>
Content-Type: application/json
```

**Request Body:**
```json
{
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
  "LNG": -74.0060
}
```

**Response:**
```json
{
  "prediction": 1,
  "has_victims": true,
  "probability_no_victims": 0.15,
  "probability_has_victims": 0.85,
  "confidence": 0.85,
  "model_version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00"
}
```

### 5. Batch Predictions

**POST** `/predict/batch`

Make predictions on multiple incidents. **Requires authentication.**

**Headers:**
```
Authorization: Bearer <token>
Content-Type: application/json
```

**Request Body:**
```json
{
  "instances": [
    {
      "Shooter_Outcome": "...",
      "Shooter_Injury": "...",
      ...
    },
    {
      "Shooter_Outcome": "...",
      "Shooter_Injury": "...",
      ...
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "prediction": 1,
      "has_victims": true,
      "probability_no_victims": 0.15,
      "probability_has_victims": 0.85,
      "confidence": 0.85,
      "model_version": "1.0.0",
      "timestamp": "2024-01-15T10:30:00"
    },
    ...
  ],
  "total_count": 2,
  "timestamp": "2024-01-15T10:30:00"
}
```

### 6. Metrics

**GET** `/metrics`

Prometheus metrics endpoint.

**Response:** (Prometheus format)
```
# HELP api_requests_total Total API requests
# TYPE api_requests_total counter
api_requests_total{method="POST",endpoint="/predict",status="200"} 42.0
...
```

## Feature Descriptions

| Feature | Type | Description | Example Values |
|---------|------|-------------|----------------|
| Shooter_Outcome | String | How the incident ended for shooter | Apprehended/Killed, Fled, Suicide/Attempted Suicide |
| Shooter_Injury | String | Shooter's injury status | Fatal, No_Injury, Unknown |
| Situation | String | Type of incident | Conflict_Related, Criminal_Activity, Personal_Crisis |
| Victim_School_Affiliation | String | Victim's relation to school | Student/Former_Student, Staff/Former_Staff |
| Victim_Gender | String | Victim's gender | Male, Female, Unknown |
| Victim_Race | String | Victim's race/ethnicity | White, Black, Hispanic, Other |
| Shooter_Died | String | Whether shooter died | Yes, No, Unknown |
| Shooter_Gender | String | Shooter's gender | Male, Female, Other |
| Shooter_Race | String | Shooter's race/ethnicity | White, Black, Hispanic, Other |
| School_Level | String | Education level | High_School, Primary_Education, Secondary_Education |
| Weapon_Type | String | Type of weapon | Handgun, Rifle, Shotgun, Multiple_Weapons |
| Targets | String | Targeting approach | Victims_Targeted, Random_Shooting |
| Victim_Injury | String | Victim injury type | Fatal, Non-Fatal, No_Injury |
| City | String | City where incident occurred | Any city name |
| Media_Attention | String | Media coverage | Yes, No, Unknown |
| Day | Integer | Day of month (1-31) | 15 |
| Year | Integer | Year (1966-2030) | 2023 |
| Shooter_Killed | String | Shooter killed flag | Yes, No, Unknown |
| Reliability | String | Data reliability | High, Medium, Low |
| Duration_min | Float | Duration in minutes (≥0) | 10.0 |
| LAT | Float | Latitude (-90 to 90) | 40.7128 |
| LNG | Float | Longitude (-180 to 180) | -74.0060 |

## Error Responses

### 400 Bad Request
```json
{
  "error": "Validation Error",
  "detail": "Field 'Day' must be between 1 and 31",
  "timestamp": "2024-01-15T10:30:00"
}
```

### 401 Unauthorized
```json
{
  "detail": "Could not validate credentials"
}
```

### 503 Service Unavailable
```json
{
  "detail": "Model not loaded"
}
```

## Code Examples

### Python

```python
import requests

# Get token
auth_response = requests.post(
    "http://localhost:8000/auth/token",
    params={"username": "user", "password": "pass"}
)
token = auth_response.json()["access_token"]

# Make prediction
headers = {"Authorization": f"Bearer {token}"}
data = {
    "Shooter_Outcome": "Apprehended/Killed",
    "Shooter_Injury": "Fatal",
    # ... other features
}

response = requests.post(
    "http://localhost:8000/predict",
    json=data,
    headers=headers
)

prediction = response.json()
print(f"Has victims: {prediction['has_victims']}")
print(f"Confidence: {prediction['confidence']:.2%}")
```

### cURL

```bash
# Get token
TOKEN=$(curl -X POST "http://localhost:8000/auth/token?username=user&password=pass" | jq -r '.access_token')

# Make prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "Shooter_Outcome": "Apprehended/Killed",
    "Shooter_Injury": "Fatal",
    ...
  }'
```

## Interactive Documentation

Visit `/docs` for interactive Swagger UI documentation where you can test endpoints directly in your browser.

Visit `/redoc` for ReDoc-style documentation.
