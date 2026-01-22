"""
FastAPI Application
Production-ready REST API for school shooting incident predictions.
"""

import time
from datetime import timedelta
from pathlib import Path
from collections import defaultdict
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest
from prometheus_client.exposition import CONTENT_TYPE_LATEST

from .schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    ErrorResponse,
    TokenResponse,
)
from .auth import create_access_token, verify_token, get_secret_key, verify_password, hash_password
from ..models.predictor import ModelPredictor
from ..utils.logger import get_logger
from ..utils.config import get_config

# Initialize
config = get_config()
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title=config.get("api.title", "School Shooting Incident Prediction API"),
    description=config.get(
        "api.description",
        "Production ML API for predicting school shooting incident outcomes",
    ),
    version=config.get("api.version", "1.0.0"),
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
if config.get("api.cors.enabled", True):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.get("api.cors.allow_origins", ["*"]),
        allow_credentials=True,
        allow_methods=config.get("api.cors.allow_methods", ["*"]),
        allow_headers=config.get("api.cors.allow_headers", ["*"]),
    )

# Prometheus metrics
REQUEST_COUNT = Counter("api_requests_total", "Total API requests", ["method", "endpoint", "status"])
REQUEST_DURATION = Histogram("api_request_duration_seconds", "API request duration")
PREDICTION_COUNT = Counter("predictions_total", "Total predictions made", ["result"])
RATE_LIMIT_EXCEEDED = Counter("rate_limit_exceeded_total", "Total rate limit exceeded events", ["client_ip"])


# Simple in-memory rate limiter
class RateLimiter:
    """Simple in-memory rate limiter using sliding window."""

    def __init__(self, max_calls: int, period_seconds: int):
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self.calls = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        """Check if client is allowed to make a request."""
        now = time.time()
        window_start = now - self.period_seconds

        # Remove old calls outside the window
        self.calls[client_id] = [t for t in self.calls[client_id] if t > window_start]

        if len(self.calls[client_id]) >= self.max_calls:
            return False

        self.calls[client_id].append(now)
        return True

    def get_remaining(self, client_id: str) -> int:
        """Get remaining calls for client."""
        now = time.time()
        window_start = now - self.period_seconds
        self.calls[client_id] = [t for t in self.calls[client_id] if t > window_start]
        return max(0, self.max_calls - len(self.calls[client_id]))


# Initialize rate limiter from config
rate_limiter = RateLimiter(
    max_calls=config.get("api.rate_limit.calls", 100),
    period_seconds=config.get("api.rate_limit.period", 60)
)

# Global state
start_time = time.time()
predictor: ModelPredictor = None

# User database - In production, use a proper database
# Default admin user with hashed password (password: "admin123")
# IMPORTANT: Change this password via ADMIN_PASSWORD environment variable
import os
USERS_DB = {
    os.getenv("ADMIN_USERNAME", "admin"): {
        "username": os.getenv("ADMIN_USERNAME", "admin"),
        "hashed_password": hash_password(os.getenv("ADMIN_PASSWORD", "admin123")),
        "disabled": False,
    }
}


@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    global predictor

    logger.info("Starting API server...")

    # Load model
    try:
        models_dir = config.get("persistence.models_dir", "data/models")
        version = config.get("persistence.version", "1.0.0")
        filename_template = config.get(
            "persistence.model_filename", "school_shooting_model_v{version}.pkl"
        )
        filename = filename_template.format(version=version)
        model_path = Path(models_dir) / filename

        if model_path.exists():
            predictor = ModelPredictor(str(model_path))
            logger.info("Model loaded successfully")
        else:
            logger.warning(f"Model file not found at {model_path}. API will run without model.")
            predictor = ModelPredictor()

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.warning("API starting without loaded model")
        predictor = ModelPredictor()

    logger.info("API server started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down API server...")


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Apply rate limiting to all requests."""
    # Skip rate limiting for health and metrics endpoints
    if request.url.path in ["/health", "/metrics", "/"]:
        return await call_next(request)

    if config.get("api.rate_limit.enabled", True):
        client_ip = request.client.host if request.client else "unknown"

        if not rate_limiter.is_allowed(client_ip):
            RATE_LIMIT_EXCEEDED.labels(client_ip=client_ip).inc()
            logger.warning(f"Rate limit exceeded for client: {client_ip}")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "detail": f"Too many requests. Limit: {config.get('api.rate_limit.calls', 100)} per {config.get('api.rate_limit.period', 60)} seconds.",
                    "retry_after": config.get("api.rate_limit.period", 60),
                },
                headers={
                    "Retry-After": str(config.get("api.rate_limit.period", 60)),
                    "X-RateLimit-Limit": str(config.get("api.rate_limit.calls", 100)),
                    "X-RateLimit-Remaining": "0",
                },
            )

    return await call_next(request)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to responses."""
    start = time.time()

    try:
        response = await call_next(request)
        process_time = time.time() - start

        response.headers["X-Process-Time"] = str(process_time)

        # Add rate limit headers
        if config.get("api.rate_limit.enabled", True):
            client_ip = request.client.host if request.client else "unknown"
            response.headers["X-RateLimit-Limit"] = str(config.get("api.rate_limit.calls", 100))
            response.headers["X-RateLimit-Remaining"] = str(rate_limiter.get_remaining(client_ip))

        # Update Prometheus metrics
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()

        REQUEST_DURATION.observe(process_time)

        return response

    except Exception as e:
        logger.error(f"Request processing error: {str(e)}")
        raise


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "School Shooting Incident Prediction API",
        "version": config.get("api.version", "1.0.0"),
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    uptime = time.time() - start_time

    model_info = predictor.get_model_info() if predictor else {"loaded": False}

    return HealthResponse(
        status="healthy" if model_info.get("loaded") else "degraded",
        model_loaded=model_info.get("loaded", False),
        model_version=model_info.get("version"),
        uptime_seconds=uptime,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info(auth: dict = Depends(verify_token)):
    """
    Get information about the loaded model.

    Requires authentication.
    """
    if predictor is None or predictor.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )

    model_info = predictor.get_model_info()

    return ModelInfoResponse(
        model_type=model_info.get("model_type", "Unknown"),
        version=model_info.get("version", "Unknown"),
        trained_date=model_info.get("trained_date", "Unknown"),
        n_features=model_info.get("n_features", 0),
        feature_names=model_info.get("feature_names", []),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest, auth: dict = Depends(verify_token)):
    """
    Make prediction on a single instance.

    Requires authentication.
    """
    if predictor is None or predictor.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not available for predictions",
        )

    try:
        # Convert request to dict
        features = request.dict()

        # Make prediction
        result = predictor.predict_single(features, return_probability=True)

        # Update metrics
        PREDICTION_COUNT.labels(result=str(result["has_victims"])).inc()

        # Create response
        response = PredictionResponse(
            prediction=result["prediction"],
            has_victims=result["has_victims"],
            probability_no_victims=result.get("probability_no_victims"),
            probability_has_victims=result.get("probability_has_victims"),
            confidence=result.get("confidence"),
            model_version=predictor.metadata.get("version", "1.0.0"),
        )

        logger.info(f"Prediction made: {result['has_victims']} (confidence: {result.get('confidence', 0):.2f})")

        return response

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest, auth: dict = Depends(verify_token)):
    """
    Make predictions on multiple instances.

    Requires authentication.
    """
    if predictor is None or predictor.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not available for predictions",
        )

    try:
        predictions = []

        for instance in request.instances:
            features = instance.dict()
            result = predictor.predict_single(features, return_probability=True)

            prediction = PredictionResponse(
                prediction=result["prediction"],
                has_victims=result["has_victims"],
                probability_no_victims=result.get("probability_no_victims"),
                probability_has_victims=result.get("probability_has_victims"),
                confidence=result.get("confidence"),
                model_version=predictor.metadata.get("version", "1.0.0"),
            )

            predictions.append(prediction)

            # Update metrics
            PREDICTION_COUNT.labels(result=str(result["has_victims"])).inc()

        logger.info(f"Batch prediction made for {len(predictions)} instances")

        return BatchPredictionResponse(
            predictions=predictions,
            total_count=len(predictions),
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}",
        )


@app.post("/auth/token", response_model=TokenResponse, tags=["Authentication"])
async def get_auth_token(username: str, password: str):
    """
    Get authentication token.

    Authenticates user against the user database and returns a JWT token.
    Default credentials: admin / admin123 (change via ADMIN_USERNAME and ADMIN_PASSWORD env vars)
    """
    # Validate credentials against user database
    user = USERS_DB.get(username)

    if not user:
        logger.warning(f"Authentication failed: user '{username}' not found")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if user.get("disabled"):
        logger.warning(f"Authentication failed: user '{username}' is disabled")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is disabled",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not verify_password(password, user["hashed_password"]):
        logger.warning(f"Authentication failed: invalid password for user '{username}'")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create access token
    access_token_expires = timedelta(
        minutes=config.get("api.auth.access_token_expire_minutes", 60)
    )
    access_token = create_access_token(
        data={"sub": username}, expires_delta=access_token_expires
    )

    logger.info(f"User '{username}' authenticated successfully")

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=config.get("api.auth.access_token_expire_minutes", 60),
    )


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus format.
    """
    from starlette.responses import Response

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal Server Error",
            detail=str(exc) if config.get("api.debug", False) else None,
        ).dict(),
    )


if __name__ == "__main__":
    import uvicorn

    host = config.get("api.host", "0.0.0.0")
    port = config.get("api.port", 8000)
    workers = config.get("api.workers", 4)

    logger.info(f"Starting server on {host}:{port} with {workers} workers")

    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        workers=workers,
        reload=config.get("api.reload", False),
    )
