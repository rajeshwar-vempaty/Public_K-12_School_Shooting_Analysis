#!/bin/bash

# Start API Server Script

set -e

echo "Starting School Shooting Prediction API..."

# Set environment
export PYTHONPATH=$(pwd)
export API_ENVIRONMENT=${API_ENVIRONMENT:-production}

# Check if model exists
if [ ! -f "data/models/school_shooting_model_v1.0.0.pkl" ]; then
    echo "Warning: Model file not found. API will start without model."
    echo "Train a model first using: python scripts/train_model.py"
fi

# Start with Gunicorn
echo "Starting Gunicorn server..."
gunicorn src.api.app:app \
    --workers ${API_WORKERS:-4} \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind ${API_HOST:-0.0.0.0}:${API_PORT:-8000} \
    --timeout 120 \
    --access-logfile logs/access.log \
    --error-logfile logs/error.log \
    --log-level ${LOG_LEVEL:-info}
