# Deployment Guide

This guide covers deploying the School Shooting Incident Prediction API to production environments.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Environment Variables](#environment-variables)
- [Monitoring & Logging](#monitoring--logging)

## Prerequisites

- Python 3.10+
- Docker & Docker Compose (for containerized deployment)
- Model file trained and saved in `data/models/`
- Environment variables configured

## Local Development

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your configuration
```

### 3. Train Model (if not already trained)

```bash
python scripts/train_model.py
```

### 4. Start API Server

```bash
# Development mode
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

# Production mode
./scripts/start_api.sh
```

## Docker Deployment

### Build and Run with Docker

```bash
# Build image
docker build -t school-shooting-ml-api:latest .

# Run container
docker run -d \
  -p 8000:8000 \
  -e API_SECRET_KEY="your-secret-key" \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/data/models:/app/data/models:ro \
  --name school-shooting-api \
  school-shooting-ml-api:latest
```

### Using Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

This includes:
- API server on port 8000
- Prometheus for metrics on port 9090
- Grafana for visualization on port 3000

## Cloud Deployment

### AWS ECS/Fargate

1. **Push to ECR**:
```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Tag and push
docker tag school-shooting-ml-api:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/school-shooting-ml-api:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/school-shooting-ml-api:latest
```

2. **Create ECS Task Definition** with environment variables
3. **Create ECS Service** with load balancer
4. **Configure auto-scaling** based on CPU/memory

### Google Cloud Run

```bash
# Build and submit
gcloud builds submit --tag gcr.io/PROJECT_ID/school-shooting-ml-api

# Deploy
gcloud run deploy school-shooting-api \
  --image gcr.io/PROJECT_ID/school-shooting-ml-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars API_SECRET_KEY=your-secret-key
```

### Azure Container Instances

```bash
# Create resource group
az group create --name school-shooting-ml --location eastus

# Deploy container
az container create \
  --resource-group school-shooting-ml \
  --name school-shooting-api \
  --image school-shooting-ml-api:latest \
  --dns-name-label school-shooting-api \
  --ports 8000 \
  --environment-variables API_SECRET_KEY=your-secret-key
```

## Environment Variables

### Required
- `API_SECRET_KEY`: Secret key for JWT authentication

### Optional
- `API_ENVIRONMENT`: Environment (development/staging/production)
- `LOG_LEVEL`: Logging level (DEBUG/INFO/WARNING/ERROR)
- `API_WORKERS`: Number of Gunicorn workers (default: 4)
- `API_HOST`: API host (default: 0.0.0.0)
- `API_PORT`: API port (default: 8000)

## Monitoring & Logging

### Prometheus Metrics

Access metrics at: `http://localhost:9090`

Key metrics:
- `api_requests_total`: Total API requests
- `api_request_duration_seconds`: Request duration
- `predictions_total`: Total predictions made

### Grafana Dashboards

Access Grafana at: `http://localhost:3000`

Default credentials:
- Username: admin
- Password: (set in docker-compose.yml)

### Application Logs

Logs are written to:
- Console (stdout/stderr)
- File: `logs/app.log` (rotating)

View logs:
```bash
# Docker
docker-compose logs -f api

# Local
tail -f logs/app.log
```

## Health Checks

- **Health endpoint**: `GET /health`
- **Metrics endpoint**: `GET /metrics`

Example health check:
```bash
curl http://localhost:8000/health
```

## Security Considerations

1. **Change default secret key** in production
2. **Use HTTPS** (configure reverse proxy like Nginx)
3. **Implement rate limiting** for API endpoints
4. **Rotate JWT tokens** regularly
5. **Secure model files** with appropriate permissions
6. **Enable CORS** only for trusted origins
7. **Use environment variables** for sensitive data

## Scaling

### Horizontal Scaling
- Increase number of containers/replicas
- Use load balancer (ALB, Cloud Load Balancing, etc.)

### Vertical Scaling
- Increase `API_WORKERS` environment variable
- Allocate more CPU/memory to containers

## Troubleshooting

### API won't start
- Check model file exists in `data/models/`
- Verify all environment variables are set
- Check logs for detailed error messages

### Predictions failing
- Ensure model is loaded (check `/health` endpoint)
- Verify input features match training schema
- Check authentication token is valid

### High latency
- Monitor Prometheus metrics
- Check if model is loaded in memory
- Consider caching predictions
- Scale horizontally with load balancer

## Rollback Procedure

1. **Identify last working version**
2. **Pull previous Docker image**
3. **Update deployment** to use previous image
4. **Verify health checks** pass
5. **Monitor metrics** for stability
