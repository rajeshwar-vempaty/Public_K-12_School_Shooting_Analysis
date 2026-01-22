# Local Deployment Guide (Without Docker)

Complete step-by-step guide to run the production ML system locally without Docker.

---

## ⚙️ Prerequisites

### Required Software
1. **Python 3.10 or 3.11** (Python 3.12+ may have compatibility issues)
2. **pip** (Python package manager)
3. **Git** (already installed)

### Check Your Python Version
```bash
python --version
# or
python3 --version
```

If Python is not installed or version < 3.10:
- **Windows**: Download from https://www.python.org/downloads/
- **macOS**: `brew install python@3.10` or download from python.org
- **Linux**: `sudo apt install python3.10 python3-pip` (Ubuntu/Debian)

---

## 📋 STEP-BY-STEP DEPLOYMENT

### STEP 1: Navigate to Project Directory

```bash
cd /home/user/Public_K-12_School_Shooting_Analysis
```

### STEP 2: Create Python Virtual Environment

This isolates project dependencies from your system Python.

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

You should see `(venv)` in your terminal prompt after activation.

### STEP 3: Upgrade pip

```bash
pip install --upgrade pip
```

### STEP 4: Install Python Dependencies

```bash
pip install -r requirements.txt
```

This will install all 40+ packages. **Time: 2-5 minutes depending on internet speed.**

**If you encounter errors:**
- **On Linux**: You may need system dependencies:
  ```bash
  sudo apt-get install python3-dev build-essential
  ```
- **On macOS**: Install Xcode command line tools:
  ```bash
  xcode-select --install
  ```
- **On Windows**: Install Microsoft C++ Build Tools from:
  https://visualstudio.microsoft.com/visual-cpp-build-tools/

### STEP 5: Set Up Environment Variables

```bash
# Copy the example file
cp .env.example .env

# Edit the .env file
nano .env  # or use any text editor
```

**Minimal .env configuration:**
```bash
# Generate a secure key first (required for production!)
API_SECRET_KEY=your-secure-key-here
API_ENVIRONMENT=development

# Authentication credentials - CHANGE THESE IN PRODUCTION!
ADMIN_USERNAME=admin
ADMIN_PASSWORD=admin123

LOG_LEVEL=INFO
MODEL_VERSION=1.0.0
```

**Generate a secure secret key:**
```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

Copy the output and paste it as your `API_SECRET_KEY`.

**Important Security Notes:**
- In production (`API_ENVIRONMENT=production`), `API_SECRET_KEY` is **required**
- Always change `ADMIN_USERNAME` and `ADMIN_PASSWORD` from defaults
- Rate limiting is enabled by default (100 requests per 60 seconds)

### STEP 6: Prepare Your Data

**Option A: If you have the original Excel file**

1. Create the data directory:
   ```bash
   mkdir -p data/raw
   ```

2. Copy your Excel file to the data directory:
   ```bash
   cp "path/to/your/Public v3.1 K-12 School Shooting Database (9 25 2023).xlsx" data/raw/
   ```

3. Update the config file if your filename is different:
   ```bash
   nano config/config.yaml
   ```
   Find the `data.input_excel` line and update it with your filename.

**Option B: If you don't have data yet**

You can still run the API in "model-less" mode for testing. Skip to Step 8.

### STEP 7: Train the Model

**This step requires the Excel data file from Step 6.**

```bash
# Run the training script
python scripts/train_model.py
```

**What happens:**
1. Loads data from Excel (4 sheets)
2. Preprocesses all data (25+ transformations)
3. Engineers features
4. Splits train/test (75/25)
5. Encodes and scales features
6. Trains SVM model with hyperparameter tuning
7. Evaluates model performance
8. Saves model to `data/models/school_shooting_model_v1.0.0.pkl`

**Time:** 10-30 minutes depending on your CPU.

**Expected output:**
```
[INFO] Loading data...
[INFO] Preprocessing data...
[INFO] Engineering features...
[INFO] Training model with hyperparameter tuning...
[INFO] Evaluating model...
[INFO] Model saved to: data/models/school_shooting_model_v1.0.0.pkl

Test Set Performance:
  ACCURACY: 0.8600
  PRECISION: 0.9300
  RECALL: 0.8400
  F1: 0.8800
  ROC_AUC: 0.8500
```

### STEP 8: Start the API Server

**Development Mode (with auto-reload):**
```bash
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

**Production Mode (using Gunicorn):**
```bash
gunicorn src.api.app:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --access-logfile logs/access.log \
    --error-logfile logs/error.log
```

**Or use the startup script:**
```bash
chmod +x scripts/start_api.sh
./scripts/start_api.sh
```

**Expected output:**
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### STEP 9: Verify API is Running

Open a new terminal and test:

```bash
# Test health endpoint
curl http://localhost:8000/health

# Expected response:
# {
#   "status": "healthy",
#   "model_loaded": true,
#   "model_version": "1.0.0",
#   "uptime_seconds": 5.2
# }
```

**Or open in your browser:**
- API Docs: http://localhost:8000/docs
- Alternative Docs: http://localhost:8000/redoc
- Root: http://localhost:8000/

### STEP 10: Get Authentication Token

Use the credentials configured in your `.env` file (default: admin/admin123):

```bash
# Using default credentials
curl -X POST "http://localhost:8000/auth/token?username=admin&password=admin123"

# Or using your custom credentials from .env
curl -X POST "http://localhost:8000/auth/token?username=$ADMIN_USERNAME&password=$ADMIN_PASSWORD"
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 60
}
```

**Save the token:**
```bash
TOKEN="<paste-your-token-here>"
```

**Note:** Invalid credentials will return HTTP 401 Unauthorized. Check your `.env` file if authentication fails.

### STEP 11: Make a Test Prediction

Create a test file `test_prediction.json`:
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
  "City": "Test City",
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

**Make the prediction:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d @test_prediction.json
```

**Expected response:**
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

---

## 🧪 Testing the System

### Run Unit Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Open coverage report
# Linux/macOS: open htmlcov/index.html
# Windows: start htmlcov/index.html
```

### Test Individual Components

```bash
# Test API endpoints only
pytest tests/test_api.py -v

# Test preprocessing
pytest tests/test_preprocessing.py -v
```

---

## 📊 Using the Interactive API Documentation

1. **Open browser** to http://localhost:8000/docs

2. **Authorize** - Click the green "Authorize" button:
   - Get token first: Use `/auth/token` endpoint
   - Copy the `access_token` from response
   - Paste in "Value" field (without "Bearer")
   - Click "Authorize"

3. **Test endpoints** - Click on any endpoint:
   - Click "Try it out"
   - Fill in the request body
   - Click "Execute"
   - See the response below

---

## 📁 Understanding the File Structure

```
Your Working Directory:
├── data/
│   ├── raw/                    # Put your Excel file here
│   ├── processed/              # Cleaned CSV will be saved here
│   ├── models/                 # Trained model saved here
│   └── evaluation_results/     # Plots and metrics
├── logs/
│   └── app.log                 # Application logs
├── src/                        # Source code (don't modify)
├── config/
│   └── config.yaml             # Edit to change settings
├── scripts/
│   └── train_model.py          # Run to train model
└── .env                        # Your environment variables
```

---

## 🐛 Troubleshooting

### Problem: "Module not found" errors

**Solution:**
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

### Problem: "Model file not found"

**Solution:**
```bash
# Train the model first
python scripts/train_model.py

# Or check if file exists
ls -la data/models/
```

### Problem: Port 8000 already in use

**Solution:**
```bash
# Use a different port
uvicorn src.api.app:app --port 8001

# Or kill the process using port 8000
# Linux/macOS:
lsof -ti:8000 | xargs kill -9

# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Problem: API is slow or hanging

**Solution:**
```bash
# Check logs
tail -f logs/app.log

# Reduce workers (if using Gunicorn)
gunicorn src.api.app:app --workers 2 ...

# Check CPU/memory usage
top  # Linux/macOS
# or Task Manager on Windows
```

### Problem: Dependencies won't install

**Solution:**
```bash
# Install problematic packages individually
pip install numpy pandas scikit-learn
pip install fastapi uvicorn
pip install -r requirements.txt
```

### Problem: Excel file loading fails

**Solution:**
```bash
# Check file exists
ls -la "data/raw/Public v3.1 K-12 School Shooting Database (9 25 2023).xlsx"

# Check file permissions
chmod 644 data/raw/*.xlsx

# Verify Excel file is not corrupted
# Try opening in Excel/LibreOffice first
```

---

## 🔄 Daily Usage Workflow

### Starting Work

```bash
# 1. Navigate to project
cd /home/user/Public_K-12_School_Shooting_Analysis

# 2. Activate virtual environment
source venv/bin/activate

# 3. Start API
uvicorn src.api.app:app --reload
```

### Stopping Work

```bash
# 1. Stop API (Ctrl+C in terminal)

# 2. Deactivate virtual environment
deactivate
```

### Updating the Model

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Place new data in data/raw/

# 3. Retrain model
python scripts/train_model.py

# 4. Restart API
# Stop (Ctrl+C) and start again
uvicorn src.api.app:app --reload
```

---

## 📈 Monitoring & Logs

### View Application Logs

```bash
# Real-time log monitoring
tail -f logs/app.log

# View last 100 lines
tail -n 100 logs/app.log

# Search for errors
grep -i error logs/app.log

# Search for specific date
grep "2024-01-15" logs/app.log
```

### Check API Metrics

```bash
# Get Prometheus metrics
curl http://localhost:8000/metrics

# Check health status
curl http://localhost:8000/health

# Get model info (requires auth)
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/model/info
```

---

## 💾 Backup Your Work

### Backup Model

```bash
# Create backups directory
mkdir -p backups

# Backup current model
cp data/models/school_shooting_model_v1.0.0.pkl \
   backups/model_backup_$(date +%Y%m%d_%H%M%S).pkl

# List backups
ls -lh backups/
```

### Backup Data

```bash
# Backup processed data
cp data/processed/K-12_School_Cleandata.csv \
   backups/data_backup_$(date +%Y%m%d).csv
```

---

## 🔐 Production Deployment Checklist

When deploying to a server:

### Security (Critical!)
- [ ] Generate and set `API_SECRET_KEY` (minimum 32 characters)
      ```bash
      python3 -c "import secrets; print(secrets.token_urlsafe(32))"
      ```
- [ ] Set `API_ENVIRONMENT=production` in `.env`
- [ ] Change `ADMIN_USERNAME` to a non-default value
- [ ] Change `ADMIN_PASSWORD` to a strong password (16+ characters)
- [ ] Set up HTTPS with nginx or similar (never expose HTTP in production)
- [ ] Configure firewall to allow only necessary ports

### Deployment
- [ ] Use `gunicorn` instead of `uvicorn --reload`
- [ ] Create systemd service for auto-start
- [ ] Set up log rotation (logs/app.log can grow large)
- [ ] Monitor disk space
- [ ] Set up automated backups of `data/models/` directory

### Monitoring
- [ ] Configure Prometheus scraping from `/metrics` endpoint
- [ ] Set up Grafana dashboards (optional)
- [ ] Configure alerting for health check failures
- [ ] Test disaster recovery procedure

### Rate Limiting
The API includes built-in rate limiting (configurable in `config/config.yaml`):
- Default: 100 requests per 60 seconds per IP
- Excludes `/health` and `/metrics` endpoints
- Returns HTTP 429 when exceeded with `Retry-After` header

---

## 🆘 Getting Help

1. **Check logs first**: `tail -f logs/app.log`
2. **Test health endpoint**: `curl http://localhost:8000/health`
3. **Review error messages** in terminal
4. **Check this guide** for troubleshooting section
5. **Review API documentation**: http://localhost:8000/docs

---

## ✅ Success Criteria

You've successfully deployed when:

✅ Virtual environment is activated
✅ All dependencies installed without errors
✅ Model trained and saved
✅ API server starts without errors
✅ Health check returns `{"status": "healthy", "model_loaded": true}`
✅ Can get authentication token
✅ Can make predictions successfully
✅ Interactive docs accessible at /docs

---

**You're ready to use the production ML system locally!** 🚀
