# Quick Start Reference Card

## 🚀 First Time Setup (Run Once)

```bash
# 1. Create virtual environment
python3 -m venv venv

# 2. Activate it
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Setup environment
cp .env.example .env
# Edit .env and set API_SECRET_KEY

# 5. Train model (requires data file in data/raw/)
python scripts/train_model.py
```

---

## 📅 Daily Usage

```bash
# Start work
cd /home/user/Public_K-12_School_Shooting_Analysis
source venv/bin/activate
uvicorn src.api.app:app --reload

# Stop work
# Press Ctrl+C
deactivate
```

---

## 🧪 Quick Tests

```bash
# Check API health
curl http://localhost:8000/health

# Get auth token
curl -X POST "http://localhost:8000/auth/token?username=admin&password=admin123"

# Make prediction (replace $TOKEN)
curl -X POST "http://localhost:8000/predict" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d @test_prediction.json
```

---

## 📊 Useful URLs

- **API Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Root**: http://localhost:8000/
- **Health**: http://localhost:8000/health
- **Metrics**: http://localhost:8000/metrics

---

## 📝 Important Files

| File | Purpose | When to Edit |
|------|---------|--------------|
| `.env` | Environment variables | Change API_SECRET_KEY |
| `config/config.yaml` | All settings | Adjust model parameters |
| `data/raw/` | Put your Excel file here | Before training |
| `logs/app.log` | Application logs | For debugging |
| `data/models/` | Trained models saved here | After training |

---

## 🐛 Quick Fixes

```bash
# Module not found
pip install -r requirements.txt

# Port in use
uvicorn src.api.app:app --port 8001

# View logs
tail -f logs/app.log

# Retrain model
python scripts/train_model.py
```

---

## 💡 Pro Tips

1. **Always activate venv first**: `source venv/bin/activate`
2. **Use --reload for development**: `uvicorn ... --reload`
3. **Check logs when stuck**: `tail -f logs/app.log`
4. **Test with /docs**: http://localhost:8000/docs
5. **Backup your model**: `cp data/models/*.pkl backups/`

---

For detailed instructions, see: **LOCAL_DEPLOYMENT_GUIDE.md**
