# 🏠 House Price Predictor

A machine learning web app that predicts house prices based on property features like area, bedrooms, bathrooms, location score, garage, year built, and floors.

**Live Demo → [house-price-predictor-ne3l.onrender.com](https://house-price-predictor-ne3l.onrender.com)**

---

## Preview

![House Price Predictor UI](https://placehold.co/900x500/1a1a2e/ffffff?text=House+Price+Predictor)

---

## Features

- Predicts house prices using a **GradientBoosting** regression model
- Shows a **price range** (low / predicted / high) with a Chart.js bar chart
- Displays **feature importance** — what's driving the price
- REST API with **API key authentication**
- Auto-generated **Swagger docs** at `/docs`
- Clean, **mobile-responsive** dark UI

---

## Tech Stack

| Layer      | Technology                        |
|------------|-----------------------------------|
| ML Model   | scikit-learn (GradientBoosting)   |
| Backend    | FastAPI + Uvicorn                 |
| Frontend   | HTML + CSS + Chart.js             |
| Templating | Jinja2                            |
| Deployment | Render                            |

---

## Project Structure

```
house-price-predictor/
├── templates/
│   └── index.html       # Frontend UI
├── app.py               # FastAPI backend
├── train.py             # Model training script
├── dataset.csv          # 400-row housing dataset
├── model.pkl            # Trained model (GradientBoosting + StandardScaler pipeline)
├── requirements.txt
├── render.yaml          # Render deployment config
├── Procfile             # Railway deployment config
├── runtime.txt          # Python version pin
└── .env                 # API key (not committed)
```

---

## Model Performance

| Model               | MAE       | R²     |
|---------------------|-----------|--------|
| Random Forest       | $17,477   | 0.9767 |
| **Gradient Boosting** | **$12,891** | **0.9877** |

GradientBoosting was selected as the best model. Both models are wrapped in a `Pipeline` with `StandardScaler`.

### Features Used

| Feature        | Description                  |
|----------------|------------------------------|
| `area`         | Area in square feet          |
| `bedrooms`     | Number of bedrooms           |
| `bathrooms`    | Number of bathrooms          |
| `location_score` | Neighbourhood score (1–10) |
| `garage`       | Number of garage spaces      |
| `year_built`   | Year the house was built     |
| `floors`       | Number of floors             |

---

## Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/Shreya-bangera/house-price-predictor.git
cd house-price-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Create a `.env` file
```bash
API_KEY=your-secret-key
```

### 4. Train the model (optional — `model.pkl` is already included)
```bash
python train.py
```

### 5. Start the server
```bash
uvicorn app:app --reload
```

Open → [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## API Usage

### `POST /predict` — Predict house price
Requires `X-API-Key` header.

```bash
curl -X POST "https://house-price-predictor-ne3l.onrender.com/predict" \
  -H "X-API-Key: your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "area": 1800,
    "bedrooms": 3,
    "bathrooms": 2,
    "location_score": 7,
    "garage": 1,
    "year_built": 2005,
    "floors": 2
  }'
```

**Response:**
```json
{
  "predicted_price": 297780.0,
  "price_range": { "low": 273957.0, "high": 321602.0 },
  "model_used": "GradientBoosting"
}
```

### `GET /feature-importance` — Get feature importance scores
Requires `X-API-Key` header.

```bash
curl "https://house-price-predictor-ne3l.onrender.com/feature-importance" \
  -H "X-API-Key: your-secret-key"
```

### `GET /health` — Health check
```bash
curl "https://house-price-predictor-ne3l.onrender.com/health"
```

### Swagger UI
Interactive API docs available at:
```
https://house-price-predictor-ne3l.onrender.com/docs
```

---

## Deployment

This app is deployed on **Render** (free tier).

> ⚠️ Free tier spins down after 15 minutes of inactivity. First request after idle may take ~30 seconds to wake up.

To deploy your own instance:
1. Fork this repo
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your fork — Render auto-detects `render.yaml`
4. Add `API_KEY` in the Environment tab
5. Deploy

---

## License

MIT
