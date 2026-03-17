import pickle
import os
import pandas as pd
from fastapi import FastAPI, Request, Form, HTTPException, Security
from fastapi.responses import HTMLResponse
from fastapi.security import APIKeyHeader
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

API_KEY        = os.getenv("API_KEY", "dev-secret-key")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

app = FastAPI(
    title="House Price Predictor API",
    description="Predict house prices using a trained GradientBoosting model.",
    version="2.0.0",
)
templates = Jinja2Templates(directory="templates")

with open("model.pkl", "rb") as f:
    saved      = pickle.load(f)
    pipeline   = saved["pipeline"]
    FEATURES   = saved["features"]
    MODEL_NAME = saved["model_name"]


# ── Auth dependency ────────────────────────────────────────────────────────────
def require_api_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key. Pass it as X-API-Key header.")
    return key


# ── Pydantic schema ────────────────────────────────────────────────────────────
class HouseFeatures(BaseModel):
    area:           float = Field(..., gt=0,   description="Area in square feet")
    bedrooms:       int   = Field(..., ge=1, le=20)
    bathrooms:      int   = Field(..., ge=1, le=20)
    location_score: float = Field(..., ge=1, le=10, description="Neighbourhood score 1–10")
    garage:         int   = Field(..., ge=0, le=5,  description="Number of garage spaces")
    year_built:     int   = Field(..., ge=1900, le=2024)
    floors:         int   = Field(..., ge=1, le=5)


# ── Helper ─────────────────────────────────────────────────────────────────────
def make_df(area, bedrooms, bathrooms, location_score, garage, year_built, floors):
    return pd.DataFrame(
        [[area, bedrooms, bathrooms, location_score, garage, year_built, floors]],
        columns=FEATURES,
    )


# ── UI routes ──────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "model_name": MODEL_NAME})


@app.post("/predict-form", response_class=HTMLResponse)
async def predict_form(
    request:        Request,
    area:           float = Form(...),
    bedrooms:       int   = Form(...),
    bathrooms:      int   = Form(...),
    location_score: float = Form(...),
    garage:         int   = Form(...),
    year_built:     int   = Form(...),
    floors:         int   = Form(...),
):
    error           = None
    predicted_price = None
    importance_data = None

    if area <= 0:
        error = "Area must be greater than 0."
    elif not (1 <= location_score <= 10):
        error = "Location score must be between 1 and 10."
    elif not (1900 <= year_built <= 2024):
        error = "Year built must be between 1900 and 2024."
    else:
        df              = make_df(area, bedrooms, bathrooms, location_score, garage, year_built, floors)
        predicted_price = round(float(pipeline.predict(df)[0]), 2)

        # Feature importance from the underlying model
        model       = pipeline.named_steps["model"]
        importances = model.feature_importances_
        importance_data = {
            feat: round(float(imp) * 100, 1)
            for feat, imp in zip(FEATURES, importances)
        }

    return templates.TemplateResponse("index.html", {
        "request":        request,
        "model_name":     MODEL_NAME,
        "predicted_price": predicted_price,
        "error":          error,
        "importance_data": importance_data,
        # repopulate form
        "area":           area,
        "bedrooms":       bedrooms,
        "bathrooms":      bathrooms,
        "location_score": location_score,
        "garage":         garage,
        "year_built":     year_built,
        "floors":         floors,
    })


# ── JSON API routes (protected) ────────────────────────────────────────────────
@app.post("/predict", summary="Predict house price (API key required)")
async def predict(features: HouseFeatures, _: str = Security(require_api_key)):
    df    = make_df(**features.model_dump())
    price = round(float(pipeline.predict(df)[0]), 2)
    low   = round(price * 0.92, 2)
    high  = round(price * 1.08, 2)
    return {
        "predicted_price": price,
        "price_range":     {"low": low, "high": high},
        "model_used":      MODEL_NAME,
    }


@app.get("/feature-importance", summary="Get feature importance scores (API key required)")
async def feature_importance(_: str = Security(require_api_key)):
    model       = pipeline.named_steps["model"]
    importances = model.feature_importances_
    return {
        "model": MODEL_NAME,
        "importance": {
            feat: round(float(imp) * 100, 2)
            for feat, imp in sorted(zip(FEATURES, importances), key=lambda x: -x[1])
        },
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME, "features": FEATURES}
