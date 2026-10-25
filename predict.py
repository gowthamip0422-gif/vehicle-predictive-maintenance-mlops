"""Model loading and inference (never load joblib artifacts from untrusted sources)."""
from functools import lru_cache
import joblib
import pandas as pd
from src.config import MODEL_FILE, FEATURES

@lru_cache(maxsize=1)
def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError("Model not found. Run: python -m src.train")
    return joblib.load(MODEL_FILE)

def predict_one(values: dict) -> dict:
    model = load_model()
    probability = float(model.predict_proba(pd.DataFrame([values], columns=FEATURES))[0, 1])
    return {"failure_probability": round(probability, 4),
            "risk_level": "HIGH" if probability >= .65 else "MEDIUM" if probability >= .35 else "LOW",
            "threshold": 0.5, "predicted_failure": probability >= .5}
