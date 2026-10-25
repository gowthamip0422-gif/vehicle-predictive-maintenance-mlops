"""Input schema, missing data and simple domain checks."""
import pandas as pd
from src.config import FEATURES, TARGET

LIMITS = {
    "engine_temp_c": (40, 180), "battery_voltage": (8, 17),
    "oil_pressure_kpa": (0, 800), "mileage_km": (0, 700000),
    "vibration_mm_s": (0, 30), "vehicle_age_years": (0, 50),
}

def validate_frame(df: pd.DataFrame, training: bool = True) -> None:
    required = FEATURES + ([TARGET] if training else [])
    missing = [c for c in required if c not in df]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df.empty:
        raise ValueError("Input dataset is empty")
    for col in FEATURES:
        values = pd.to_numeric(df[col], errors="coerce")
        if values.isna().all():
            raise ValueError(f"Feature {col} has no valid numeric values")
        lo, hi = LIMITS[col]
        if ((values.dropna() < lo) | (values.dropna() > hi)).any():
            raise ValueError(f"Feature {col} has values outside [{lo}, {hi}]")
    if training and not df[TARGET].dropna().isin([0, 1]).all():
        raise ValueError("Target must contain only 0 or 1")
    if training and df[TARGET].isna().any():
        raise ValueError("Target may not be missing")
