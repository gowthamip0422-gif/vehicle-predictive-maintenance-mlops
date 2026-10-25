"""Generate reproducible SYNTHETIC vehicle telemetry. Not real failure outcomes."""
import argparse
import numpy as np
import pandas as pd
from src.config import DATA_FILE

def generate(n_rows: int = 1500, seed: int = 42) -> pd.DataFrame:
    if n_rows < 20:
        raise ValueError("n_rows must be at least 20")
    rng = np.random.default_rng(seed)
    age = rng.integers(0, 16, n_rows)
    mileage = np.clip(age * 13500 + rng.normal(12000, 16000, n_rows), 100, 280000)
    engine_temp = np.clip(rng.normal(93, 13, n_rows) + age * .28, 65, 145)
    battery = np.clip(rng.normal(12.6, .48, n_rows) - age * .025, 10.5, 14.5)
    oil = np.clip(rng.normal(290, 65, n_rows) - age * 2.0, 70, 450)
    vibration = np.clip(rng.normal(2.4, 1.05, n_rows) + age * .11, .1, 9.5)
    risk = (-4.1 + .075 * (engine_temp-90) + 1.9 * (12.3-battery)
            + .011 * (260-oil) + .38 * (vibration-2) + .000007 * mileage)
    probabilities = 1 / (1 + np.exp(-np.clip(risk, -30, 30)))
    failure = rng.binomial(1, probabilities)
    return pd.DataFrame({
        "vehicle_id": [f"DEMO-{i:06d}" for i in range(n_rows)],
        "engine_temp_c": np.round(engine_temp, 2),
        "battery_voltage": np.round(battery, 2),
        "oil_pressure_kpa": np.round(oil, 2),
        "mileage_km": np.round(mileage, 1),
        "vibration_mm_s": np.round(vibration, 2),
        "vehicle_age_years": age,
        "failure_next_30_days": failure,
    })

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    df = generate(args.rows, args.seed)
    df.to_csv(DATA_FILE, index=False)
    print(f"Saved {len(df)} synthetic vehicle rows to {DATA_FILE}; failure rate={df.failure_next_30_days.mean():.3f}")

if __name__ == "__main__":
    main()
