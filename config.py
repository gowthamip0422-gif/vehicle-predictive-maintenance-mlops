from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "vehicle_sensor_data.csv"
MODEL_FILE = ROOT / "artifacts" / "model.joblib"
METRICS_FILE = ROOT / "artifacts" / "metrics.json"
FEATURES = ["engine_temp_c", "battery_voltage", "oil_pressure_kpa", "mileage_km", "vibration_mm_s", "vehicle_age_years"]
TARGET = "failure_next_30_days"
