"""FastAPI inference service: uvicorn api.main:app --reload"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.predict import predict_one

app = FastAPI(title="Vehicle Predictive Maintenance API", version="1.0.0",
              description="Portfolio demo only — trained on synthetic vehicle data.")

class VehicleTelemetry(BaseModel):
    engine_temp_c: float = Field(ge=40, le=180)
    battery_voltage: float = Field(ge=8, le=17)
    oil_pressure_kpa: float = Field(ge=0, le=800)
    mileage_km: float = Field(ge=0, le=700000)
    vibration_mm_s: float = Field(ge=0, le=30)
    vehicle_age_years: int = Field(ge=0, le=50)

@app.get("/health")
def health():
    return {"status": "ok", "model_trained": __import__("src.config", fromlist=["MODEL_FILE"]).MODEL_FILE.exists()}

@app.post("/predict")
def predict(vehicle: VehicleTelemetry):
    try:
        return predict_one(vehicle.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
