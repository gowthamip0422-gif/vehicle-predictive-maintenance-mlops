import pandas as pd
import pytest
from fastapi.testclient import TestClient
from src.generate_data import generate
from src.validate import validate_frame
from src.train import train
from api.main import app

def test_generate_is_deterministic():
    pd.testing.assert_frame_equal(generate(100, 10), generate(100, 10))

def test_valid_synthetic_dataset():
    validate_frame(generate(100))

def test_missing_column_fails():
    df = generate(100).drop(columns=["battery_voltage"])
    with pytest.raises(ValueError, match="Missing columns"):
        validate_frame(df)

def test_train_writes_artifacts(tmp_path):
    input_file = tmp_path / "synthetic.csv"
    generate(400).to_csv(input_file, index=False)
    metrics = train(input_file, tmp_path / "model.joblib", tmp_path / "metrics.json")
    assert 0 <= metrics["roc_auc"] <= 1
    assert (tmp_path / "model.joblib").exists()

def test_health_endpoint():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_prediction_input_validation():
    response = TestClient(app).post("/predict", json={"battery_voltage": -99})
    assert response.status_code == 422
