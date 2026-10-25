"""Train/evaluate a transparent baseline model on synthetic data."""
import argparse
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from src.config import DATA_FILE, MODEL_FILE, METRICS_FILE, FEATURES, TARGET
from src.validate import validate_frame

def train(input_path=DATA_FILE, model_path=MODEL_FILE, metrics_path=METRICS_FILE, use_mlflow=False):
    df = pd.read_csv(input_path)
    validate_frame(df)
    X, y = df[FEATURES], df[TARGET].astype(int)
    if y.nunique() < 2:
        raise ValueError("Training requires both positive and negative labels")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.25, random_state=42, stratify=y)
    preprocessor = ColumnTransformer([("numeric", SimpleImputer(strategy="median"), FEATURES)])
    pipeline = Pipeline([("preprocessing", preprocessor),
                         ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    pipeline.fit(X_train, y_train)
    prob = pipeline.predict_proba(X_test)[:, 1]
    pred = (prob >= .5).astype(int)
    metrics = {"roc_auc": float(roc_auc_score(y_test, prob)),
               "average_precision": float(average_precision_score(y_test, prob)),
               "precision": float(precision_score(y_test, pred, zero_division=0)),
               "recall": float(recall_score(y_test, pred, zero_division=0)),
               "f1": float(f1_score(y_test, pred, zero_division=0)),
               "test_rows": int(len(X_test)), "train_rows": int(len(X_train)),
               "positive_rate_test": float(y_test.mean()),
               "decision_threshold": 0.5,
               "disclaimer": "Synthetic demonstration data. Not validated for real vehicle maintenance decisions."}
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    if use_mlflow:
        try:
            import mlflow
        except ImportError as exc:
            raise RuntimeError("Install mlflow to use --mlflow") from exc
        mlflow.set_experiment("vehicle-predictive-maintenance-demo")
        with mlflow.start_run():
            mlflow.log_params({"model": "logistic_regression", "seed": 42, "test_size": .25})
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
            mlflow.log_artifact(str(metrics_path))
    print(json.dumps(metrics, indent=2))
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlflow", action="store_true", help="Opt in to MLflow tracking")
    args = parser.parse_args()
    train(use_mlflow=args.mlflow)
