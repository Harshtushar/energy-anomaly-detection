from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
import joblib
from pathlib import Path


MODEL_FILENAMES = {
    "iso": "isolation_forest_model.pkl",
    "lof": "local_outlier_factor_model.pkl",
    "rc": "elliptic_envelope_model.pkl",
}


def get_models_dir(model_dir=None):
    models_dir = Path(model_dir) if model_dir else Path(__file__).resolve().parents[2] / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def map_anomaly_labels(labels):
    return [0 if label == 1 else 1 for label in labels]


def save_models(iso, lof, rc, model_dir=None):
    models_dir = get_models_dir(model_dir)
    joblib.dump(iso, models_dir / MODEL_FILENAMES["iso"])
    joblib.dump(lof, models_dir / MODEL_FILENAMES["lof"])
    joblib.dump(rc, models_dir / MODEL_FILENAMES["rc"])


def load_models(model_dir=None):
    models_dir = get_models_dir(model_dir)
    model_paths = {name: models_dir / filename for name, filename in MODEL_FILENAMES.items()}
    missing_paths = [str(path) for path in model_paths.values() if not path.exists()]

    if missing_paths:
        raise FileNotFoundError(
            "Saved anomaly model files were not found. Run anomaly detection first "
            "to train and save the models."
        )

    return {
        name: joblib.load(path)
        for name, path in model_paths.items()
    }


def run_anomaly_models(df, X, model_dir=None):
    models_dir = get_models_dir(model_dir)
    X_values = X.to_numpy() if hasattr(X, "to_numpy") else X

    if len(df) < 3:
        raise ValueError(
            "At least 3 rows are required after preprocessing to run anomaly detection."
        )

    lof_neighbors = min(100, len(df) - 1)
    
    # Isolation Forest Model
    iso = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42,
        n_jobs=1
    )

    iso.fit(X_values)
    df["iso_anomaly"] = map_anomaly_labels(iso.predict(X_values))

    # Local Outlier Factor (LOF Model)
    lof = LocalOutlierFactor(
        n_neighbors=lof_neighbors,
        contamination=0.05,
        novelty=True,
    )

    lof.fit(X_values)
    df["lof_anomaly"] = map_anomaly_labels(lof.predict(X_values))

    # Robust Covariance (Elliptic Envelope)
    rc = EllipticEnvelope(
        contamination=0.05,
        support_fraction=0.8
        )

    rc.fit(X_values)
    df["rc_anomaly"] = map_anomaly_labels(rc.predict(X_values))

    save_models(iso, lof, rc, models_dir)

    return df, iso


def predict_with_saved_models(df, X, model_dir=None):
    models = load_models(model_dir)
    X_values = X.to_numpy() if hasattr(X, "to_numpy") else X

    df["iso_anomaly"] = map_anomaly_labels(models["iso"].predict(X_values))
    df["lof_anomaly"] = map_anomaly_labels(models["lof"].predict(X_values))
    df["rc_anomaly"] = map_anomaly_labels(models["rc"].predict(X_values))

    return df


def get_model_predictions(df, X, model_dir=None):
    return run_anomaly_models(df, X, model_dir=model_dir)
