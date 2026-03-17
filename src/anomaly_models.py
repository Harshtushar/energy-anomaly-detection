from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
import joblib
import os

def run_anomaly_models(df, X):

    os.makedirs("../models", exist_ok=True)
    
    # Isolation Forest Model
    iso = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42,
        n_jobs=-1
    )

    df["iso_anomaly"] = iso.fit_predict(X)
    df["iso_anomaly"] = df["iso_anomaly"].map({1: 0, -1: 1})

    joblib.dump(iso, "../models/isolation_forest_model.pkl")

    # Local Outlier Factor (LOF Model)
    lof = LocalOutlierFactor(
        n_neighbors=100,
        contamination=0.05
    )

    df["lof_anomaly"] = lof.fit_predict(X)
    df["lof_anomaly"] = df["lof_anomaly"].map({1: 0, -1: 1})

    # Robust Covariance (Elliptic Envelope)
    rc = EllipticEnvelope(
        contamination=0.05,
        support_fraction=0.8
        )

    df["rc_anomaly"] = rc.fit_predict(X)
    df["rc_anomaly"] = df["rc_anomaly"].map({1: 0, -1: 1})

    # Ensemble Voting
    df["anomaly_votes"] = (
        df["iso_anomaly"] +
        df["lof_anomaly"] +
        df["rc_anomaly"]
    )

    df["final_anomaly"] = (df["anomaly_votes"] >= 2).astype(int)

    return df, iso