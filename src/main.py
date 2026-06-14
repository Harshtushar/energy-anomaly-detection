from src.data.preprocessing import load_data, preprocess_data
from src.data.feature_engineering import create_features
from src.data.schema_validator import validate_schema

from src.models.baseline_models import get_model_predictions, predict_with_saved_models
from src.models.ensemble_detector import ensemble_vote


FEATURES = [
    "meter_reading",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "rolling_mean_7d",
    "rolling_std_7d",
    "deviation_from_baseline",
    "lag1",
    "lag24"
]


def prepare_pipeline_data(
    df,
    timestamp_col="timestamp",
    reading_col="meter_reading",
    building_col="building",
):
    validate_schema(
        df,
        timestamp_col=timestamp_col,
        reading_col=reading_col,
        building_col=building_col,
    )

    df = preprocess_data(
        df,
        timestamp_col=timestamp_col,
        reading_col=reading_col,
        building_col=building_col,
    )

    df = create_features(df)

    X = df[FEATURES].ffill().bfill().fillna(0)
    df[FEATURES] = X

    return df, X


def run_pipeline(
    df,
    timestamp_col="timestamp",
    reading_col="meter_reading",
    building_col="building",
    model_dir=None,
):
    df, X = prepare_pipeline_data(
        df,
        timestamp_col=timestamp_col,
        reading_col=reading_col,
        building_col=building_col,
    )

    df, iso = get_model_predictions(df, X, model_dir=model_dir)

    df = ensemble_vote(df)

    return df


def run_live_prediction(
    df,
    timestamp_col="timestamp",
    reading_col="meter_reading",
    building_col="building",
    model_dir=None,
):
    df, X = prepare_pipeline_data(
        df,
        timestamp_col=timestamp_col,
        reading_col=reading_col,
        building_col=building_col,
    )

    df = predict_with_saved_models(df, X, model_dir=model_dir)
    df = ensemble_vote(df)

    return df


if __name__ == "__main__":
    data_path = "data/BDG2/raw/*.csv"

    df = load_data(data_path, sample_size=100000)

    results = run_pipeline(df)

    print(results.head())
    print(results["final_anomaly"].value_counts())
