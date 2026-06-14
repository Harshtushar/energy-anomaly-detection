import pandas as pd

def create_features(df):
    df = df.sort_values(by=["building", "timestamp"]).reset_index(drop=True)        # Sort data

    # Temporal features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)

    # Rolling Statistics (7-Day Window) (7 days = 168 hours)
    df["rolling_mean_7d"] = (
        df.groupby("building")["meter_reading"]
        .transform(lambda x: x.rolling(window=168, min_periods=1).mean())
    )
    df["rolling_std_7d"] = (
        df.groupby("building")["meter_reading"]
        .transform(lambda x: x.rolling(window=168, min_periods=1).std())
    )
    df["deviation_from_baseline"] = (
        df["meter_reading"] - df["rolling_mean_7d"]
    )
    # Lag feature
    df["lag1"] = df.groupby("building")["meter_reading"].shift(1)
    df["lag24"] = df.groupby("building")["meter_reading"].shift(24)

    df = df.ffill().bfill()     # Handle missing values from lag

    df = df.drop_duplicates(subset=["building", "timestamp"])       # Remove duplicates
    print("Shape after removing duplicates:", df.shape)

    return df