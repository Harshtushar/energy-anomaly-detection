import glob

import pandas as pd


def load_data(data_path, sample_size=None):
    csv_files = glob.glob(data_path)

    print("Total files found:", len(csv_files))

    df_list = [pd.read_csv(file) for file in csv_files]
    df = pd.concat(df_list, ignore_index=True)

    print("Data loaded successfully!")
    print("Shape:", df.shape)

    if sample_size:
        df = df.sample(sample_size, random_state=42)

    return df


def preprocess_data(
    df,
    timestamp_col="timestamp",
    reading_col="meter_reading",
    building_col="building",
):
    df = df.copy()

    rename_map = {}
    if timestamp_col in df.columns and timestamp_col != "timestamp":
        rename_map[timestamp_col] = "timestamp"
    if reading_col in df.columns and reading_col != "meter_reading":
        rename_map[reading_col] = "meter_reading"
    if building_col in df.columns and building_col != "building":
        rename_map[building_col] = "building"

    if rename_map:
        df = df.rename(columns=rename_map)

    timestamp_col = "timestamp"
    reading_col = "meter_reading"
    building_col = "building"

    if timestamp_col not in df.columns:
        raise KeyError(f"Missing required timestamp column: {timestamp_col}")

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col]).reset_index(drop=True)

    is_long_format = building_col in df.columns and reading_col in df.columns

    if is_long_format:
        df = df[[timestamp_col, building_col, reading_col]].copy()
        print("Input already in long format; skipping melt.")
    else:
        building_cols = [
            col
            for col in df.columns
            if col != timestamp_col and not str(col).startswith("Unnamed:")
        ]

        if not building_cols:
            raise ValueError(
                "No building columns found to melt. Expected wide data with "
                "one timestamp column and one or more building reading columns."
            )

        print("Number of building columns:", len(building_cols))

        df = df.melt(
            id_vars=[timestamp_col],
            value_vars=building_cols,
            var_name=building_col,
            value_name=reading_col,
        )

        print("Shape after melt:", df.shape)

    df[reading_col] = pd.to_numeric(df[reading_col], errors="coerce")
    df = df.dropna(subset=[reading_col]).reset_index(drop=True)

    sort_cols = [building_col, timestamp_col]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    lower = df[reading_col].quantile(0.01)
    upper = df[reading_col].quantile(0.99)
    df[reading_col] = df[reading_col].clip(lower, upper)

    df[building_col] = df[building_col].astype("category")

    print("Shape:", df.shape)
    print("Missing values:", df.isnull().sum().sum())
    print("Timestamp dtype:", df[timestamp_col].dtype)

    return df
