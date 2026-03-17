import pandas as pd
import glob

def load_data(data_path):
    csv_files = glob.glob(data_path)
    
    print("Total files found:", len(csv_files))
    
    df_list = [pd.read_csv(file) for file in csv_files]
    df = pd.concat(df_list, ignore_index=True)
    
    print("Data loaded successfully!")
    print("Shape:", df.shape)
    
    return df

# Data cleaning and preprocessing

def preprocess_data(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"])       # Convert timestamp
    df = df.sort_values(by=["building_id","timestamp"]).reset_index(drop=True)       # Sort data

    # Melt data
    building_cols = [col for col in df.columns if col.startswith("Panther")]

    print("Number of building columns:", len(building_cols))

    df = df.melt(
        id_vars=["timestamp"],      # column to keep unchanged
        value_vars=building_cols,   # columns to melt
        var_name="building",        # new column name for buildings
        value_name="meter_reading"  # new column name for values
    )

    print("Shape after melt:", df.shape)

    # Drop rows where meter_reading is missing
    df = df.dropna(subset=["meter_reading"]).reset_index(drop=True)

    # Fill missing timestamps if any
    df = df.sort_values("timestamp")
    df = df.ffill().bfill()
    print("Shape:", df.shape)
    print("Missing values:", df.isnull().sum().sum())
    print("Timestamp dtype:", df["timestamp"].dtype)

    # Cap outliers
    lower = df["meter_reading"].quantile(0.01)
    upper = df["meter_reading"].quantile(0.99)

    df["meter_reading"] = df["meter_reading"].clip(lower, upper)

    # Convert buildings to category (Memory optimization)(Converts Buildings strings into integer codes)
    df["building"] = df["building"].astype("category")

    return df