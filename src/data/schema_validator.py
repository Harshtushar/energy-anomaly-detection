def validate_schema(
    df,
    timestamp_col="timestamp",
    reading_col="meter_reading",
    building_col="building",
):
    if df is None or df.empty:
        raise ValueError("The uploaded dataset is empty.")

    if timestamp_col not in df.columns:
        raise ValueError(
            f"Missing timestamp column '{timestamp_col}'. "
            "Please choose the correct timestamp column."
        )

    has_long_columns = (
        building_col in df.columns and reading_col in df.columns
    )

    if has_long_columns:
        return True

    wide_value_columns = [col for col in df.columns if col != timestamp_col]
    if not wide_value_columns:
        raise ValueError(
            "Wide-format data must contain at least one reading column besides "
            "the timestamp column."
        )

    return True
