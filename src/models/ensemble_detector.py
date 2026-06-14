
# Ensemble Voting

def ensemble_vote(df):
    df["anomaly_votes"] = (
        df["iso_anomaly"] +
        df["lof_anomaly"] +
        df["rc_anomaly"]
    )

    df["final_anomaly"] = (df["anomaly_votes"] >= 2).astype(int)

    return df