import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_feature_importance(df, features):
    feature_importance = (df[features + ["final_anomaly"]].corr()["final_anomaly"] \
                        .drop("final_anomaly") \
                        .sort_values(ascending=False)
    )
    plt.figure(figsize=(10,6))
    feature_importance.plot(kind="bar")

    plt.title("Feature Importance (Correlation with Anomaly)")
    plt.ylabel("Correlation Score")
    plt.xlabel("Features")

    plt.tight_layout()
    plt.savefig("../results/feature_importance.png")
    plt.show()

def plot_anomaly_distribution(df):
    plt.figure(figsize=(6,4))

    sns.countplot(x="final_anomaly", data=df)

    plt.title("Anomaly Distribution")
    plt.xlabel("Anomaly Label")
    plt.ylabel("Count")

    plt.savefig("../results/anomaly_distribution.png")
    plt.show()

def plot_energy_anomalies(df):
    sample_building = df[df["final_anomaly"] == 1]["building"].iloc[0]

    df_sample = df[df["building"] == sample_building]
    # Plotting energy usage
    plt.figure(figsize=(14,6))

    plt.plot(
        df_sample["timestamp"],
        df_sample["meter_reading"],
        label="Energy Usage",
        linewidth=0.8
    )

    anomalies = df_sample[df_sample["final_anomaly"] == 1]

    plt.scatter(
        anomalies["timestamp"],
        anomalies["meter_reading"],
        color="red",
        label="Anomaly",
        s=40,
        zorder = 5
    )

    plt.title(f"Energy Usage with Anomalies - {sample_building}")
    plt.xlabel("Timestamp")
    plt.ylabel("Meter Reading")

    plt.legend()

    plt.savefig("../results/energy_anomalies.png")
    plt.show()

def plot_monthly_anomalies(df):
    monthly_anomalies = df.groupby("month")["final_anomaly"].sum()

    plt.figure(figsize=(8,5))

    monthly_anomalies.plot(kind="bar")

    plt.title("Monthly Anomaly Count")
    plt.xlabel("Month")
    plt.ylabel("Number of Anomalies")

    plt.savefig("../results/monthly_anomalies.png")
    plt.show()