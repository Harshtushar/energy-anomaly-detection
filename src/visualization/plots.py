import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


def _finalize_figure(fig, show):
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_feature_importance(df, features, show=True):
    feature_importance = (
        df[features + ["final_anomaly"]]
        .corr(numeric_only=True)["final_anomaly"]
        .drop("final_anomaly")
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    feature_importance.plot(kind="bar", ax=ax)

    ax.set_title("Feature Importance (Correlation with Anomaly)")
    ax.set_ylabel("Correlation Score")
    ax.set_xlabel("Features")

    return _finalize_figure(fig, show)


def plot_anomaly_distribution(df, show=True):
    fig, ax = plt.subplots(figsize=(6, 4))

    sns.countplot(x="final_anomaly", data=df, order=[0, 1], ax=ax)

    ax.set_title("Anomaly Distribution")
    ax.set_xlabel("Anomaly Label")
    ax.set_ylabel("Count")

    return _finalize_figure(fig, show)


def plot_energy_anomalies(df, show=True):
    anomaly_counts = df.groupby("building")["final_anomaly"].sum().sort_values(ascending=False)
    if anomaly_counts.empty:
        raise ValueError("No building data is available for plotting.")

    sample_building = anomaly_counts.index[0]
    df_sample = df[df["building"] == sample_building].sort_values("timestamp")
    anomalies = df_sample[df_sample["final_anomaly"] == 1]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(
        df_sample["timestamp"],
        df_sample["meter_reading"],
        label="Energy Usage",
        linewidth=0.8,
    )

    if not anomalies.empty:
        ax.scatter(
            anomalies["timestamp"],
            anomalies["meter_reading"],
            color="red",
            label="Anomaly",
            s=40,
            zorder=5,
        )

    title_suffix = (
        "with Anomalies" if not anomalies.empty else "Sample Building"
    )
    ax.set_title(f"Energy Usage - {sample_building} ({title_suffix})")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Meter Reading")
    ax.legend()

    return _finalize_figure(fig, show)


def plot_monthly_anomalies(df, show=True):
    monthly_anomalies = df.groupby("month")["final_anomaly"].sum().sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    monthly_anomalies.plot(kind="bar", ax=ax)

    ax.set_title("Monthly Anomaly Count")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Anomalies")

    return _finalize_figure(fig, show)
