from pathlib import Path
import importlib
import sys
from uuid import uuid4

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.auth_db import (
    authenticate_user,
    create_user,
    delete_run_history,
    get_user_history,
    init_auth_db,
    save_run_history,
)
import src.main as pipeline_main
from src.visualization.plots import (
    plot_anomaly_distribution,
    plot_energy_anomalies,
    plot_feature_importance,
    plot_monthly_anomalies,
)

pipeline_main = importlib.reload(pipeline_main)
FEATURES = pipeline_main.FEATURES
run_pipeline = pipeline_main.run_pipeline
run_live_prediction = getattr(pipeline_main, "run_live_prediction", None)


st.set_page_config(
    page_title="Energy Anomaly Detection",
    layout="wide",
)

RESULTS_HISTORY_ROOT = PROJECT_ROOT / "data" / "history_results"
MODEL_ARTIFACTS_ROOT = PROJECT_ROOT / "data" / "live_models"


def init_session_state():
    st.session_state.setdefault("auth_mode", None)
    st.session_state.setdefault("auth_page", "login")
    st.session_state.setdefault("app_page", "dashboard")
    st.session_state.setdefault("app_page_selector", "dashboard")
    st.session_state.setdefault("user_name", None)
    st.session_state.setdefault("auth_feedback", None)
    st.session_state.setdefault("active_result_path", None)
    st.session_state.setdefault("active_result_label", None)
    st.session_state.setdefault("active_history_id", None)
    st.session_state.setdefault("active_model_dir", None)
    st.session_state.setdefault("model_session_id", uuid4().hex[:12])
    st.session_state.setdefault("prediction_ready", False)
    st.session_state.setdefault("prediction_model_label", None)
    st.session_state.setdefault("show_live_prediction", False)


def open_auth_page(page_name):
    st.session_state["auth_page"] = page_name
    st.rerun()


def reset_prediction_state():
    st.session_state["app_page"] = "dashboard"
    st.session_state["app_page_selector"] = "dashboard"
    st.session_state["active_result_path"] = None
    st.session_state["active_result_label"] = None
    st.session_state["active_history_id"] = None
    st.session_state["active_model_dir"] = None
    st.session_state["model_session_id"] = uuid4().hex[:12]
    st.session_state["prediction_ready"] = False
    st.session_state["prediction_model_label"] = None
    st.session_state["show_live_prediction"] = False


def login_as_guest():
    st.session_state["auth_mode"] = "guest"
    st.session_state["user_name"] = "Guest"
    st.session_state["auth_feedback"] = None
    reset_prediction_state()
    st.rerun()


def login_as_user(username, password):
    cleaned_username = username.strip()

    if authenticate_user(cleaned_username, password):
        st.session_state["auth_mode"] = "user"
        st.session_state["user_name"] = cleaned_username
        st.session_state["auth_feedback"] = None
        reset_prediction_state()
        st.rerun()

    st.error("Invalid username or password.")


def sign_up_user(username, password, confirm_password):
    cleaned_username = username.strip()
    is_created, error_message = create_user(
        cleaned_username,
        password,
        confirm_password,
    )

    if not is_created:
        st.error(error_message)
        return

    st.session_state["auth_mode"] = "user"
    st.session_state["user_name"] = cleaned_username
    st.session_state["auth_feedback"] = "Account created successfully."
    reset_prediction_state()
    st.rerun()


def logout():
    st.session_state["auth_mode"] = None
    st.session_state["auth_page"] = "login"
    st.session_state["user_name"] = None
    st.session_state["auth_feedback"] = None
    reset_prediction_state()
    st.rerun()


def render_login_page():
    st.title("Energy Anomaly Detection")
    st.write("Sign in to continue, create an account, or enter as a guest.")

    login_col, guest_col = st.columns([3, 2])

    with login_col:
        st.subheader("Login")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_submitted = st.form_submit_button("Login")

        if login_submitted:
            login_as_user(username, password)

        st.caption("Don't have an account yet?")
        if st.button("Sign Up", key="open_signup"):
            open_auth_page("signup")

    with guest_col:
        st.subheader("Guest Access")
        st.write(
            "Guest mode skips account login and opens the same analysis workspace."
        )
        if st.button("Continue as Guest", key="guest_from_login", use_container_width=True):
            login_as_guest()

    # with st.expander("Account Storage", expanded=False):
    #     st.write(
    #         "User accounts are stored in the local SQLite database "
    #         "`data/auth_users.db`. Passwords are saved as salted hashes."
    #     )


def render_signup_page():
    st.title("Create Account")
    st.write("Sign up for a user account or continue into the app as a guest.")

    signup_col, guest_col = st.columns([3, 2])

    with signup_col:
        st.subheader("Sign Up")
        with st.form("signup_form"):
            username = st.text_input("Choose a username")
            password = st.text_input("Choose a password", type="password")
            confirm_password = st.text_input("Confirm password", type="password")
            signup_submitted = st.form_submit_button("Create Account")

        if signup_submitted:
            sign_up_user(username, password, confirm_password)

        if st.button("Back to Login", key="back_to_login"):
            open_auth_page("login")

    with guest_col:
        st.subheader("Guest Access")
        st.write(
            "You can skip account creation for now and still use the dashboard."
        )
        if st.button("Continue as Guest", key="guest_from_signup", use_container_width=True):
            login_as_guest()


def render_auth_page():
    feedback_message = st.session_state.pop("auth_feedback", None)
    if feedback_message:
        st.success(feedback_message)

    if st.session_state["auth_page"] == "signup":
        render_signup_page()
    else:
        render_login_page()


def sanitize_path_fragment(value):
    sanitized = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in value.strip()
    )
    return sanitized.strip("_") or "user"


def get_session_model_dir():
    owner_name = sanitize_path_fragment(st.session_state.get("user_name") or "guest")
    session_id = st.session_state.get("model_session_id") or "default"
    model_dir = MODEL_ARTIFACTS_ROOT / owner_name / session_id
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def create_model_run_dir():
    st.session_state["model_session_id"] = uuid4().hex[:12]
    return get_session_model_dir()


def get_active_model_dir():
    model_dir = st.session_state.get("active_model_dir")
    if not model_dir:
        return None

    return Path(model_dir)


def guess_column(columns, exact_names, keyword_names, exclude=None, use_fallback=True):
    exclude = set(exclude or [])
    available = [col for col in columns if col not in exclude]
    lowercase_map = {col.lower(): col for col in available}

    for name in exact_names:
        if name.lower() in lowercase_map:
            return lowercase_map[name.lower()]

    for col in available:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in keyword_names):
            return col

    if use_fallback and available:
        return available[0]

    return None


def load_uploaded_files(uploaded_files):
    dataframes = []

    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        dataframes.append(df)

    return pd.concat(dataframes, ignore_index=True, sort=False)


def infer_layout(df, timestamp_col, building_col, reading_col):
    if (
        building_col
        and reading_col
        and building_col != reading_col
        and timestamp_col not in {building_col, reading_col}
        and building_col in df.columns
        and reading_col in df.columns
    ):
        return "Long format"
    return "Wide format"


def build_download_frame(results_df):
    csv_data = results_df.to_csv(index=False)
    return csv_data.encode("utf-8")


def build_source_name(uploaded_files):
    if not uploaded_files:
        return "Uploaded dataset"

    file_names = [uploaded_file.name for uploaded_file in uploaded_files]
    if len(file_names) == 1:
        return file_names[0]

    return ", ".join(file_names[:3]) + ("..." if len(file_names) > 3 else "")


def prepare_uploaded_context(uploaded_files):
    raw_df = load_uploaded_files(uploaded_files)
    all_columns = list(raw_df.columns)

    timestamp_guess = guess_column(
        all_columns,
        exact_names=["timestamp", "datetime", "date"],
        keyword_names=["time", "date"],
    )
    building_guess = guess_column(
        all_columns,
        exact_names=["building", "building_id", "site"],
        keyword_names=["building", "site", "name"],
        use_fallback=False,
    )
    reading_guess = guess_column(
        all_columns,
        exact_names=["meter_reading", "reading", "value"],
        keyword_names=["meter", "reading", "value", "usage", "energy", "consumption"],
        use_fallback=False,
    )

    return raw_df, all_columns, timestamp_guess, building_guess, reading_guess


def render_upload_preview(raw_df, uploaded_files, heading):
    st.subheader(heading)

    preview_col1, preview_col2, preview_col3 = st.columns(3)
    preview_col1.metric("Uploaded rows", f"{len(raw_df):,}")
    preview_col2.metric("Uploaded columns", len(raw_df.columns))
    preview_col3.metric("Files", len(uploaded_files))

    st.dataframe(raw_df.head(20), use_container_width=True)


def render_dataset_configuration_form(
    raw_df,
    all_columns,
    timestamp_guess,
    building_guess,
    reading_guess,
    form_key,
    button_label,
):
    with st.form(form_key):
        layout_choice = st.radio(
            "Data layout",
            options=["Auto detect", "Long format", "Wide format"],
            horizontal=True,
        )

        timestamp_col = st.selectbox(
            "Timestamp column",
            options=all_columns,
            index=all_columns.index(timestamp_guess) if timestamp_guess in all_columns else 0,
        )

        inferred_layout = infer_layout(
            raw_df,
            timestamp_col=timestamp_col,
            building_col=building_guess or "building",
            reading_col=reading_guess or "meter_reading",
        )
        resolved_layout = (
            inferred_layout if layout_choice == "Auto detect" else layout_choice
        )

        building_col = "building"
        reading_col = "meter_reading"

        if resolved_layout == "Long format":
            remaining_for_building = [col for col in all_columns if col != timestamp_col]
            remaining_for_reading = [col for col in all_columns if col not in {timestamp_col}]

            if not remaining_for_building or not remaining_for_reading:
                st.error("Long-format uploads need building and reading columns.")
                return None

            building_default = (
                remaining_for_building.index(building_guess)
                if building_guess in remaining_for_building
                else 0
            )

            building_col = st.selectbox(
                "Building column",
                options=remaining_for_building,
                index=building_default,
            )
            reading_options = [col for col in remaining_for_reading if col != building_col]
            if not reading_options:
                st.error("Select a long-format file with a separate reading column.")
                return None
            reading_default_value = (
                reading_guess if reading_guess in reading_options else reading_options[0]
            )
            reading_col = st.selectbox(
                "Reading column",
                options=reading_options,
                index=reading_options.index(reading_default_value),
            )

        submitted = st.form_submit_button(button_label)

    if not submitted:
        return None

    return {
        "resolved_layout": resolved_layout,
        "timestamp_col": timestamp_col,
        "reading_col": reading_col,
        "building_col": building_col,
    }


def save_results_snapshot(results_df, owner_name):
    owner_dir = RESULTS_HISTORY_ROOT / sanitize_path_fragment(owner_name)
    owner_dir.mkdir(parents=True, exist_ok=True)

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    result_path = owner_dir / f"results_{timestamp}.csv"
    suffix = 1

    while result_path.exists():
        result_path = owner_dir / f"results_{timestamp}_{suffix}.csv"
        suffix += 1

    results_df.to_csv(result_path, index=False)
    return result_path


def load_results_snapshot(result_path):
    resolved_path = Path(result_path)
    results_df = pd.read_csv(resolved_path)

    if "timestamp" in results_df.columns:
        results_df["timestamp"] = pd.to_datetime(results_df["timestamp"], errors="coerce")

    for numeric_column in ["meter_reading", "final_anomaly", "month"]:
        if numeric_column in results_df.columns:
            results_df[numeric_column] = pd.to_numeric(
                results_df[numeric_column],
                errors="coerce",
            )

    return results_df


def build_history_label(row):
    return (
        f"#{row['id']} | {row['created_at']} | "
        f"{row['anomaly_count']} anomalies | {row['source_name']}"
    )


def set_active_result(
    result_path,
    result_label,
    history_id=None,
    model_dir=None,
    prediction_label=None,
):
    st.session_state["active_result_path"] = str(result_path) if result_path else None
    st.session_state["active_result_label"] = result_label
    st.session_state["active_history_id"] = history_id
    st.session_state["active_model_dir"] = str(model_dir) if model_dir else None
    st.session_state["prediction_ready"] = bool(model_dir)
    st.session_state["prediction_model_label"] = prediction_label if model_dir else None


def activate_history_run(row, open_prediction=False):
    result_label = f"Saved run from {row['created_at']} | {row['source_name']}"
    set_active_result(
        row["result_path"],
        result_label,
        history_id=row["id"],
        model_dir=row.get("model_dir"),
        prediction_label=row["source_name"],
    )
    st.session_state["show_live_prediction"] = open_prediction and bool(
        row.get("model_dir")
    )


def clear_active_history_if_selected(history_id):
    if st.session_state.get("active_history_id") != history_id:
        return

    st.session_state["active_result_path"] = None
    st.session_state["active_result_label"] = None
    st.session_state["active_history_id"] = None
    st.session_state["active_model_dir"] = None
    st.session_state["prediction_ready"] = False
    st.session_state["prediction_model_label"] = None
    st.session_state["show_live_prediction"] = False


def delete_history_entry(row):
    deleted_row = delete_run_history(row["id"], st.session_state["user_name"])
    if deleted_row is None:
        return False, "That saved run no longer exists."

    clear_active_history_if_selected(row["id"])

    result_path = Path(deleted_row["result_path"])
    try:
        result_path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        return True, f"History entry removed, but the saved CSV could not be deleted: {exc}"

    return True, "History entry deleted."


def render_history_sidebar():
    if st.session_state["auth_mode"] != "user":
        return

    history_rows = get_user_history(st.session_state["user_name"], limit=15)
    sidebar = st.sidebar
    sidebar.divider()
    st.session_state["app_page_selector"] = st.session_state.get(
        "app_page",
        "dashboard",
    )
    selected_page = sidebar.radio(
        "Page",
        options=["dashboard", "history"],
        format_func=lambda value: "Dashboard" if value == "dashboard" else "History",
        key="app_page_selector",
    )
    if selected_page != st.session_state.get("app_page"):
        st.session_state["app_page"] = selected_page
    sidebar.divider()
    sidebar.subheader("Run History")

    if not history_rows:
        sidebar.caption("No saved runs yet.")
        return

    history_lookup = {}
    history_labels = []
    for row in history_rows:
        label = build_history_label(row)
        history_lookup[label] = row
        history_labels.append(label)

    selected_label = sidebar.selectbox(
        "Recent runs",
        options=history_labels,
        key="history_selection",
    )

    selected_row = history_lookup[selected_label]
    sidebar.caption(
        f"Rows: {selected_row['processed_rows']:,} | "
        f"Buildings: {selected_row['building_count']:,}"
    )

    if sidebar.button("Open Selected Run", use_container_width=True):
        activate_history_run(selected_row, open_prediction=False)
        st.session_state["app_page"] = "dashboard"
        st.rerun()

    if sidebar.button(
        "Predict From Selected Run",
        use_container_width=True,
        disabled=not bool(selected_row.get("model_dir")),
    ):
        activate_history_run(selected_row, open_prediction=True)
        st.session_state["app_page"] = "dashboard"
        st.rerun()

    if not selected_row.get("model_dir"):
        sidebar.caption("Live prediction is available for newly saved runs.")


def render_plots(results_df):
    st.subheader("Visualizations")

    feature_col, distribution_col = st.columns(2)
    with feature_col:
        st.pyplot(plot_feature_importance(results_df, FEATURES, show=False))
    with distribution_col:
        st.pyplot(plot_anomaly_distribution(results_df, show=False))

    energy_col, monthly_col = st.columns(2)
    with energy_col:
        st.pyplot(plot_energy_anomalies(results_df, show=False))
    with monthly_col:
        st.pyplot(plot_monthly_anomalies(results_df, show=False))


def format_time_delta(delta):
    total_seconds = int(delta.total_seconds())
    if total_seconds < 0:
        return "Already due"

    total_hours = total_seconds // 3600
    days = total_hours // 24
    hours = total_hours % 24
    minutes = (total_seconds % 3600) // 60

    if days > 0:
        return f"{days}d {hours}h"
    if total_hours > 0:
        return f"{total_hours}h {minutes}m"
    return f"{minutes}m"


def render_next_anomaly_outlook(results_df):
    if "timestamp" not in results_df.columns or "final_anomaly" not in results_df.columns:
        return

    valid_window = results_df.dropna(subset=["timestamp"]).sort_values("timestamp")
    if valid_window.empty:
        return

    anomaly_rows = (
        results_df[results_df["final_anomaly"] == 1]
        .dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    st.subheader("Next Anomaly Outlook")

    window_start = valid_window["timestamp"].min()
    window_end = valid_window["timestamp"].max()

    if anomaly_rows.empty:
        st.success(
            "No anomaly is predicted inside the uploaded prediction window "
            f"({window_start:%Y-%m-%d %H:%M} to {window_end:%Y-%m-%d %H:%M})."
        )
        return

    next_row = anomaly_rows.iloc[0]
    lead_time = format_time_delta(next_row["timestamp"] - window_start)

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric(
        "Next predicted anomaly",
        next_row["timestamp"].strftime("%Y-%m-%d %H:%M"),
    )
    metric_col2.metric("Building", str(next_row["building"]))
    metric_col3.metric("Lead time from window start", lead_time)
    metric_col4.metric("Predicted anomalies in window", int(len(anomaly_rows)))

    next_by_building = (
        anomaly_rows.groupby("building", observed=False)
        .agg(
            next_predicted_anomaly=("timestamp", "min"),
            anomaly_count=("final_anomaly", "size"),
        )
        .reset_index()
        .sort_values("next_predicted_anomaly")
    )
    next_by_building["next_predicted_anomaly"] = next_by_building["next_predicted_anomaly"].dt.strftime(
        "%Y-%m-%d %H:%M"
    )

    st.caption("Earliest predicted anomaly per building")
    st.dataframe(next_by_building.head(10), use_container_width=True)


def render_results_panel(results_df, result_label, panel_key="results", show_prediction_outlook=False):
    st.subheader("Results")
    if result_label:
        st.caption(result_label)

    result_col1, result_col2, result_col3, result_col4 = st.columns(4)
    result_col1.metric("Processed rows", f"{len(results_df):,}")
    result_col2.metric("Buildings", results_df["building"].nunique())
    result_col3.metric("Anomalies", int(results_df["final_anomaly"].sum()))
    result_col4.metric(
        "Anomaly rate",
        f"{results_df['final_anomaly'].mean() * 100:.2f}%",
    )

    if show_prediction_outlook:
        render_next_anomaly_outlook(results_df)

    st.dataframe(results_df.head(200), use_container_width=True)
    render_plots(results_df)

    st.download_button(
        label="Download results as CSV",
        data=build_download_frame(results_df),
        file_name="energy_anomaly_results.csv",
        mime="text/csv",
        key=f"download_results_{panel_key}",
    )


def render_history_page():
    st.title("Run History")
    st.write(
        "Review saved anomaly detection runs, reopen their results, start live "
        "prediction from a saved model, or delete history entries you no longer need."
    )

    history_rows = get_user_history(st.session_state["user_name"], limit=100)
    if not history_rows:
        st.info("No saved history is available yet. Run anomaly detection first.")
        return

    history_lookup = {
        build_history_label(row): row
        for row in history_rows
    }
    selected_label = st.selectbox(
        "Saved runs",
        options=list(history_lookup.keys()),
        key="history_page_selection",
    )
    selected_row = history_lookup[selected_label]

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Processed rows", f"{selected_row['processed_rows']:,}")
    metric_col2.metric("Buildings", f"{selected_row['building_count']:,}")
    metric_col3.metric("Anomalies", f"{selected_row['anomaly_count']:,}")
    metric_col4.metric("Anomaly rate", f"{selected_row['anomaly_rate'] * 100:.2f}%")

    st.caption(f"Saved result file: {selected_row['result_path']}")
    if selected_row.get("model_dir"):
        st.caption(f"Saved model artifacts: {selected_row['model_dir']}")
    else:
        st.warning(
            "This history entry was created before live-prediction model storage "
            "was added, so it can be opened but not used for live prediction."
        )

    open_col, predict_col, delete_col = st.columns(3)
    if open_col.button("Open results", key=f"history_open_{selected_row['id']}", use_container_width=True):
        activate_history_run(selected_row, open_prediction=False)
        st.session_state["app_page"] = "dashboard"
        st.rerun()

    if predict_col.button(
        "Predict anomalies",
        key=f"history_predict_{selected_row['id']}",
        use_container_width=True,
        disabled=not bool(selected_row.get("model_dir")),
    ):
        activate_history_run(selected_row, open_prediction=True)
        st.session_state["app_page"] = "dashboard"
        st.rerun()

    confirm_delete = delete_col.checkbox(
        "Confirm delete",
        key=f"history_confirm_delete_{selected_row['id']}",
    )
    if delete_col.button(
        "Delete history",
        key=f"history_delete_{selected_row['id']}",
        use_container_width=True,
    ):
        if not confirm_delete:
            st.warning("Tick confirm delete before removing this history entry.")
        else:
            is_deleted, message = delete_history_entry(selected_row)
            if is_deleted:
                st.success(message)
                st.rerun()
            else:
                st.error(message)

    history_table = pd.DataFrame(history_rows)
    history_table["anomaly_rate"] = (history_table["anomaly_rate"] * 100).round(2)
    history_table["live_prediction_ready"] = history_table["model_dir"].notna()
    history_table = history_table[
        [
            "id",
            "created_at",
            "source_name",
            "processed_rows",
            "building_count",
            "anomaly_count",
            "anomaly_rate",
            "live_prediction_ready",
        ]
    ].rename(
        columns={
            "created_at": "saved_at",
            "source_name": "source",
            "processed_rows": "rows",
            "building_count": "buildings",
            "anomaly_count": "anomalies",
            "anomaly_rate": "anomaly_rate_pct",
        }
    )

    st.subheader("All Saved Runs")
    st.dataframe(history_table, use_container_width=True, hide_index=True)


def persist_logged_in_run(results_df, source_name, model_dir=None):
    result_path = save_results_snapshot(
        results_df,
        st.session_state["user_name"],
    )
    save_run_history(
        username=st.session_state["user_name"],
        source_name=source_name,
        result_path=result_path,
        model_dir=model_dir,
        processed_rows=len(results_df),
        building_count=results_df["building"].nunique(),
        anomaly_count=int(results_df["final_anomaly"].sum()),
        anomaly_rate=float(results_df["final_anomaly"].mean()),
    )
    return result_path


def render_live_prediction_section():
    if not st.session_state.get("show_live_prediction"):
        return None

    if run_live_prediction is None:
        st.error(
            "Live prediction mode is unavailable because the current pipeline "
            "module does not expose `run_live_prediction`."
        )
        return None

    model_dir = get_active_model_dir()
    if model_dir is None:
        st.error(
            "Live prediction is not ready yet because no trained model artifacts "
            "are attached to the current run."
        )
        return None

    st.subheader("Live Prediction Mode")
    st.caption(
        "Upload new data to score it with the trained anomaly models "
        f"from {st.session_state.get('prediction_model_label') or 'the selected run'}."
    )

    prediction_files = st.file_uploader(
        "Upload one or more CSV files for live prediction",
        type=["csv"],
        accept_multiple_files=True,
        key="prediction_uploader",
        help=(
            "Long format: timestamp, building, meter reading. "
            "Wide format: one timestamp column plus one column per building."
        ),
    )

    if not prediction_files:
        st.info("Upload new CSV data here to generate live anomaly predictions.")
        return None

    try:
        raw_df, all_columns, timestamp_guess, building_guess, reading_guess = (
            prepare_uploaded_context(prediction_files)
        )
    except Exception as exc:
        st.error(f"Could not read the prediction file(s): {exc}")
        return None

    render_upload_preview(raw_df, prediction_files, "Prediction Preview")
    config = render_dataset_configuration_form(
        raw_df,
        all_columns,
        timestamp_guess,
        building_guess,
        reading_guess,
        form_key="prediction_settings",
        button_label="Run live prediction",
    )

    if config is None:
        return None

    with st.spinner("Running live anomaly prediction..."):
        try:
            if config["resolved_layout"] == "Long format":
                results_df = run_live_prediction(
                    raw_df.copy(),
                    timestamp_col=config["timestamp_col"],
                    reading_col=config["reading_col"],
                    building_col=config["building_col"],
                    model_dir=model_dir,
                )
            else:
                results_df = run_live_prediction(
                    raw_df.copy(),
                    timestamp_col=config["timestamp_col"],
                    model_dir=model_dir,
                )
        except Exception as exc:
            st.error(f"Live prediction failed: {exc}")
            return None

    source_name = f"Prediction | {build_source_name(prediction_files)}"
    result_label = f"Live prediction from {build_source_name(prediction_files)}"

    if st.session_state["auth_mode"] == "user":
        persist_logged_in_run(results_df, source_name, model_dir=model_dir)

    return results_df, result_label


def render_dashboard():
    sidebar = st.sidebar
    sidebar.title("Session")
    sidebar.write(f"Signed in as: **{st.session_state['user_name']}**")
    sidebar.write(
        "Mode: **Guest**"
        if st.session_state["auth_mode"] == "guest"
        else "Mode: **User**"
    )
    if sidebar.button("Logout", use_container_width=True):
        logout()
    render_history_sidebar()

    if (
        st.session_state["auth_mode"] == "user"
        and st.session_state.get("app_page") == "history"
    ):
        render_history_page()
        return

    st.title("Energy Anomaly Detection")
    st.write(
        "Upload your building energy data, map the columns if needed, and run "
        "the anomaly detection pipeline from the browser."
    )

    active_results_df = None
    active_result_label = st.session_state.get("active_result_label")

    uploaded_files = st.file_uploader(
        "Upload one or more CSV files",
        type=["csv"],
        accept_multiple_files=True,
        help=(
            "Long format: timestamp, building, meter reading. "
            "Wide format: one timestamp column plus one column per building."
        ),
    )

    if uploaded_files:
        try:
            raw_df, all_columns, timestamp_guess, building_guess, reading_guess = (
                prepare_uploaded_context(uploaded_files)
            )
        except Exception as exc:
            st.error(f"Could not read the uploaded file(s): {exc}")
            raw_df = None

        if raw_df is not None:
            render_upload_preview(raw_df, uploaded_files, "Preview")
            config = render_dataset_configuration_form(
                raw_df,
                all_columns,
                timestamp_guess,
                building_guess,
                reading_guess,
                form_key="upload_settings",
                button_label="Run anomaly detection",
            )

            if config is not None:
                model_dir = create_model_run_dir()
                with st.spinner(
                    "Running preprocessing, feature engineering, anomaly detection, and plots..."
                ):
                    try:
                        if config["resolved_layout"] == "Long format":
                            results_df = run_pipeline(
                                raw_df.copy(),
                                timestamp_col=config["timestamp_col"],
                                reading_col=config["reading_col"],
                                building_col=config["building_col"],
                                model_dir=model_dir,
                            )
                        else:
                            results_df = run_pipeline(
                                raw_df.copy(),
                                timestamp_col=config["timestamp_col"],
                                model_dir=model_dir,
                            )
                    except Exception as exc:
                        st.error(f"Pipeline failed: {exc}")
                        results_df = None

                if results_df is not None:
                    source_name = build_source_name(uploaded_files)
                    active_result_label = f"Latest run from {source_name}"
                    st.session_state["show_live_prediction"] = False

                    if st.session_state["auth_mode"] == "user":
                        result_path = persist_logged_in_run(
                            results_df,
                            source_name,
                            model_dir=model_dir,
                        )
                        set_active_result(
                            result_path,
                            active_result_label,
                            model_dir=model_dir,
                            prediction_label=source_name,
                        )
                    else:
                        set_active_result(
                            None,
                            active_result_label,
                            model_dir=model_dir,
                            prediction_label=source_name,
                        )

                    active_results_df = results_df

    elif st.session_state["auth_mode"] == "user":
        st.info("Upload a CSV to run a new analysis, or open a saved run from the sidebar.")
    else:
        st.info("Upload at least one CSV file to get started.")

    if active_results_df is None and st.session_state["auth_mode"] == "user":
        active_result_path = st.session_state.get("active_result_path")
        if active_result_path:
            try:
                active_results_df = load_results_snapshot(active_result_path)
                active_result_label = st.session_state.get("active_result_label")
            except Exception as exc:
                st.warning(f"Could not load the selected saved run: {exc}")
                st.session_state["active_result_path"] = None
                st.session_state["active_result_label"] = None

    if active_results_df is not None:
        render_results_panel(active_results_df, active_result_label, panel_key="primary")

        if st.session_state.get("prediction_ready"):
            if st.button("Predict anomalies", key="open_live_prediction_mode"):
                st.session_state["show_live_prediction"] = True
        elif st.session_state["auth_mode"] == "user":
            st.caption(
                "This saved run can be viewed, but it does not include stored model "
                "artifacts for live prediction."
            )

        prediction_result = render_live_prediction_section()
        if prediction_result is not None:
            prediction_results_df, prediction_label = prediction_result
            render_results_panel(
                prediction_results_df,
                prediction_label,
                panel_key="prediction",
                show_prediction_outlook=True,
            )


init_auth_db()
init_session_state()

if st.session_state["auth_mode"] is None:
    render_auth_page()
else:
    render_dashboard()
