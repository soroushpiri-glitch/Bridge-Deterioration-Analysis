import re
import os
import boto3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import io
import contextlib
import traceback

from botocore.exceptions import BotoCoreError, ClientError
from sklearn.cluster import KMeans
from scipy.stats import linregress
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBRegressor = None
    XGBOOST_AVAILABLE = False

# ---------------------------
# Config
# ---------------------------
N_CLUSTERS = 6
STATIC_FILE = "STEEL_Bridges.csv"

# Project-style forecasting constants
EMPIRICAL_DETERIORATION_RATE = 0.214
ADT_GROWTH_RATE = 0.0003
TEMP_INCREASE_RATE = 0.0015
CRITICAL_BHI_THRESHOLD = 60

# ---------------------------
# AWS config
# ---------------------------
AWS_REGION = st.secrets.get("AWS_REGION", "us-east-2")
BEDROCK_MODEL_ID = st.secrets.get("BEDROCK_MODEL_ID", "us.amazon.nova-lite-v1:0")
AWS_ACCESS_KEY_ID = st.secrets.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = st.secrets.get("AWS_SECRET_ACCESS_KEY")

if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
    st.error("AWS secrets are missing in Streamlit app settings.")
    st.stop()

try:
    bedrock = boto3.client(
        "bedrock-runtime",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
except Exception as e:
    st.error(f"Could not create Bedrock client: {e}")
    st.stop()

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(
    page_title="Bridge Deterioration Analysis",
    layout="wide"
)

st.title("Bridge Deterioration Analysis")
st.caption("Amazon Bedrock + Streamlit + Bridge Health Index time-series clustering")

# ---------------------------
# File reader
# ---------------------------
def read_table_file(file_path: str) -> pd.DataFrame:
    ext = os.path.splitext(file_path)[1].lower()

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if ext == ".csv":
        return pd.read_csv(file_path, sep=",", header=0, low_memory=False)

    if ext == ".xlsx":
        return pd.read_excel(file_path, engine="openpyxl")

    if ext == ".xls":
        try:
            return pd.read_excel(file_path, engine="xlrd")
        except Exception as xls_err:
            try:
                return pd.read_excel(file_path, engine="openpyxl")
            except Exception:
                try:
                    return pd.read_csv(file_path, sep=",", header=0, low_memory=False)
                except Exception:
                    try:
                        return pd.read_csv(file_path, sep="\t", header=0, low_memory=False)
                    except Exception:
                        raise ValueError(
                            f"Could not read '{file_path}'. "
                            f"It may be mislabeled, corrupted, or saved in another format. "
                            f"Original xlrd error: {xls_err}"
                        ) from xls_err

    raise ValueError(f"Unsupported file type: {ext}")

# ---------------------------
# Load raw data only
# ---------------------------
@st.cache_data
def load_data():
    if not os.path.exists(STATIC_FILE):
        st.error(f"Missing required file: {STATIC_FILE}")
        st.stop()

    try:
        static_df = read_table_file(STATIC_FILE)
    except Exception as e:
        st.error(f"Failed to read static file '{STATIC_FILE}': {e}")
        st.stop()

    static_df.columns = static_df.columns.str.strip()

    required_cols = [
        "Year of Data",
        "STRUCTURE_NUMBER_008",
        "Bridge Health Index (Overall)",
        "Bridge Health Index (Deck)",
        "Bridge Health Index (Super)",
        "Bridge Health Index (Sub)"
    ]
    missing_cols = [c for c in required_cols if c not in static_df.columns]

    if missing_cols:
        st.error(f"Static file is missing required columns: {missing_cols}")
        st.stop()

    return static_df

# ---------------------------
# Prepare analysis
# ---------------------------
@st.cache_data
def prepare_analysis(static_df, n_clusters=N_CLUSTERS):
    data = static_df.copy()
    data = data.dropna()

    df_full = data[[
        "Year of Data",
        "STRUCTURE_NUMBER_008",
        "Bridge Health Index (Overall)",
        "Bridge Health Index (Deck)",
        "Bridge Health Index (Super)",
        "Bridge Health Index (Sub)"
    ]].copy()

    counts = df_full["STRUCTURE_NUMBER_008"].value_counts()
    structure_ids_20 = counts[(counts >= 20)].index
    df_filtered = df_full[df_full["STRUCTURE_NUMBER_008"].isin(structure_ids_20)].copy()

    df_check = df_filtered[[
        "STRUCTURE_NUMBER_008",
        "Year of Data",
        "Bridge Health Index (Overall)"
    ]].dropna().copy()

    df_check = df_check.sort_values(["STRUCTURE_NUMBER_008", "Year of Data"])
    df_check["Year of Data"] = df_check["Year of Data"].astype(int)

    unchanged_bridges = []

    for bridge_id, group in df_check.groupby("STRUCTURE_NUMBER_008"):
        group = group.sort_values("Year of Data")
        years = group["Year of Data"].tolist()
        values = group["Bridge Health Index (Overall)"].tolist()

        for i in range(len(values) - 20):
            if all(v == values[i] for v in values[i:i + 20]) and \
               all(y2 - y1 == 1 for y1, y2 in zip(years[i:i + 19], years[i + 1:i + 20])):
                unchanged_bridges.append(bridge_id)
                break

    result_df = df_check[df_check["STRUCTURE_NUMBER_008"].isin(unchanged_bridges)].copy()

    unique_structures = result_df["STRUCTURE_NUMBER_008"].unique()
    df_filtered_cleaned = df_filtered[
        ~df_filtered["STRUCTURE_NUMBER_008"].isin(unique_structures)
    ].copy()

    ts_df = df_filtered_cleaned.copy()

    df_cluster = ts_df[[
        "STRUCTURE_NUMBER_008",
        "Year of Data",
        "Bridge Health Index (Overall)"
    ]].dropna()

    pivot_df = df_cluster.pivot(
        index="STRUCTURE_NUMBER_008",
        columns="Year of Data",
        values="Bridge Health Index (Overall)"
    )

    pivot_df = pivot_df.sort_index(axis=1).interpolate(
        axis=1,
        limit_direction="both"
    ).ffill(axis=1).bfill(axis=1)

    filtered_df = pivot_df.copy()

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(filtered_df)

    clustered = pivot_df.copy()
    clustered["Cluster"] = cluster_labels

    slopes = []
    for bridge_id, row in pivot_df.iterrows():
        y = row.values.astype(float)
        x = np.array(row.index.tolist(), dtype=float)

        if len(x) >= 2 and np.isfinite(y).sum() >= 2:
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
        else:
            slope = np.nan

        slopes.append({
            "STRUCTURE_NUMBER_008": bridge_id,
            "deterioration_slope_per_year": slope
        })

    slope_df = pd.DataFrame(slopes)

    cluster_df = clustered[["Cluster"]].reset_index()
    bridge_summary = cluster_df.merge(slope_df, on="STRUCTURE_NUMBER_008", how="left")

    latest_static = (
        static_df.sort_values(["STRUCTURE_NUMBER_008", "Year of Data"])
        .groupby("STRUCTURE_NUMBER_008", as_index=False)
        .last()
    )

    bridge_summary = bridge_summary.merge(
        latest_static,
        on="STRUCTURE_NUMBER_008",
        how="left"
    )

    years = [c for c in pivot_df.columns.tolist() if isinstance(c, (int, np.integer))]
    cluster_sizes = bridge_summary["Cluster"].value_counts().sort_index()

    preprocessing_summary = {
        "raw_rows_after_dropna": int(len(data)),
        "bridges_with_20plus_records": int(len(structure_ids_20)),
        "constant_20year_bridges_removed": len(set(unchanged_bridges)),
        "final_rows_used": int(len(ts_df)),
        "final_unique_bridges": int(ts_df["STRUCTURE_NUMBER_008"].nunique())
    }

    return {
        "ts_df": ts_df,
        "pivot_df": pivot_df,
        "clustered_df": clustered,
        "bridge_summary": bridge_summary,
        "years": years,
        "cluster_sizes": cluster_sizes,
        "kmeans": kmeans,
        "preprocessing_summary": preprocessing_summary
    }

# ---------------------------
# Forecast methodology
# ---------------------------
def _find_optional_column(df, candidates=None, contains=None):
    cols = list(df.columns)
    if candidates:
        lowered = {c.lower(): c for c in cols}
        for cand in candidates:
            if cand.lower() in lowered:
                return lowered[cand.lower()]
    if contains:
        for col in cols:
            col_low = col.lower()
            if all(term.lower() in col_low for term in contains):
                return col
    return None


def _prepare_bridge_forecast_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data.columns = data.columns.str.strip()

    required_cols = [
        "Year of Data",
        "STRUCTURE_NUMBER_008",
        "Bridge Health Index (Overall)"
    ]
    missing_cols = [c for c in required_cols if c not in data.columns]
    if missing_cols:
        raise ValueError(f"Forecast data is missing required columns: {missing_cols}")

    data["Year of Data"] = pd.to_numeric(data["Year of Data"], errors="coerce")
    data["Bridge Health Index (Overall)"] = pd.to_numeric(
        data["Bridge Health Index (Overall)"], errors="coerce"
    )
    data = data.dropna(subset=required_cols).copy()
    data["Year of Data"] = data["Year of Data"].astype(int)
    data["STRUCTURE_NUMBER_008"] = data["STRUCTURE_NUMBER_008"].astype(str).str.strip()

    year_built_col = _find_optional_column(data, candidates=["YEAR_BUILT_027", "Year Built"])
    recon_col = _find_optional_column(
        data,
        candidates=["YEAR_OF_LAST_RECONSTRUCTION", "LAST_RECONSTRUCTION_YEAR"],
        contains=["reconstruction", "year"]
    )
    if recon_col is None:
        recon_col = _find_optional_column(data, contains=["improvement", "year"])
    temp_col = _find_optional_column(
        data,
        candidates=["Approx_Avg_Temp", "AVG_TEMP", "MEAN_TEMP"],
        contains=["temp"]
    )

    for col in [year_built_col, recon_col, temp_col]:
        if col is not None:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.sort_values(["STRUCTURE_NUMBER_008", "Year of Data"]).copy()

    data["BHI_t_minus_1"] = data.groupby("STRUCTURE_NUMBER_008")["Bridge Health Index (Overall)"].shift(1)
    data["BHI_t_minus_2"] = data.groupby("STRUCTURE_NUMBER_008")["Bridge Health Index (Overall)"].shift(2)
    data["BHI_change_1yr"] = data["BHI_t_minus_1"] - data["BHI_t_minus_2"]

    first_year = data.groupby("STRUCTURE_NUMBER_008")["Year of Data"].transform("min")
    data["Years_Since_First_Observation"] = data["Year of Data"] - first_year

    if year_built_col is not None:
        data["Bridge_Age"] = data["Year of Data"] - data[year_built_col]
        data.loc[(data["Bridge_Age"] < 0) | (data["Bridge_Age"] > 300), "Bridge_Age"] = np.nan
    else:
        data["Bridge_Age"] = data["Years_Since_First_Observation"]

    if recon_col is not None:
        data["Time_Since_Last_Reconstruction"] = data["Year of Data"] - data[recon_col]
        data.loc[(data["Time_Since_Last_Reconstruction"] < 0) | (data["Time_Since_Last_Reconstruction"] > 300), "Time_Since_Last_Reconstruction"] = np.nan
    else:
        data["Time_Since_Last_Reconstruction"] = data["Years_Since_First_Observation"]

    if temp_col is not None:
        data["Approx_Avg_Temp"] = data[temp_col]
    else:
        data["Approx_Avg_Temp"] = 0.0

    for col in ["Bridge_Age", "Time_Since_Last_Reconstruction", "Approx_Avg_Temp"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        data[col] = data.groupby("STRUCTURE_NUMBER_008")[col].transform(lambda s: s.ffill().bfill())
        data[col] = data[col].fillna(0)

    return data


@st.cache_data
def prepare_forecast_data(static_df: pd.DataFrame):
    data = _prepare_bridge_forecast_dataframe(static_df)

    feature_cols = [
        "Year of Data",
        "BHI_t_minus_1",
        "BHI_t_minus_2",
        "BHI_change_1yr",
        "Years_Since_First_Observation",
        "Bridge_Age",
        "Time_Since_Last_Reconstruction",
        "Approx_Avg_Temp",
    ]

    model_df = data.dropna(subset=feature_cols + ["Bridge Health Index (Overall)"]).copy()
    target_col = "Bridge Health Index (Overall)"
    return model_df, feature_cols, target_col


@st.cache_resource
def train_forecast_model(static_df: pd.DataFrame):
    model_df, feature_cols, target_col = prepare_forecast_data(static_df)

    if model_df.empty:
        raise ValueError("No valid rows available to train the forecast model.")

    X = model_df[feature_cols].copy()
    y = model_df[target_col].copy()

    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    gb = GradientBoostingRegressor(random_state=42)

    estimators = [("rf", rf), ("gb", gb)]
    if XGBOOST_AVAILABLE:
        xgb = XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            verbosity=0
        )
        estimators.append(("xgb", xgb))

    model = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0)
    )
    model.fit(X, y)

    preds = model.predict(X)
    residuals = y - preds
    mae = mean_absolute_error(y, preds)
    residual_std = float(np.std(residuals))

    return {
        "model": model,
        "feature_cols": feature_cols,
        "training_mae": float(mae),
        "residual_std": residual_std,
        "training_rows": int(len(model_df)),
        "used_xgboost": bool(XGBOOST_AVAILABLE)
    }


def is_forecast_question(question: str):
    q = question.lower().strip()
    phrases = [
        "forecast",
        "predict",
        "projection",
        "20 year projection",
        "20-year projection",
        "20 year forecast",
        "20-year forecast"
    ]
    return any(p in q for p in phrases)



def extract_bridge_id_from_question(question: str):
    candidates = re.findall(r"[A-Za-z0-9\-]+", question)
    for token in candidates:
        matched = find_best_bridge_match(token)
        if matched is not None:
            return matched
    return None


def is_forecast_explanation_question(question: str):
    q = question.lower().strip()
    phrases = [
        "how is this projection executed",
        "how was this projection executed",
        "how is this forecast executed",
        "how was this forecast executed",
        "how did you do this forecast",
        "how did you calculate this forecast",
        "how did you calculate this projection",
        "explain this forecast",
        "explain this projection",
        "what methodology did you use",
        "how was this projection calculated",
        "how was this forecast calculated"
    ]
    return any(p in q for p in phrases)


def build_forecast_execution_explanation(
    bridge_id: str,
    bridge_hist: pd.DataFrame,
    trained: dict,
    latest_cluster,
    forecast_horizon: int = 20
):
    model_name = "StackingRegressor (RF + GB + XGB)" if trained.get("used_xgboost") else "StackingRegressor (RF + GB)"
    feature_names = trained.get("feature_cols", [])
    feature_text = ", ".join(feature_names) if feature_names else "lag-based and temporal features"

    steps = [
        f"Matched the user request to bridge {bridge_id}.",
        f"Loaded {len(bridge_hist)} historical yearly records for this bridge and sorted them by year.",
        f"Prepared forecasting inputs using these features: {feature_text}.",
        f"Used the trained forecasting model: {model_name}.",
        "Projected the bridge year by year recursively, meaning each predicted BHI value was fed into the next forecast step.",
        f"Applied the project-style deterioration overlay using an empirical deterioration rate of {EMPIRICAL_DETERIORATION_RATE:.3f} BHI/year.",
        f"Updated temperature during projection using the annual increase assumption of {TEMP_INCREASE_RATE:.4f} per year when temperature data was available.",
        "Computed a residual-based 95% prediction interval using model residual spread.",
        "Computed an empirical uncertainty band that increases with forecast horizon.",
        f"Plotted the projected BHI, both uncertainty bands, and the critical threshold at BHI = {CRITICAL_BHI_THRESHOLD}."
    ]

    explanation_text = f"""Projection execution for bridge {bridge_id}:

- Historical records used: {len(bridge_hist)}
- Forecast horizon: {forecast_horizon} years
- Model used: {model_name}
- Features used: {feature_text}
- Training MAE: {trained.get('training_mae', float('nan')):.2f}
- Residual standard deviation: {trained.get('residual_std', float('nan')):.3f}
- Cluster: {int(latest_cluster) if pd.notna(latest_cluster) else 'N/A'}
- Empirical deterioration rate: {EMPIRICAL_DETERIORATION_RATE:.3f} BHI/year
- Temperature increase assumption: {TEMP_INCREASE_RATE:.4f} per year
- Critical threshold: BHI = {CRITICAL_BHI_THRESHOLD}

This forecast was executed using a recursive ML-based projection pipeline combined with your project-style deterioration and uncertainty overlay."""

    return {
        "text": explanation_text,
        "execution_steps": steps
    }


def forecast_bridge_20_years(bridge_id: str, forecast_horizon: int = 20):
    matched = find_best_bridge_match(bridge_id)
    if matched is None:
        return {"text": f"I could not find a matching bridge for '{bridge_id}'."}

    prepared = _prepare_bridge_forecast_dataframe(static_df)
    bridge_hist = prepared[prepared["STRUCTURE_NUMBER_008"] == str(matched)].copy()
    bridge_hist = bridge_hist.sort_values("Year of Data")

    if len(bridge_hist) < 5:
        return {
            "text": f"Bridge {matched} does not have enough history for a 20-year forecast. At least 5 yearly observations are needed."
        }

    trained = train_forecast_model(static_df)
    model = trained["model"]
    residual_std = trained["residual_std"]
    training_mae = trained["training_mae"]

    feature_cols = trained["feature_cols"]
    last_row = bridge_hist.iloc[-1].copy()
    last_year = int(last_row["Year of Data"])
    last_bhi = float(last_row["Bridge Health Index (Overall)"])

    future_rows = []

    for step in range(1, forecast_horizon + 1):
        forecast_year = last_year + step
        next_row = last_row.copy()
        next_row["Year of Data"] = forecast_year
        next_row["Years_Since_First_Observation"] = float(last_row["Years_Since_First_Observation"]) + 1
        next_row["Bridge_Age"] = float(last_row["Bridge_Age"]) + 1
        next_row["Time_Since_Last_Reconstruction"] = float(last_row["Time_Since_Last_Reconstruction"]) + 1
        next_row["BHI_t_minus_2"] = float(last_row["BHI_t_minus_1"])
        next_row["BHI_t_minus_1"] = float(last_row["Bridge Health Index (Overall)"])
        next_row["BHI_change_1yr"] = next_row["BHI_t_minus_1"] - next_row["BHI_t_minus_2"]
        if "Approx_Avg_Temp" in next_row.index:
            next_row["Approx_Avg_Temp"] = float(last_row["Approx_Avg_Temp"]) + TEMP_INCREASE_RATE

        X_next = pd.DataFrame([{col: next_row[col] for col in feature_cols}])
        model_pred = float(model.predict(X_next)[0])
        model_pred = max(0.0, min(100.0, model_pred))

        # Project-style deterioration overlay
        deteriorated_pred = max(0.0, min(100.0, model_pred - EMPIRICAL_DETERIORATION_RATE * (step ** 0.9)))

        empirical_uncertainty = min(
    residual_std * np.sqrt(step),
    10 + 0.2 * step
)
        lower_empirical = max(0.0, deteriorated_pred - 1.96 * empirical_uncertainty)
        upper_empirical = min(100.0, deteriorated_pred + 1.96 * empirical_uncertainty)

        residual_pi = 1.96 * residual_std
        lower_pi = max(0.0, deteriorated_pred - residual_pi)
        upper_pi = min(100.0, deteriorated_pred + residual_pi)

        future_rows.append({
            "STRUCTURE_NUMBER_008": str(matched),
            "Forecast Year": forecast_year,
            "Model Predicted BHI": round(model_pred, 2),
            "Predicted BHI": round(deteriorated_pred, 2),
            "Empirical Uncertainty": round(empirical_uncertainty, 3),
            "Lower 95% PI": round(lower_pi, 2),
            "Upper 95% PI": round(upper_pi, 2),
            "Lower Empirical": round(lower_empirical, 2),
            "Upper Empirical": round(upper_empirical, 2),
            "Critical Threshold": CRITICAL_BHI_THRESHOLD,
        })

        next_row["Bridge Health Index (Overall)"] = deteriorated_pred
        last_row = next_row

    forecast_df = pd.DataFrame(future_rows)

    latest_cluster = None
    bs = bridge_summary[bridge_summary["STRUCTURE_NUMBER_008"].astype(str) == str(matched)]
    if not bs.empty and "Cluster" in bs.columns:
        latest_cluster = bs["Cluster"].iloc[0]

    model_name = "StackingRegressor (RF + GB + XGB)" if trained["used_xgboost"] else "StackingRegressor (RF + GB)"
    execution_info = build_forecast_execution_explanation(
        bridge_id=str(matched),
        bridge_hist=bridge_hist,
        trained=trained,
        latest_cluster=latest_cluster,
        forecast_horizon=forecast_horizon
    )
    summary_text = (
        f"20-year forecast for bridge {matched}:\n\n"
        f"- Historical records used: {len(bridge_hist)}\n"
        f"- Last observed year: {last_year}\n"
        f"- Last observed overall BHI: {last_bhi:.2f}\n"
        f"- Forecast methodology: {model_name} + empirical deterioration overlay\n"
        f"- Empirical deterioration rate: {EMPIRICAL_DETERIORATION_RATE:.3f} BHI/year\n"
        f"- Training MAE: {training_mae:.2f}\n"
        f"- Residual std: {residual_std:.3f}\n"
        f"- Cluster: {int(latest_cluster) if pd.notna(latest_cluster) else 'N/A'}\n\n"
        f"The table and plot show project-style projected BHI values for the next {forecast_horizon} years, including a residual 95% prediction interval, an empirical uncertainty band, and the critical threshold at BHI = {CRITICAL_BHI_THRESHOLD}."
    )

    return {
        "text": summary_text,
        "analysis_df": forecast_df,
        "bridge_ids": [str(matched)],
        "label": "bridge_20yr_forecast",
        "figure": make_bridge_forecast_figure(str(matched), forecast_df),
        "forecast_explanation": execution_info["text"],
        "execution_steps": execution_info["execution_steps"]
    }


def make_bridge_forecast_figure(bridge_id: str, forecast_df: pd.DataFrame):
    matched = find_best_bridge_match(bridge_id)
    if matched is None or forecast_df is None or forecast_df.empty:
        return None

    # Project-style plot: forecast window only
    fig, ax = plt.subplots(figsize=(10, 5))

    years = forecast_df["Forecast Year"].astype(int).values
    pred = forecast_df["Predicted BHI"].values
    low_pi = forecast_df["Lower 95% PI"].values
    up_pi = forecast_df["Upper 95% PI"].values
    low_emp = forecast_df["Lower Empirical"].values
    up_emp = forecast_df["Upper Empirical"].values

    ax.plot(years, pred, color="blue", linewidth=2, label="Predicted BHI (Deteriorated)")
    ax.fill_between(years, low_pi, up_pi, color="lightgreen", alpha=0.6, label="95% PI (Residual)")
    ax.fill_between(years, low_emp, up_emp, color="gold", alpha=0.35, label="Uncertainty Interval (Empirical)")
    ax.axhline(CRITICAL_BHI_THRESHOLD, color="red", linestyle="--", linewidth=1.8, label=f"Critical Threshold (BHI={CRITICAL_BHI_THRESHOLD})")

    import matplotlib.ticker as mticker
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    ax.text(years[-1] - 1.3, pred[-1] + 0.2, f"BHI={pred[-1]:.1f}", color="navy", fontsize=9)

    ax.set_title(f"Steel Bridge {matched} - 20-Year Forecast with Deterioration")
    ax.set_xlabel("Year")
    ax.set_ylabel("Predicted BHI")
    ax.set_ylim(max(58, np.floor(min(low_emp.min(), low_pi.min(), pred.min()) - 2)), 102)
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(loc="lower left")
    return fig


# ---------------------------
# Run pipeline
# ---------------------------
static_df = load_data()
analysis = prepare_analysis(static_df, n_clusters=N_CLUSTERS)

ts_df = analysis["ts_df"]
pivot_df = analysis["pivot_df"]
clustered_df = analysis["clustered_df"]
bridge_summary = analysis["bridge_summary"]
years_available = analysis["years"]
cluster_sizes = analysis["cluster_sizes"]
preprocessing_summary = analysis["preprocessing_summary"]

bridge_ids = sorted(pivot_df.index.astype(str).tolist())

with st.spinner("Training forecast model at startup for faster forecast responses..."):
    forecast_artifacts = train_forecast_model(static_df)

# ---------------------------
# Session state init
# ---------------------------
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Ask me about bridge deterioration trends, bridge profiles, clusters, Bridge Health Index patterns, or inspect and browse the dataset."
            }
        ]

    if "pending_compare_cluster" not in st.session_state:
        st.session_state.pending_compare_cluster = None

    if "last_result_context" not in st.session_state:
        st.session_state.last_result_context = {
            "bridge_ids": None,
            "cluster_ids": None,
            "year": None,
            "label": None,
            "result_type": None,
            "question": None
        }

    if "last_forecast_result" not in st.session_state:
        st.session_state.last_forecast_result = None

initialize_session_state()

# ---------------------------
# Helpers
# ---------------------------
def find_best_bridge_match(bridge_id: str):
    if not bridge_id:
        return None

    candidate = str(bridge_id).strip()
    bridge_ids_str = [str(b).strip() for b in bridge_ids]

    exact_matches = [b for b in bridge_ids_str if b == candidate]
    if exact_matches:
        return exact_matches[0]

    contains_matches = [b for b in bridge_ids_str if candidate in b]
    if contains_matches:
        return contains_matches[0]

    reverse_contains = [b for b in bridge_ids_str if b in candidate]
    if reverse_contains:
        return reverse_contains[0]

    return None


def extract_top_n(user_query, default=5):
    patterns = [
        r"top\s+(\d+)",
        r"show\s+the\s+(\d+)",
        r"show\s+me\s+the\s+(\d+)",
        r"\b(\d+)\s+worst\b",
        r"\b(\d+)\s+best\b"
    ]
    q = user_query.lower()
    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            return int(match.group(1))
    return default


def extract_single_cluster_id(text: str):
    matches = re.findall(r"cluster\s+(\d+)", text.lower())
    if matches:
        return int(matches[0])
    return None


def extract_compare_target(text: str):
    text = text.lower().strip()

    m = re.search(r"(?:compare\s+(?:to|with)\s+)?cluster\s+(\d+)", text)
    if m:
        return int(m.group(1))

    m = re.fullmatch(r"\s*(\d+)\s*", text)
    if m:
        return int(m.group(1))

    return None


def strip_thinking_blocks(text: str):
    if not text:
        return text
    return re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def clean_year_built(series):
    s = pd.to_numeric(series, errors="coerce")
    s[(s < 1800) | (s > 2100)] = np.nan
    return s


def make_json_safe(obj):
    if obj is None:
        return None

    if isinstance(obj, pd.DataFrame):
        return [make_json_safe(record) for record in obj.to_dict(orient="records")]

    if isinstance(obj, pd.Series):
        return make_json_safe(obj.to_dict())

    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(v) for v in obj]

    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)

    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()

    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj

    if isinstance(obj, (str, int, bool)):
        return obj

    return str(obj)


def render_paginated_dataframe(df, key_prefix="data_viewer", title="Data Explorer"):
    if df is None or df.empty:
        st.info("No data to display.")
        return

    st.subheader(title)

    total_rows = len(df)

    page_key = f"{key_prefix}_page"
    page_size_key = f"{key_prefix}_page_size"
    cols_key = f"{key_prefix}_columns"

    if page_key not in st.session_state:
        st.session_state[page_key] = 0
    if page_size_key not in st.session_state:
        st.session_state[page_size_key] = 25
    if cols_key not in st.session_state:
        st.session_state[cols_key] = list(df.columns)

    control_col1, control_col2 = st.columns([1, 2])

    with control_col1:
        page_size = st.selectbox(
            "Rows per page",
            options=[10, 25, 50, 100],
            index=[10, 25, 50, 100].index(st.session_state[page_size_key]) if st.session_state[page_size_key] in [10, 25, 50, 100] else 1,
            key=page_size_key
        )

    with control_col2:
        selected_cols = st.multiselect(
            "Columns to display",
            options=list(df.columns),
            default=st.session_state[cols_key],
            key=cols_key
        )

    if not selected_cols:
        st.warning("Select at least one column.")
        return

    total_pages = max(1, int(np.ceil(total_rows / page_size)))

    if st.session_state[page_key] >= total_pages:
        st.session_state[page_key] = total_pages - 1
    if st.session_state[page_key] < 0:
        st.session_state[page_key] = 0

    nav1, nav2, nav3 = st.columns([1, 1, 3])

    with nav1:
        if st.button("⬅ Previous", key=f"{key_prefix}_prev", disabled=(st.session_state[page_key] == 0)):
            st.session_state[page_key] -= 1

    with nav2:
        if st.button("Next ➡", key=f"{key_prefix}_next", disabled=(st.session_state[page_key] >= total_pages - 1)):
            st.session_state[page_key] += 1

    with nav3:
        st.write(f"Page {st.session_state[page_key] + 1} of {total_pages} | Rows: {total_rows:,}")

    start_idx = st.session_state[page_key] * page_size
    end_idx = min(start_idx + page_size, total_rows)

    page_df = df.iloc[start_idx:end_idx][selected_cols]

    st.dataframe(
        page_df,
        use_container_width=True,
        height=500
    )

# ---------------------------
# Conversational follow-up logic
# ---------------------------
def has_bridge_context():
    ctx = st.session_state.last_result_context
    return bool(ctx.get("bridge_ids"))

def has_cluster_context():
    ctx = st.session_state.last_result_context
    return bool(ctx.get("cluster_ids"))


def is_contextual_followup(question: str):
    q = question.lower().strip()

    explicit_reference_phrases = [
        "these bridges",
        "those bridges",
        "these 5 bridges",
        "those 5 bridges",
        "these five bridges",
        "those five bridges",
        "them",
        "their trend",
        "their profile",
        "their profiles",
        "these ones",
        "those ones",
        "which one",
        "which bridge",
        "among them",
        "of these",
        "of those"
    ]

    analytical_followup_phrases = [
        "deteriorated the fastest",
        "deteriorated fastest",
        "improved the most",
        "improved most",
        "average bhi",
        "average overall bhi",
        "average of these bridges",
        "average of them",
        "trend for these",
        "show trend",
        "plot trend",
        "compare them",
        "compare these bridges",
        "rank them",
        "which is worst",
        "which is best",
        "fastest",
        "slowest"
    ]

    return (
        any(p in q for p in explicit_reference_phrases) or
        any(p in q for p in analytical_followup_phrases)
    )


def is_cluster_followup(question: str):
    q = question.lower().strip()

    phrases = [
        "this cluster",
        "that cluster",
        "the cluster",
        "it",
        "what else can you do",
        "what else can you analyze",
        "more analysis",
        "interesting analysis",
        "deeper analysis",
        "analyze it further",
        "tell me more",
        "what else about it",
        "what else for this cluster",
        "median bhi line",
        "median line",
        "interpret the median",
        "interpret this plot",
        "interpret this graph",
        "fluctuations in bhi",
        "fluctuations over time",
        "variation over time",
        "what do the fluctuations mean"
    ]

    return any(p in q for p in phrases)


def resolve_cluster_followup_intent(question: str):
    q = question.lower().strip()

    if any(p in q for p in [
        "median bhi line",
        "median line",
        "interpret the median",
        "interpret this plot",
        "interpret this graph"
    ]):
        return "cluster_median_interpretation"

    if any(p in q for p in [
        "fluctuations in bhi",
        "fluctuations over time",
        "variation over time",
        "what do the fluctuations mean"
    ]):
        return "cluster_fluctuation_interpretation"

    if any(p in q for p in [
        "interesting analysis",
        "deeper analysis",
        "what else can you do",
        "what else can you analyze",
        "more analysis",
        "tell me more",
        "what else for this cluster"
    ]):
        return "cluster_deep_dive"

    if any(p in q for p in [
        "key drivers",
        "important features",
        "what characterizes",
        "pca",
        "pc1",
        "loadings"
    ]):
        return "cluster_pca"

    if any(p in q for p in [
        "trend",
        "median trend",
        "show trend",
        "plot trend"
    ]):
        return "cluster_trend"

    if any(p in q for p in [
        "summary",
        "summarize"
    ]):
        return "cluster_summary"

    return "cluster_deep_dive"


def resolve_followup_intent(question: str):
    q = question.lower().strip()

    if any(p in q for p in [
        "trend for these", "trend for those", "show their trend", "show me the trend",
        "plot trend", "plot the trend", "show trend", "trend for them"
    ]):
        return "multi_trend"

    if any(p in q for p in [
        "deteriorated the fastest", "deteriorated fastest", "fastest deterioration",
        "which one deteriorated the fastest", "which bridge deteriorated the fastest",
        "which one got worse the fastest"
    ]):
        return "fastest_deterioration"

    if any(p in q for p in [
        "improved the most", "improved most", "which one improved the most"
    ]):
        return "most_improved"

    if any(p in q for p in [
        "average bhi", "average overall bhi", "average of these bridges",
        "average of them", "average bhi of these bridges"
    ]):
        return "average_bhi"

    if any(p in q for p in [
        "compare them", "compare these bridges", "compare those bridges"
    ]):
        return "compare_subset"

    if any(p in q for p in [
        "their profiles", "show their profiles", "give me their profiles",
        "show profile for these bridges", "show me their profile"
    ]):
        return "profiles"

    if any(p in q for p in [
        "which is worst", "which one is worst", "worst among them"
    ]):
        return "worst_in_subset"

    if any(p in q for p in [
        "which is best", "which one is best", "best among them"
    ]):
        return "best_in_subset"

    return None


def compute_bridge_subset_metrics(bridge_id_list):
    rows = []

    for bridge_id in bridge_id_list:
        matched = find_best_bridge_match(bridge_id)
        if matched is None or matched not in pivot_df.index:
            continue

        ts_row = pivot_df.loc[matched].dropna()
        if len(ts_row) < 2:
            slope = np.nan
            first_bhi = ts_row.iloc[0] if len(ts_row) > 0 else np.nan
            last_bhi = ts_row.iloc[-1] if len(ts_row) > 0 else np.nan
        else:
            x = np.array(ts_row.index.tolist(), dtype=float)
            y = np.array(ts_row.values, dtype=float)
            slope, _, _, _, _ = linregress(x, y)
            first_bhi = y[0]
            last_bhi = y[-1]

        summary_row = bridge_summary[bridge_summary["STRUCTURE_NUMBER_008"] == matched]
        latest_bhi = summary_row["Bridge Health Index (Overall)"].iloc[0] if not summary_row.empty else np.nan
        latest_year = summary_row["Year of Data"].iloc[0] if not summary_row.empty else np.nan
        cluster = summary_row["Cluster"].iloc[0] if not summary_row.empty else np.nan

        rows.append({
            "STRUCTURE_NUMBER_008": matched,
            "Latest Year": latest_year,
            "Latest Overall BHI": latest_bhi,
            "Slope": slope,
            "First BHI": first_bhi,
            "Last BHI": last_bhi,
            "Net Change": (last_bhi - first_bhi) if pd.notna(first_bhi) and pd.notna(last_bhi) else np.nan,
            "Cluster": cluster
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def analyze_bridge_subset(question, bridge_id_list):
    subset_df = compute_bridge_subset_metrics(bridge_id_list)

    if subset_df.empty:
        return {
            "text": "I couldn’t find valid time-series data for the selected bridges.",
            "bridge_ids": bridge_id_list,
            "label": "subset_analysis_empty"
        }

    intent = resolve_followup_intent(question)

    if intent == "fastest_deterioration":
        ranked = subset_df.sort_values("Slope", ascending=True).copy()
        winner = ranked.iloc[0]

        lines = [
            "Deterioration analysis for the selected bridges:",
            ""
        ]
        for _, row in ranked.iterrows():
            lines.append(f"- Bridge {row['STRUCTURE_NUMBER_008']}: slope {row['Slope']:.3f}")

        lines.append("")
        lines.append(
            f"The bridge that deteriorated the fastest among these bridges is "
            f"{winner['STRUCTURE_NUMBER_008']} with a slope of {winner['Slope']:.3f}."
        )

        return {
            "text": "\n".join(lines),
            "analysis_df": ranked[["STRUCTURE_NUMBER_008", "Slope", "Latest Overall BHI", "Cluster"]].copy(),
            "execution_steps": [
                "Detected a related follow-up question referring to the previous bridge set.",
                "Computed deterioration slope for each bridge using its BHI time series.",
                "Ranked the bridges by slope from most negative to least negative.",
                "Returned the fastest deteriorating bridge from the selected subset only."
            ],
            "bridge_ids": bridge_id_list,
            "label": "subset_fastest_deterioration"
        }

    if intent == "most_improved":
        ranked = subset_df.sort_values("Slope", ascending=False).copy()
        winner = ranked.iloc[0]

        lines = [
            "Improvement analysis for the selected bridges:",
            ""
        ]
        for _, row in ranked.iterrows():
            lines.append(f"- Bridge {row['STRUCTURE_NUMBER_008']}: slope {row['Slope']:.3f}")

        lines.append("")
        lines.append(
            f"The bridge that improved the most among these bridges is "
            f"{winner['STRUCTURE_NUMBER_008']} with a slope of {winner['Slope']:.3f}."
        )

        return {
            "text": "\n".join(lines),
            "analysis_df": ranked[["STRUCTURE_NUMBER_008", "Slope", "Latest Overall BHI", "Cluster"]].copy(),
            "execution_steps": [
                "Detected a related follow-up question referring to the previous bridge set.",
                "Computed slope for each selected bridge.",
                "Ranked the bridges by slope from highest to lowest.",
                "Returned the most improved bridge from the selected subset only."
            ],
            "bridge_ids": bridge_id_list,
            "label": "subset_most_improved"
        }

    if intent == "average_bhi":
        avg_bhi = subset_df["Latest Overall BHI"].mean()

        return {
            "text": (
                f"The average latest overall BHI for these {len(subset_df)} selected bridges "
                f"is {avg_bhi:.2f}."
            ),
            "analysis_df": subset_df[["STRUCTURE_NUMBER_008", "Latest Overall BHI", "Slope", "Cluster"]].copy(),
            "execution_steps": [
                "Detected a related follow-up question referring to the previous bridge set.",
                "Retrieved the latest overall BHI for each selected bridge.",
                "Computed the mean across the selected subset only."
            ],
            "bridge_ids": bridge_id_list,
            "label": "subset_average_bhi"
        }

    if intent == "compare_subset":
        ranked = subset_df.sort_values("Latest Overall BHI", ascending=True).copy()
        lines = ["Comparison of the selected bridges:"]
        for _, row in ranked.iterrows():
            lines.append(
                f"- Bridge {row['STRUCTURE_NUMBER_008']}: latest BHI {row['Latest Overall BHI']:.2f}, "
                f"slope {row['Slope']:.3f}, cluster {int(row['Cluster']) if pd.notna(row['Cluster']) else 'N/A'}"
            )

        return {
            "text": "\n".join(lines),
            "analysis_df": ranked[["STRUCTURE_NUMBER_008", "Latest Overall BHI", "Slope", "Cluster"]].copy(),
            "execution_steps": [
                "Detected a related follow-up question referring to the previous bridge set.",
                "Pulled latest BHI and slope for each selected bridge.",
                "Returned a comparison limited to the selected subset."
            ],
            "bridge_ids": bridge_id_list,
            "label": "subset_compare"
        }

    if intent == "profiles":
        lines = ["Profiles for the selected bridges:"]
        for bid in bridge_id_list:
            lines.append("")
            lines.append(get_bridge_profile(bid))

        return {
            "text": "\n".join(lines),
            "analysis_df": subset_df.copy(),
            "bridge_ids": bridge_id_list,
            "label": "subset_profiles"
        }

    if intent == "worst_in_subset":
        ranked = subset_df.sort_values("Latest Overall BHI", ascending=True).copy()
        winner = ranked.iloc[0]
        return {
            "text": (
                f"The worst bridge among these selected bridges based on latest overall BHI is "
                f"{winner['STRUCTURE_NUMBER_008']} with a BHI of {winner['Latest Overall BHI']:.2f}."
            ),
            "analysis_df": ranked[["STRUCTURE_NUMBER_008", "Latest Overall BHI", "Slope", "Cluster"]].copy(),
            "bridge_ids": bridge_id_list,
            "label": "subset_worst"
        }

    if intent == "best_in_subset":
        ranked = subset_df.sort_values("Latest Overall BHI", ascending=False).copy()
        winner = ranked.iloc[0]
        return {
            "text": (
                f"The best bridge among these selected bridges based on latest overall BHI is "
                f"{winner['STRUCTURE_NUMBER_008']} with a BHI of {winner['Latest Overall BHI']:.2f}."
            ),
            "analysis_df": ranked[["STRUCTURE_NUMBER_008", "Latest Overall BHI", "Slope", "Cluster"]].copy(),
            "bridge_ids": bridge_id_list,
            "label": "subset_best"
        }

    if intent == "multi_trend":
        return {
            "text": "Showing the trend for these bridges:\n\n" + "\n".join([f"- {b}" for b in bridge_id_list]),
            "figure": make_multi_bridge_trend_figure(bridge_id_list),
            "analysis_df": subset_df[["STRUCTURE_NUMBER_008", "Latest Overall BHI", "Slope", "Cluster"]].copy(),
            "execution_steps": [
                "Detected a related follow-up question referring to the previous bridge set.",
                "Loaded the previously selected bridge IDs from session state.",
                "Plotted the BHI trend for the selected subset only."
            ],
            "bridge_ids": bridge_id_list,
            "label": "subset_multi_trend"
        }

    return None


def update_last_result_context(question, result):
    bridge_ids_local = result.get("bridge_ids")
    cluster_ids_local = result.get("cluster_ids")
    label = result.get("label")

    if bridge_ids_local is not None:
        st.session_state.last_result_context = {
            "bridge_ids": bridge_ids_local,
            "cluster_ids": None,
            "year": None,
            "label": label,
            "result_type": "bridge_subset",
            "question": question
        }
        return

    if cluster_ids_local is not None:
        st.session_state.last_result_context = {
            "bridge_ids": None,
            "cluster_ids": cluster_ids_local,
            "year": None,
            "label": label,
            "result_type": "cluster",
            "question": question
        }
        return

# ---------------------------
# Dataset inspection functions
# ---------------------------
def get_dataset_schema(max_sample_values=5):
    df = static_df.copy()

    rows = []
    for col in df.columns:
        series = df[col]
        dtype = str(series.dtype)
        missing = int(series.isna().sum())
        non_null = int(series.notna().sum())

        sample_values = series.dropna().astype(str).unique()[:max_sample_values]
        sample_values_text = ", ".join(sample_values) if len(sample_values) > 0 else "No non-null values"

        rows.append({
            "Column": col,
            "Data Type": dtype,
            "Non-Null Count": non_null,
            "Missing Count": missing,
            "Sample Values": sample_values_text
        })

    schema_df = pd.DataFrame(rows)

    text = (
        f"The dataset has {df.shape[0]:,} rows and {df.shape[1]:,} columns.\n\n"
        f"The table below shows each column, its data type, missing values, and sample values."
    )

    return {
        "text": text,
        "schema_df": schema_df
    }


def inspect_column(column_name, max_unique=20):
    df = static_df.copy()

    if column_name not in df.columns:
        matches = [c for c in df.columns if c.lower() == str(column_name).lower()]
        if matches:
            column_name = matches[0]
        else:
            contains = [c for c in df.columns if str(column_name).lower() in c.lower()]
            if len(contains) == 1:
                column_name = contains[0]
            else:
                return {
                    "text": f"I couldn’t find a column named '{column_name}' in the dataset."
                }

    s = df[column_name]
    dtype = str(s.dtype)
    missing = int(s.isna().sum())
    non_null = int(s.notna().sum())
    unique_non_null = s.dropna().nunique()

    result = {
        "Column": column_name,
        "Data Type": dtype,
        "Non-Null Count": non_null,
        "Missing Count": missing,
        "Unique Non-Null Values": int(unique_non_null)
    }

    if pd.api.types.is_numeric_dtype(s):
        numeric_s = pd.to_numeric(s, errors="coerce")
        result.update({
            "Min": numeric_s.min(),
            "Max": numeric_s.max(),
            "Mean": numeric_s.mean()
        })

    unique_values = s.dropna().astype(str).unique()[:max_unique]
    values_df = pd.DataFrame({"Sample / Unique Values": unique_values})
    column_df = pd.DataFrame([result])

    text = (
        f"Column inspection for '{column_name}':\n"
        f"- Data type: {dtype}\n"
        f"- Non-null count: {non_null}\n"
        f"- Missing count: {missing}\n"
        f"- Unique non-null values: {unique_non_null}"
    )

    if pd.api.types.is_numeric_dtype(s):
        numeric_s = pd.to_numeric(s, errors="coerce")
        mean_text = f"{numeric_s.mean():.4f}" if pd.notna(numeric_s.mean()) else "N/A"
        text += (
            f"\n- Min: {numeric_s.min()}"
            f"\n- Max: {numeric_s.max()}"
            f"\n- Mean: {mean_text}"
        )

    return {
        "text": text,
        "column_df": column_df,
        "values_df": values_df
    }


def preview_dataset(n_rows=10):
    df = static_df.copy().head(int(n_rows))
    return {
        "text": f"Showing the first {len(df)} rows of the dataset.",
        "preview_df": df
    }


def browse_dataset_rows(offset=0, limit=25, columns=None):
    df = static_df.copy()

    offset = max(0, int(offset))
    limit = max(1, int(limit))

    if columns:
        valid_cols = [c for c in columns if c in df.columns]
        if valid_cols:
            df = df[valid_cols]

    sliced = df.iloc[offset:offset + limit].copy()

    return {
        "text": f"Showing rows {offset + 1} to {min(offset + limit, len(df))} of {len(df):,}.",
        "browse_df": sliced,
        "total_rows": len(df)
    }

# ---------------------------
# Python execution fallback
# ---------------------------
def is_safe_python_code(code: str):
    banned_patterns = [
        r"\bimport\s+os\b",
        r"\bimport\s+sys\b",
        r"\bimport\s+subprocess\b",
        r"\bimport\s+requests\b",
        r"\bimport\s+pathlib\b",
        r"\bfrom\s+os\b",
        r"\bfrom\s+sys\b",
        r"\bfrom\s+subprocess\b",
        r"\bfrom\s+requests\b",
        r"\bfrom\s+pathlib\b",
        r"\bopen\s*\(",
        r"\beval\s*\(",
        r"\bexec\s*\(",
        r"__import__",
        r"\bcompile\s*\(",
        r"\bglobals\s*\(",
        r"\blocals\s*\(",
        r"\binput\s*\(",
    ]

    for pattern in banned_patterns:
        if re.search(pattern, code, flags=re.IGNORECASE):
            return False, f"Blocked unsafe pattern: {pattern}"

    return True, None


def generate_python_analysis_code(user_request: str):
    code_prompt = f"""
You are generating Python analysis code for a bridge dataset app.

Available dataframes:
- static_df
- bridge_summary
- pivot_df
- ts_df
- clustered_df

Important schema notes:
- clustered_df contains cluster assignments.
- In clustered_df, the cluster column is named 'Cluster'.
- Bridge IDs are stored in 'STRUCTURE_NUMBER_008' if clustered_df has been reset_index(),
  or may already be available as 'Bridge_ID'.
- bridge_summary also contains 'Cluster' and 'STRUCTURE_NUMBER_008'.

Rules:
- Use only pandas (pd), numpy (np), and matplotlib.pyplot (plt) if needed.
- Do not import os, sys, subprocess, pathlib, requests, or any network/file libraries.
- Do not read or write files.
- Do not call open().
- Do not use eval() or exec().
- You may use existing imported objects: pd, np, plt.
- Store the final natural-language answer in a variable named result_text.
- Store a step-by-step trace in a variable named execution_steps as a Python list of strings.
- If returning a table, store it in a variable named result_df.
- If the question cannot be answered from the available dataframes, set:
  result_text = "I couldn’t find this information in the dataset."
- Do not attempt forecasting or projection. Forecasting requests must be handled by the dedicated forecast tool, not generated Python code.
- Return only executable Python code. No markdown fences.

User request:
{user_request}
"""
    try:
        response = bedrock.converse(
            modelId=BEDROCK_MODEL_ID,
            system=[{"text": "Return only executable Python code."}],
            messages=[{"role": "user", "content": [{"text": code_prompt}]}]
        )
        output_message = response["output"]["message"]
        code_text = extract_text_from_content_blocks(output_message["content"]).strip()
        code_text = code_text.replace("```python", "").replace("```", "").strip()
        return code_text, None
    except Exception as e:
        return None, f"Code generation failed: {e}"


def run_python_analysis(user_request: str):
    code_text, code_error = generate_python_analysis_code(user_request)
    if code_error:
        return {
            "text": code_error,
            "generated_code": None,
            "execution_steps": None,
            "analysis_df": None,
            "stdout": None
        }

    is_safe, safety_error = is_safe_python_code(code_text)
    if not is_safe:
        return {
            "text": f"Generated code was blocked for safety reasons. {safety_error}",
            "generated_code": code_text,
            "execution_steps": None,
            "analysis_df": None,
            "stdout": None
        }

    clustered_df_for_python = clustered_df.reset_index().copy()

    if "STRUCTURE_NUMBER_008" in clustered_df_for_python.columns:
        clustered_df_for_python["Bridge_ID"] = clustered_df_for_python["STRUCTURE_NUMBER_008"]

    if "Cluster" in clustered_df_for_python.columns:
        clustered_df_for_python["cluster"] = clustered_df_for_python["Cluster"]

    safe_globals = {
        "__builtins__": {
            "len": len,
            "min": min,
            "max": max,
            "sum": sum,
            "abs": abs,
            "round": round,
            "range": range,
            "enumerate": enumerate,
            "sorted": sorted,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "float": float,
            "int": int,
            "str": str,
            "bool": bool,
            "print": print,
        },
        "pd": pd,
        "np": np,
        "plt": plt,
        "static_df": static_df.copy(),
        "bridge_summary": bridge_summary.copy(),
        "pivot_df": pivot_df.copy(),
        "ts_df": ts_df.copy(),
        "clustered_df": clustered_df_for_python,
    }

    local_vars = {}
    stdout_buffer = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout_buffer):
            exec(code_text, safe_globals, local_vars)

        result_text = local_vars.get("result_text", "Python analysis completed, but no result_text was returned.")
        execution_steps = local_vars.get("execution_steps", [])
        result_df = local_vars.get("result_df", None)

        if result_df is not None and not isinstance(result_df, pd.DataFrame):
            try:
                result_df = pd.DataFrame(result_df)
            except Exception:
                result_df = None

        return {
            "text": result_text,
            "generated_code": code_text,
            "execution_steps": execution_steps,
            "analysis_df": result_df,
            "stdout": stdout_buffer.getvalue()
        }

    except Exception as e:
        return {
            "text": f"Python analysis execution failed: {e}",
            "generated_code": code_text,
            "execution_steps": local_vars.get("execution_steps", []),
            "analysis_df": None,
            "stdout": stdout_buffer.getvalue() + "\n" + traceback.format_exc()
        }

# ---------------------------
# Analysis functions
# ---------------------------
def overall_dataset_summary():
    total_bridges = pivot_df.shape[0]
    year_min = min(years_available)
    year_max = max(years_available)

    slopes = bridge_summary["deterioration_slope_per_year"].dropna()
    avg_slope = slopes.mean() if not slopes.empty else np.nan

    cluster_df = (
        bridge_summary["Cluster"]
        .value_counts()
        .sort_index()
        .reset_index()
    )
    cluster_df.columns = ["Cluster", "Number of Bridges"]

    summary_df = pd.DataFrame({
        "Metric": [
            "Total Bridges Used for Clustering",
            "Start Year",
            "End Year",
            "Number of Clusters",
            "Average Deterioration Slope",
            "Raw Rows After dropna()",
            "Bridges With 20+ Records",
            "Constant 20-Year Bridges Removed",
            "Final Rows Used",
            "Final Unique Bridges"
        ],
        "Value": [
            total_bridges,
            year_min,
            year_max,
            N_CLUSTERS,
            round(avg_slope, 4) if pd.notna(avg_slope) else np.nan,
            preprocessing_summary["raw_rows_after_dropna"],
            preprocessing_summary["bridges_with_20plus_records"],
            preprocessing_summary["constant_20year_bridges_removed"],
            preprocessing_summary["final_rows_used"],
            preprocessing_summary["final_unique_bridges"]
        ]
    })

    cluster_lines = [
        f"Cluster {int(row['Cluster'])}: {int(row['Number of Bridges'])}"
        for _, row in cluster_df.iterrows()
    ]

    avg_slope_text = f"{avg_slope:.4f}" if pd.notna(avg_slope) else "N/A"

    summary_text = (
        f"Here is the overall summary of the bridge deterioration dataset:\n\n"
        f"Total bridges used for clustering: {total_bridges:,}\n"
        f"Data span: {year_min} to {year_max}\n"
        f"Clusters: {N_CLUSTERS} clusters using KMeans on raw interpolated BHI trajectories\n"
        f"Bridges with >=20 records: {preprocessing_summary['bridges_with_20plus_records']:,}\n"
        f"Constant 20-year bridges removed: {preprocessing_summary['constant_20year_bridges_removed']:,}\n"
        f"Cluster sizes:\n" + "\n".join(cluster_lines) + "\n"
        f"Average deterioration slope: {avg_slope_text} BHI points per year\n\n"
        f"The tables below show the dataset summary and cluster distribution."
    )

    return {
        "text": summary_text,
        "summary_df": summary_df,
        "cluster_df": cluster_df
    }


def get_bridge_profile(bridge_id):
    matched = find_best_bridge_match(bridge_id)
    if matched is None:
        return f"I could not find a matching bridge for '{bridge_id}'."

    row = bridge_summary[bridge_summary["STRUCTURE_NUMBER_008"] == matched]
    if row.empty:
        return f"No profile found for bridge {matched}."

    row = row.iloc[0]

    details = {
        "Bridge ID": matched,
        "Cluster": row.get("Cluster", np.nan),
        "Latest Year": row.get("Year of Data", np.nan),
        "Overall BHI": row.get("Bridge Health Index (Overall)", np.nan),
        "Deck BHI": row.get("Bridge Health Index (Deck)", np.nan),
        "Superstructure BHI": row.get("Bridge Health Index (Super)", np.nan),
        "Substructure BHI": row.get("Bridge Health Index (Sub)", np.nan),
        "Year Built": row.get("YEAR_BUILT_027", np.nan),
        "ADT": row.get("ADT_029", np.nan),
        "Traffic Lanes": row.get("TRAFFIC_LANES_ON_028A", np.nan),
        "Max Span Length": row.get("MAX_SPAN_LEN_MT_048", np.nan),
        "Structure Length": row.get("STRUCTURE_LEN_MT_049", np.nan),
        "Deck Width": row.get("DECK_WIDTH_MT_052", np.nan),
        "Deterioration Slope": row.get("deterioration_slope_per_year", np.nan),
    }

    text = []
    for key, value in details.items():
        if pd.notna(value):
            if isinstance(value, (float, np.floating)):
                text.append(f"{key}: {value:.2f}")
            else:
                text.append(f"{key}: {value}")

    return "Bridge profile:\n" + "\n".join(text)


def get_bridge_trend(bridge_id):
    matched = find_best_bridge_match(bridge_id)
    if matched is None:
        return f"I could not find a matching bridge for '{bridge_id}'."

    if matched not in pivot_df.index:
        return f"No time-series data found for bridge {matched}."

    row = pivot_df.loc[matched]
    lines = [f"Trend for bridge {matched}:"]
    for year, value in row.items():
        if pd.notna(value):
            lines.append(f"{int(year)}: {float(value):.2f}")

    slope_row = bridge_summary[bridge_summary["STRUCTURE_NUMBER_008"] == matched]
    if not slope_row.empty:
        slope = slope_row["deterioration_slope_per_year"].iloc[0]
        if pd.notna(slope):
            direction = "declining" if slope < 0 else "improving" if slope > 0 else "stable"
            lines.append(f"\nEstimated slope: {slope:.3f} BHI points/year ({direction}).")

    return "\n".join(lines)


def compare_two_bridges(bridge_id_1, bridge_id_2):
    matched1 = find_best_bridge_match(bridge_id_1)
    matched2 = find_best_bridge_match(bridge_id_2)

    if matched1 is None:
        return f"I could not find a matching bridge for '{bridge_id_1}'."
    if matched2 is None:
        return f"I could not find a matching bridge for '{bridge_id_2}'."

    row1 = bridge_summary[bridge_summary["STRUCTURE_NUMBER_008"] == matched1]
    row2 = bridge_summary[bridge_summary["STRUCTURE_NUMBER_008"] == matched2]

    if row1.empty or row2.empty:
        return "Missing summary data for one or both bridges."

    row1 = row1.iloc[0]
    row2 = row2.iloc[0]

    return (
        f"Bridge {matched1} is in Cluster {int(row1['Cluster'])} with latest overall BHI "
        f"{row1['Bridge Health Index (Overall)']:.2f} and deterioration slope "
        f"{row1['deterioration_slope_per_year']:.3f} per year.\n\n"
        f"Bridge {matched2} is in Cluster {int(row2['Cluster'])} with latest overall BHI "
        f"{row2['Bridge Health Index (Overall)']:.2f} and deterioration slope "
        f"{row2['deterioration_slope_per_year']:.3f} per year."
    )


def get_cluster_summary(cluster_id):
    try:
        cluster_id = int(cluster_id)
    except Exception:
        return f"Invalid cluster id: {cluster_id}"

    subset = bridge_summary[bridge_summary["Cluster"] == cluster_id].copy()
    if subset.empty:
        return f"No bridges found in cluster {cluster_id}."

    cols_to_fix = [
        "Bridge Health Index (Overall)",
        "Bridge Health Index (Deck)",
        "Bridge Health Index (Super)",
        "Bridge Health Index (Sub)",
        "YEAR_BUILT_027",
        "ADT_029",
        "MAX_SPAN_LEN_MT_048",
        "deterioration_slope_per_year"
    ]

    for col in cols_to_fix:
        if col in subset.columns:
            subset[col] = pd.to_numeric(subset[col], errors="coerce")

    if "YEAR_BUILT_027" in subset.columns:
        subset["YEAR_BUILT_027"] = clean_year_built(subset["YEAR_BUILT_027"])

    metrics = {
        "count": len(subset),
        "avg_latest_overall_bhi": subset["Bridge Health Index (Overall)"].mean() if "Bridge Health Index (Overall)" in subset.columns else np.nan,
        "avg_deck_bhi": subset["Bridge Health Index (Deck)"].mean() if "Bridge Health Index (Deck)" in subset.columns else np.nan,
        "avg_super_bhi": subset["Bridge Health Index (Super)"].mean() if "Bridge Health Index (Super)" in subset.columns else np.nan,
        "avg_sub_bhi": subset["Bridge Health Index (Sub)"].mean() if "Bridge Health Index (Sub)" in subset.columns else np.nan,
        "avg_year_built": subset["YEAR_BUILT_027"].mean() if "YEAR_BUILT_027" in subset.columns else np.nan,
        "avg_adt": subset["ADT_029"].mean() if "ADT_029" in subset.columns else np.nan,
        "avg_span_len": subset["MAX_SPAN_LEN_MT_048"].mean() if "MAX_SPAN_LEN_MT_048" in subset.columns else np.nan,
        "avg_slope": subset["deterioration_slope_per_year"].mean() if "deterioration_slope_per_year" in subset.columns else np.nan,
    }

    return (
        f"Cluster {cluster_id} contains {metrics['count']:,} bridges.\n"
        f"Average latest overall BHI: {metrics['avg_latest_overall_bhi']:.2f}\n"
        f"Average deck BHI: {metrics['avg_deck_bhi']:.2f}\n"
        f"Average superstructure BHI: {metrics['avg_super_bhi']:.2f}\n"
        f"Average substructure BHI: {metrics['avg_sub_bhi']:.2f}\n"
        f"Average year built: {metrics['avg_year_built']:.1f}\n"
        f"Average ADT: {metrics['avg_adt']:.1f}\n"
        f"Average max span length: {metrics['avg_span_len']:.2f}\n"
        f"Average deterioration slope: {metrics['avg_slope']:.3f} BHI points/year"
    )


def compare_two_clusters(cluster_id_1, cluster_id_2):
    try:
        cluster_id_1 = int(cluster_id_1)
        cluster_id_2 = int(cluster_id_2)
    except Exception:
        return "Invalid cluster ids."

    subset1 = bridge_summary[bridge_summary["Cluster"] == cluster_id_1].copy()
    subset2 = bridge_summary[bridge_summary["Cluster"] == cluster_id_2].copy()

    if subset1.empty:
        return f"No bridges found in cluster {cluster_id_1}."
    if subset2.empty:
        return f"No bridges found in cluster {cluster_id_2}."

    cols_to_fix = [
        "Bridge Health Index (Overall)",
        "Bridge Health Index (Deck)",
        "Bridge Health Index (Super)",
        "Bridge Health Index (Sub)",
        "YEAR_BUILT_027",
        "ADT_029",
        "MAX_SPAN_LEN_MT_048",
        "deterioration_slope_per_year"
    ]

    for col in cols_to_fix:
        if col in subset1.columns:
            subset1[col] = pd.to_numeric(subset1[col], errors="coerce")
        if col in subset2.columns:
            subset2[col] = pd.to_numeric(subset2[col], errors="coerce")

    if "YEAR_BUILT_027" in subset1.columns:
        subset1["YEAR_BUILT_027"] = clean_year_built(subset1["YEAR_BUILT_027"])
    if "YEAR_BUILT_027" in subset2.columns:
        subset2["YEAR_BUILT_027"] = clean_year_built(subset2["YEAR_BUILT_027"])

    metrics1 = {
        "count": len(subset1),
        "avg_latest_overall_bhi": subset1["Bridge Health Index (Overall)"].mean(),
        "avg_deck_bhi": subset1["Bridge Health Index (Deck)"].mean() if "Bridge Health Index (Deck)" in subset1.columns else np.nan,
        "avg_super_bhi": subset1["Bridge Health Index (Super)"].mean() if "Bridge Health Index (Super)" in subset1.columns else np.nan,
        "avg_sub_bhi": subset1["Bridge Health Index (Sub)"].mean() if "Bridge Health Index (Sub)" in subset1.columns else np.nan,
        "avg_year_built": subset1["YEAR_BUILT_027"].mean() if "YEAR_BUILT_027" in subset1.columns else np.nan,
        "avg_adt": subset1["ADT_029"].mean() if "ADT_029" in subset1.columns else np.nan,
        "avg_span_len": subset1["MAX_SPAN_LEN_MT_048"].mean() if "MAX_SPAN_LEN_MT_048" in subset1.columns else np.nan,
        "avg_slope": subset1["deterioration_slope_per_year"].mean() if "deterioration_slope_per_year" in subset1.columns else np.nan,
    }

    metrics2 = {
        "count": len(subset2),
        "avg_latest_overall_bhi": subset2["Bridge Health Index (Overall)"].mean(),
        "avg_deck_bhi": subset2["Bridge Health Index (Deck)"].mean() if "Bridge Health Index (Deck)" in subset2.columns else np.nan,
        "avg_super_bhi": subset2["Bridge Health Index (Super)"].mean() if "Bridge Health Index (Super)" in subset2.columns else np.nan,
        "avg_sub_bhi": subset2["Bridge Health Index (Sub)"].mean() if "Bridge Health Index (Sub)" in subset2.columns else np.nan,
        "avg_year_built": subset2["YEAR_BUILT_027"].mean() if "YEAR_BUILT_027" in subset2.columns else np.nan,
        "avg_adt": subset2["ADT_029"].mean() if "ADT_029" in subset2.columns else np.nan,
        "avg_span_len": subset2["MAX_SPAN_LEN_MT_048"].mean() if "MAX_SPAN_LEN_MT_048" in subset2.columns else np.nan,
        "avg_slope": subset2["deterioration_slope_per_year"].mean() if "deterioration_slope_per_year" in subset2.columns else np.nan,
    }

    interpretation = []

    if metrics1["avg_latest_overall_bhi"] > metrics2["avg_latest_overall_bhi"]:
        interpretation.append(f"Cluster {cluster_id_1} has the higher average latest overall BHI.")
    else:
        interpretation.append(f"Cluster {cluster_id_2} has the higher average latest overall BHI.")

    if metrics1["avg_slope"] > metrics2["avg_slope"]:
        interpretation.append(f"Cluster {cluster_id_1} shows the stronger positive average deterioration slope.")
    else:
        interpretation.append(f"Cluster {cluster_id_2} shows the stronger positive average deterioration slope.")

    if metrics1["avg_adt"] > metrics2["avg_adt"]:
        interpretation.append(f"Cluster {cluster_id_1} carries higher average daily traffic.")
    else:
        interpretation.append(f"Cluster {cluster_id_2} carries higher average daily traffic.")

    if metrics1["avg_span_len"] > metrics2["avg_span_len"]:
        interpretation.append(f"Cluster {cluster_id_1} has longer spans on average.")
    else:
        interpretation.append(f"Cluster {cluster_id_2} has longer spans on average.")

    text = (
        f"Here is a comparison between Cluster {cluster_id_1} and Cluster {cluster_id_2}:\n\n"
        f"Cluster {cluster_id_1}:\n"
        f"- Contains {metrics1['count']:,} bridges\n"
        f"- Average latest overall BHI: {metrics1['avg_latest_overall_bhi']:.2f}\n"
        f"- Average deck BHI: {metrics1['avg_deck_bhi']:.2f}\n"
        f"- Average superstructure BHI: {metrics1['avg_super_bhi']:.2f}\n"
        f"- Average substructure BHI: {metrics1['avg_sub_bhi']:.2f}\n"
        f"- Average year built: {metrics1['avg_year_built']:.1f}\n"
        f"- Average ADT: {metrics1['avg_adt']:.1f}\n"
        f"- Average max span length: {metrics1['avg_span_len']:.2f}\n"
        f"- Average deterioration slope: {metrics1['avg_slope']:.3f} BHI points/year\n\n"
        f"Cluster {cluster_id_2}:\n"
        f"- Contains {metrics2['count']:,} bridges\n"
        f"- Average latest overall BHI: {metrics2['avg_latest_overall_bhi']:.2f}\n"
        f"- Average deck BHI: {metrics2['avg_deck_bhi']:.2f}\n"
        f"- Average superstructure BHI: {metrics2['avg_super_bhi']:.2f}\n"
        f"- Average substructure BHI: {metrics2['avg_sub_bhi']:.2f}\n"
        f"- Average year built: {metrics2['avg_year_built']:.1f}\n"
        f"- Average ADT: {metrics2['avg_adt']:.1f}\n"
        f"- Average max span length: {metrics2['avg_span_len']:.2f}\n"
        f"- Average deterioration slope: {metrics2['avg_slope']:.3f} BHI points/year\n\n"
        f"Interpretation:\n- " + "\n- ".join(interpretation)
    )

    return text


def get_cluster_pca_drivers(cluster_id, top_n=8):
    try:
        cluster_id = int(cluster_id)
        top_n = int(top_n)
    except Exception:
        return {"text": f"Invalid cluster_id or top_n: cluster_id={cluster_id}, top_n={top_n}"}

    if "Cluster" not in clustered_df.columns:
        return {"text": "I couldn’t find cluster assignments in the dataset."}

    cluster_subset = clustered_df[clustered_df["Cluster"] == cluster_id].copy()

    if cluster_subset.empty:
        return {"text": f"No bridges found in cluster {cluster_id}."}

    cluster_subset = cluster_subset.reset_index()
    bridge_ids_local = cluster_subset["STRUCTURE_NUMBER_008"].astype(str).tolist()

    cluster_static = static_df[static_df["STRUCTURE_NUMBER_008"].astype(str).isin(bridge_ids_local)].copy()

    if cluster_static.empty:
        return {"text": f"I couldn’t find static records for cluster {cluster_id} bridges."}

    cluster_static["Cluster"] = cluster_id

    cols_to_drop = [col for col in ["Unnamed: 0", "Cluster"] if col in cluster_static.columns]

    df_numeric = cluster_static.select_dtypes(include=["number"]).drop(columns=cols_to_drop, errors="ignore")

    if df_numeric.empty:
        return {"text": f"No numeric PCA input columns were available for cluster {cluster_id}."}

    corr_matrix = df_numeric.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]
    df_filtered = df_numeric.drop(columns=to_drop, errors="ignore")

    manual_drop = [
        "STRUCTURE_TYPE_043B",
        "OPERATING_RATING_064",
        "COUNTY_CODE_003",
        "APPR_TYPE_044B",
        "LOWEST_RATING",
        "SUFFICIENCY_RATING",
        "STRUCTURE_KIND_043A",
        "Year of Data"
    ]
    df_filtered_2 = df_filtered.drop(columns=manual_drop, errors="ignore")

    if df_filtered_2.shape[1] < 2:
        return {
            "text": f"I found partial information, but not enough to answer fully. Too few PCA variables remained for cluster {cluster_id}."
        }

    df_filtered_2 = df_filtered_2.dropna()

    if df_filtered_2.shape[0] < 2:
        return {
            "text": f"I found partial information, but not enough to answer fully. Too few rows remained after cleaning for cluster {cluster_id}."
        }

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_filtered_2)

    pca = PCA()
    pca.fit(X_scaled)

    loadings = pd.DataFrame(pca.components_, columns=df_filtered_2.columns)
    pc1_sorted = loadings.loc[0].sort_values(ascending=False)

    pc1_df = pc1_sorted.reset_index()
    pc1_df.columns = ["Feature", "PC1_Loading"]

    top_positive = pc1_sorted.head(top_n)
    top_negative = pc1_sorted.sort_values(ascending=True).head(top_n)

    explained_var = float(pca.explained_variance_ratio_[0]) if len(pca.explained_variance_ratio_) > 0 else np.nan

    positive_lines = [f"- {feature}: {value:.6f}" for feature, value in top_positive.items()]
    negative_lines = [f"- {feature}: {value:.6f}" for feature, value in top_negative.items()]

    text = (
        f"PCA results for cluster {cluster_id} based on PC1 loadings:\n\n"
        f"PC1 explained variance ratio: {explained_var:.4f}\n\n"
        f"Top positive PC1 loadings:\n" + "\n".join(positive_lines) + "\n\n"
        f"Top negative PC1 loadings:\n" + "\n".join(negative_lines) + "\n\n"
        f"Interpretation note: larger absolute PC1 loadings indicate stronger contribution to the first principal component. "
        f"This does not automatically mean causal deterioration drivers."
    )

    return {
        "text": text,
        "pc1_table": pc1_df,
        "explained_variance_ratio_pc1": explained_var
    }


def get_top_deteriorating_bridges(top_n=5):
    subset = bridge_summary.dropna(subset=["deterioration_slope_per_year"]).copy()
    subset = subset.sort_values("deterioration_slope_per_year", ascending=True).head(top_n)

    lines = [f"Top {top_n} fastest deteriorating bridges based on slope:"]
    bridge_id_list = []

    for _, row in subset.iterrows():
        bid = str(row["STRUCTURE_NUMBER_008"])
        bridge_id_list.append(bid)
        lines.append(
            f"- Bridge {bid}: slope {row['deterioration_slope_per_year']:.3f}, "
            f"latest overall BHI {row['Bridge Health Index (Overall)']:.2f}, cluster {int(row['Cluster'])}"
        )
    return {
        "text": "\n".join(lines),
        "bridge_ids": bridge_id_list,
        "analysis_df": subset[[
            "STRUCTURE_NUMBER_008",
            "deterioration_slope_per_year",
            "Bridge Health Index (Overall)",
            "Cluster"
        ]].copy(),
        "label": f"top_{top_n}_deteriorating_bridges"
    }


def get_top_best_bridges(year, top_n=5):
    year = int(year)
    subset = ts_df[ts_df["Year of Data"] == year].copy()
    if subset.empty:
        return {"text": f"No data found for year {year}."}

    subset = subset.sort_values("Bridge Health Index (Overall)", ascending=False).head(top_n)

    lines = [f"Top {top_n} bridges by overall BHI in {year}:"]
    bridge_id_list = []

    for _, row in subset.iterrows():
        bid = str(row["STRUCTURE_NUMBER_008"])
        bridge_id_list.append(bid)
        lines.append(f"- Bridge {bid}: {row['Bridge Health Index (Overall)']:.2f}")

    return {
        "text": "\n".join(lines),
        "bridge_ids": bridge_id_list,
        "analysis_df": subset[[
            "STRUCTURE_NUMBER_008",
            "Year of Data",
            "Bridge Health Index (Overall)"
        ]].copy(),
        "label": f"top_{top_n}_best_bridges_{year}"
    }


def get_top_worst_bridges(year, top_n=5):
    year = int(year)
    subset = ts_df[ts_df["Year of Data"] == year].copy()
    if subset.empty:
        return {"text": f"No data found for year {year}."}

    subset = subset.sort_values("Bridge Health Index (Overall)", ascending=True).head(top_n)

    lines = [f"Top {top_n} worst bridges by overall BHI in {year}:"]
    bridge_id_list = []

    for _, row in subset.iterrows():
        bid = str(row["STRUCTURE_NUMBER_008"])
        bridge_id_list.append(bid)
        lines.append(f"- Bridge {bid}: {row['Bridge Health Index (Overall)']:.2f}")

    return {
        "text": "\n".join(lines),
        "bridge_ids": bridge_id_list,
        "analysis_df": subset[[
            "STRUCTURE_NUMBER_008",
            "Year of Data",
            "Bridge Health Index (Overall)"
        ]].copy(),
        "label": f"top_{top_n}_worst_bridges_{year}"
    }

def get_bridges_in_cluster(cluster_id):
    try:
        cluster_id = int(cluster_id)
    except Exception:
        return {"text": f"Invalid cluster id: {cluster_id}"}

    if "Cluster" not in clustered_df.columns:
        return {"text": "I couldn’t find cluster assignments in the dataset."}

    subset = clustered_df[clustered_df["Cluster"] == cluster_id].copy()

    if subset.empty:
        return {"text": f"No bridges found in cluster {cluster_id}."}

    subset = subset.reset_index()

    if "STRUCTURE_NUMBER_008" not in subset.columns:
        return {
            "text": "I found partial information, but not enough to answer fully."
        }

    result_df = subset[["STRUCTURE_NUMBER_008"]].copy()
    result_df = result_df.rename(columns={"STRUCTURE_NUMBER_008": "Bridge ID"})

    bridge_ids_local = result_df["Bridge ID"].astype(str).tolist()

    return {
        "text": f"I found {len(result_df)} bridges in cluster {cluster_id}.",
        "analysis_df": result_df,
        "bridge_ids": bridge_ids_local,
        "label": f"bridges_in_cluster_{cluster_id}"
    }


def get_cluster_deep_dive(cluster_id):
    try:
        cluster_id = int(cluster_id)
    except Exception:
        return {"text": f"Invalid cluster id: {cluster_id}"}

    subset = bridge_summary[bridge_summary["Cluster"] == cluster_id].copy()
    if subset.empty:
        return {"text": f"No bridges found in cluster {cluster_id}."}

    numeric_cols = [
        "Bridge Health Index (Overall)",
        "Bridge Health Index (Deck)",
        "Bridge Health Index (Super)",
        "Bridge Health Index (Sub)",
        "YEAR_BUILT_027",
        "ADT_029",
        "MAX_SPAN_LEN_MT_048",
        "STRUCTURE_LEN_MT_049",
        "DECK_WIDTH_MT_052",
        "deterioration_slope_per_year"
    ]

    for col in numeric_cols:
        if col in subset.columns:
            subset[col] = pd.to_numeric(subset[col], errors="coerce")

    if "YEAR_BUILT_027" in subset.columns:
        subset["YEAR_BUILT_027"] = clean_year_built(subset["YEAR_BUILT_027"])

    result_df = pd.DataFrame({
        "Metric": [
            "Bridge count",
            "Avg overall BHI",
            "Median overall BHI",
            "Min overall BHI",
            "Max overall BHI",
            "Avg deterioration slope",
            "Median deterioration slope",
            "Worst slope",
            "Best slope",
            "Avg year built",
            "Avg ADT",
            "Avg max span length",
            "Avg structure length",
            "Avg deck width"
        ],
        "Value": [
            len(subset),
            subset["Bridge Health Index (Overall)"].mean(),
            subset["Bridge Health Index (Overall)"].median(),
            subset["Bridge Health Index (Overall)"].min(),
            subset["Bridge Health Index (Overall)"].max(),
            subset["deterioration_slope_per_year"].mean(),
            subset["deterioration_slope_per_year"].median(),
            subset["deterioration_slope_per_year"].min(),
            subset["deterioration_slope_per_year"].max(),
            subset["YEAR_BUILT_027"].mean(),
            subset["ADT_029"].mean(),
            subset["MAX_SPAN_LEN_MT_048"].mean(),
            subset["STRUCTURE_LEN_MT_049"].mean(),
            subset["DECK_WIDTH_MT_052"].mean(),
        ]
    })

    worst_bridges = subset.sort_values("deterioration_slope_per_year", ascending=True).head(5)[[
        "STRUCTURE_NUMBER_008",
        "Bridge Health Index (Overall)",
        "deterioration_slope_per_year",
        "ADT_029",
        "YEAR_BUILT_027"
    ]].copy()

    text = (
        f"Here is a deeper analysis of cluster {cluster_id}.\n\n"
        f"This goes beyond the basic summary and includes:\n"
        f"- distribution of BHI and slopes\n"
        f"- best and worst deterioration behavior\n"
        f"- age, traffic, and geometry patterns\n"
        f"- the 5 fastest deteriorating bridges in the cluster"
    )

    return {
        "text": text,
        "summary_df": result_df,
        "analysis_df": worst_bridges,
        "cluster_ids": [cluster_id],
        "label": "cluster_deep_dive"
    }




def get_cluster_trend_stats(cluster_id):
    try:
        cluster_id = int(cluster_id)
    except Exception:
        return None

    subset = clustered_df[clustered_df["Cluster"] == cluster_id].drop(columns="Cluster", errors="ignore").copy()
    if subset.empty:
        return None

    years_local = [c for c in subset.columns if isinstance(c, (int, np.integer, float, np.floating))]
    if not years_local:
        return None

    subset = subset[years_local].copy()
    subset = subset.apply(pd.to_numeric, errors="coerce")

    median_trend = subset.median(axis=0)
    q1_trend = subset.quantile(0.25, axis=0)
    q3_trend = subset.quantile(0.75, axis=0)
    iqr_trend = q3_trend - q1_trend

    valid_idx = median_trend.dropna().index.tolist()
    if len(valid_idx) < 2:
        return None

    first_year = int(valid_idx[0])
    last_year = int(valid_idx[-1])

    first_median = float(median_trend.loc[first_year])
    last_median = float(median_trend.loc[last_year])
    net_change = last_median - first_median

    x = np.array(valid_idx, dtype=float)
    y = np.array([median_trend.loc[yr] for yr in valid_idx], dtype=float)

    if len(x) >= 2 and np.isfinite(y).sum() >= 2:
        slope, _, _, _, _ = linregress(x, y)
    else:
        slope = np.nan

    peak_year = int(median_trend.idxmax())
    trough_year = int(median_trend.idxmin())
    peak_value = float(median_trend.max())
    trough_value = float(median_trend.min())

    avg_iqr = float(iqr_trend.mean()) if not iqr_trend.empty else np.nan
    max_iqr_year = int(iqr_trend.idxmax()) if not iqr_trend.empty else None
    max_iqr_value = float(iqr_trend.max()) if not iqr_trend.empty else np.nan

    return {
        "cluster_id": cluster_id,
        "n_bridges": int(subset.shape[0]),
        "years": valid_idx,
        "first_year": first_year,
        "last_year": last_year,
        "first_median": first_median,
        "last_median": last_median,
        "net_change": net_change,
        "slope": float(slope) if pd.notna(slope) else np.nan,
        "peak_year": peak_year,
        "peak_value": peak_value,
        "trough_year": trough_year,
        "trough_value": trough_value,
        "avg_iqr": avg_iqr,
        "max_iqr_year": max_iqr_year,
        "max_iqr_value": max_iqr_value,
        "median_trend": median_trend,
        "iqr_trend": iqr_trend,
    }


def interpret_slope_text(slope):
    if pd.isna(slope):
        return "no clear trend could be estimated"
    if slope > 0.1:
        return "an overall improving trend"
    if slope < -0.1:
        return "an overall deteriorating trend"
    return "a largely stable trend"


def interpret_cluster_trend(cluster_id):
    stats = get_cluster_trend_stats(cluster_id)
    if stats is None:
        return f"I couldn’t compute the median trend for cluster {cluster_id}."

    trend_text = interpret_slope_text(stats["slope"])

    return (
        f"For cluster {cluster_id}, the median BHI line represents the middle Bridge Health Index value across all bridges in the cluster at each year.\n\n"
        f"In this cluster, the median line shows {trend_text} from {stats['first_year']} to {stats['last_year']}. "
        f"The median BHI changes from {stats['first_median']:.2f} to {stats['last_median']:.2f}, "
        f"which is a net change of {stats['net_change']:.2f} points. "
        f"The highest median value occurs around {stats['peak_year']} at {stats['peak_value']:.2f}, "
        f"and the lowest occurs around {stats['trough_year']} at {stats['trough_value']:.2f}.\n\n"
        f"So, the median line should be read as the typical bridge trajectory in this cluster, not every individual bridge. "
        f"If individual lines spread away from the median, that means some bridges behave differently from the cluster’s typical pattern."
    )


def interpret_cluster_fluctuations(cluster_id):
    stats = get_cluster_trend_stats(cluster_id)
    if stats is None:
        return f"I couldn’t compute fluctuation statistics for cluster {cluster_id}."

    if pd.isna(stats["avg_iqr"]):
        variability_text = "I could not estimate the year-to-year spread reliably."
    elif stats["avg_iqr"] < 5:
        variability_text = "Fluctuations are relatively small, which means bridges in this cluster behave fairly consistently around the median."
    elif stats["avg_iqr"] < 15:
        variability_text = "Fluctuations are moderate, which means there is some variation across bridges, but the cluster still follows a common overall pattern."
    else:
        variability_text = "Fluctuations are relatively large, which means bridges in this cluster differ substantially from one another over time."

    extra_text = ""
    if stats["max_iqr_year"] is not None and pd.notna(stats["max_iqr_value"]):
        extra_text = (
            f" The largest spread appears around {stats['max_iqr_year']}, "
            f"where the interquartile range is {stats['max_iqr_value']:.2f}."
        )

    return (
        f"For cluster {cluster_id}, fluctuations in BHI over time should be interpreted as variation around the cluster median line.\n\n"
        f"{variability_text}{extra_text}\n\n"
        f"If the median line itself moves sharply upward or downward, that indicates a cluster-level shift in typical bridge condition. "
        f"If the median stays fairly steady but the individual bridge lines are widely scattered, that means the cluster contains bridges with mixed behaviors even though the typical value is stable."
    )

# ---------------------------
# Plotting
# ---------------------------
def make_bridge_trend_figure(bridge_id):
    matched = find_best_bridge_match(bridge_id)
    if matched is None or matched not in pivot_df.index:
        return None

    row = pivot_df.loc[matched]
    years_local = list(row.index)
    values = list(row.values)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(years_local, values, marker="o", linewidth=2)
    ax.set_title(f"Bridge Trend: {matched}", fontsize=12, pad=10)
    ax.set_xlabel("Year")
    ax.set_ylabel("Bridge Health Index (Overall)")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def make_multi_bridge_trend_figure(bridge_id_list):
    if not bridge_id_list:
        return None

    valid_ids = [find_best_bridge_match(b) for b in bridge_id_list]
    valid_ids = [b for b in valid_ids if b is not None and b in pivot_df.index]

    if not valid_ids:
        return None

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for bid in valid_ids:
        row = pivot_df.loc[bid]
        ax.plot(row.index, row.values, marker="o", linewidth=2, label=bid)

    ax.set_title("Trend for Selected Bridges", fontsize=12, pad=10)
    ax.set_xlabel("Year")
    ax.set_ylabel("Bridge Health Index (Overall)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def make_compare_bridges_figure(bridge_id_1, bridge_id_2):
    matched1 = find_best_bridge_match(bridge_id_1)
    matched2 = find_best_bridge_match(bridge_id_2)

    if matched1 is None or matched2 is None:
        return None
    if matched1 not in pivot_df.index or matched2 not in pivot_df.index:
        return None

    row1 = pivot_df.loc[matched1]
    row2 = pivot_df.loc[matched2]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(row1.index, row1.values, marker="o", linewidth=2, label=matched1)
    ax.plot(row2.index, row2.values, marker="o", linewidth=2, label=matched2)
    ax.set_title("Bridge Deterioration Comparison", fontsize=12, pad=10)
    ax.set_xlabel("Year")
    ax.set_ylabel("Bridge Health Index (Overall)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def make_cluster_median_figure(cluster_id):
    try:
        cluster_id = int(cluster_id)
    except Exception:
        return None

    subset = clustered_df[clustered_df["Cluster"] == cluster_id].drop(columns="Cluster")
    if subset.empty:
        return None

    years_local = subset.columns.tolist()
    median_trend = subset.median(axis=0)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for _, row in subset.iterrows():
        ax.plot(years_local, row.values, alpha=0.08, linewidth=1)

    ax.plot(years_local, median_trend.values, linewidth=3, marker="o")
    ax.set_title(f"Cluster {cluster_id} Median Deterioration Trend", fontsize=12, pad=10)
    ax.set_xlabel("Year")
    ax.set_ylabel("Bridge Health Index (Overall)")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def make_compare_clusters_figure(cluster_id_1, cluster_id_2):
    try:
        cluster_id_1 = int(cluster_id_1)
        cluster_id_2 = int(cluster_id_2)
    except Exception:
        return None

    subset1 = clustered_df[clustered_df["Cluster"] == cluster_id_1].drop(columns="Cluster")
    subset2 = clustered_df[clustered_df["Cluster"] == cluster_id_2].drop(columns="Cluster")

    if subset1.empty or subset2.empty:
        return None

    years_local = subset1.columns.tolist()
    median_trend_1 = subset1.median(axis=0)
    median_trend_2 = subset2.median(axis=0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(years_local, median_trend_1.values, marker="o", linewidth=2, label=f"Cluster {cluster_id_1}")
    ax.plot(years_local, median_trend_2.values, marker="o", linewidth=2, label=f"Cluster {cluster_id_2}")
    ax.set_title(f"Cluster {cluster_id_1} vs Cluster {cluster_id_2}", fontsize=12, pad=10)
    ax.set_xlabel("Year")
    ax.set_ylabel("Bridge Health Index (Overall)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig

# ---------------------------
# Bedrock prompt + tools
# ---------------------------
SYSTEM_PROMPT = """
You are an AI assistant for analyzing bridge deterioration data.
You have access to tools to retrieve real data.

You must ALWAYS use tools to answer questions about the dataset.

You must only answer from tool outputs or dataset-derived results.
If the requested information is missing, unavailable, ambiguous, or not supported by the data, say so clearly.

Do not guess, infer missing values, or invent explanations.

If no tool result supports the answer, respond with:
"I couldn’t find this information in the dataset."

If the answer is partial, say:
"I found partial information, but not enough to answer fully."

You must NEVER answer using your own knowledge.
If you do not use a tool, your answer is invalid.

Available analyses include:
- overall dataset summary
- bridge profile
- bridge trend
- compare bridges
- cluster summary
- compare clusters
- top deteriorating bridges
- top best bridges by year
- top worst bridges by year
- PCA-based cluster feature drivers
- dataset schema
- dataset preview
- inspect a column
- browse dataset rows
- python analysis fallback

Important PCA rule:
- If the user asks about key drivers, important features, PC1 loadings, or what characterizes a cluster, use the PCA tool.
- Report PCA loadings as feature contributions to PC1.
- Do not describe PCA loadings as causal unless the data explicitly supports causality.

Dataset inspection rule:
- If the user asks what columns exist, what values a column contains, what the dataset looks like, what data types are present, or to inspect the dataset, use the dataset inspection tools.
- If the user asks to browse, inspect, show rows, or explore the table, use the browse dataset rows tool.

Python analysis rule:
- If the user's request cannot be answered by an existing tool, you may use the python_analysis tool.
- Only use python_analysis when a more specific existing tool is insufficient.
- The final answer must come only from executed Python results.
- Do not guess.
- Return a traceable summary of the analysis steps.

Follow-up context rule:
- If the user asks a related follow-up after a prior result, continue from that prior result.
- If the prior result returned bridge_ids, and the new question is about "which one", "them", "these bridges", "those bridges", "their trend", "their average", "compare them", or similar, the question refers to the previously returned bridge_ids unless the user clearly says otherwise.
- Never switch from a selected subset back to the full dataset for a related follow-up unless the user explicitly asks for all bridges.
- If a subset context exists and the user asks "Which one deteriorated the fastest?", answer from that subset only.
"""

def extract_text_from_content_blocks(content_blocks):
    parts = []
    for block in content_blocks:
        if "text" in block:
            cleaned = strip_thinking_blocks(block["text"])
            if cleaned:
                parts.append(cleaned)
    return "\n".join(parts).strip()


def extract_cluster_ids(text):
    matches = re.findall(r"cluster\s+(\d+)", text.lower())
    return [int(x) for x in matches]


def route_question(question: str):
    q = question.lower().strip()
    cluster_ids_local = extract_cluster_ids(q)

    if is_forecast_explanation_question(question):
        return {
            "mode": "forecast_explanation"
        }

    if is_forecast_question(question):
        bridge_id = extract_bridge_id_from_question(question)
        if bridge_id is None:
            return {
                "mode": "direct_text",
                "text": "Please include a valid bridge ID for forecasting, for example: Forecast bridge 200000BC3107010 for the next 20 years."
            }
        return {
            "mode": "direct_tool",
            "tool_name": "forecast_bridge_20_years",
            "tool_input": {"bridge_id": bridge_id, "forecast_horizon": 20}
        }

    if (
        len(cluster_ids_local) == 1 and
        any(phrase in q for phrase in [
            "median bhi line",
            "median line",
            "interpret the median",
            "interpret this plot",
            "interpret this graph"
        ])
    ):
        return {
            "mode": "direct_text",
            "text": interpret_cluster_trend(cluster_ids_local[0]),
            "cluster_ids": [cluster_ids_local[0]],
            "label": "cluster_median_interpretation"
        }

    if (
        len(cluster_ids_local) == 1 and
        any(phrase in q for phrase in [
            "fluctuations in bhi",
            "fluctuations over time",
            "variation over time",
            "what do the fluctuations mean"
        ])
    ):
        return {
            "mode": "direct_text",
            "text": interpret_cluster_fluctuations(cluster_ids_local[0]),
            "cluster_ids": [cluster_ids_local[0]],
            "label": "cluster_fluctuation_interpretation"
        }

    if (
        len(cluster_ids_local) == 1 and
        any(phrase in q for phrase in [
            "what features characterize",
            "what characterizes",
            "main factors",
            "important variables",
            "key variables",
            "key drivers",
            "feature drivers"
        ])
    ):
        return {
            "mode": "direct_tool",
            "tool_name": "cluster_pca_drivers",
            "tool_input": {"cluster_id": cluster_ids_local[0], "top_n": 8}
        }

    if (
        len(cluster_ids_local) == 1 and
        any(phrase in q for phrase in [
            "why is cluster",
            "how is cluster",
            "why cluster"
        ]) and
        "compare" not in q and
        "vs" not in q and
        "versus" not in q
    ):
        return {
            "mode": "direct_text",
            "text": (
                f"Your question is ambiguous. Cluster {cluster_ids_local[0]} is different compared to which cluster?\n"
                f"You can ask:\n"
                f"- Compare cluster {cluster_ids_local[0]} and cluster 1\n"
                f"- Summarize cluster {cluster_ids_local[0]}\n"
                f"- What features characterize cluster {cluster_ids_local[0]}?\n"
                f"- Compare to cluster 2"
            ),
            "pending_compare_cluster": cluster_ids_local[0]
        }

    if len(cluster_ids_local) >= 2 and any(x in q for x in ["compare", "vs", "versus", "different"]):
        return {
            "mode": "direct_tool",
            "tool_name": "compare_clusters",
            "tool_input": {
                "cluster_id_1": cluster_ids_local[0],
                "cluster_id_2": cluster_ids_local[1]
            }
        }

    if len(cluster_ids_local) == 1 and any(p in q for p in [
        "interesting analysis",
        "deeper analysis",
        "what else can you do",
        "tell me more"
    ]):
        return {
            "mode": "direct_tool",
            "tool_name": "cluster_deep_dive",
            "tool_input": {"cluster_id": cluster_ids_local[0]}
        }

    if (
        len(cluster_ids_local) == 1 and
        any(phrase in q for phrase in [
            "list of bridges",
            "list the bridges",
            "which bridges are in",
            "bridges in cluster",
            "show bridges in cluster",
            "give me the bridges in cluster"
        ])
    ):
        return {
            "mode": "direct_tool",
            "tool_name": "bridges_in_cluster",
            "tool_input": {"cluster_id": cluster_ids_local[0]}
        }

    if any(phrase in q for phrase in [
        "show dataset",
        "inspect dataset",
        "inspect data",
        "browse dataset",
        "browse data",
        "show rows",
        "show table",
        "explore dataset",
        "let me inspect the data"
    ]):
        return {
            "mode": "direct_tool",
            "tool_name": "browse_dataset_rows",
            "tool_input": {"offset": 0, "limit": 100}
        }

    return {"mode": "bedrock"}


def get_tool_config():
    return {
        "tools": [
            {
                "toolSpec": {
                    "name": "overall_summary",
                    "description": "Summarize the overall bridge deterioration dataset and cluster results.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {}
                        }
                    }
                }
            },
            {
                "toolSpec": {
                    "name": "bridge_profile",
                    "description": "Get a bridge profile for a specific bridge ID.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "bridge_id": {"type": "string"}
                            },
                            "required": ["bridge_id"]
                        }
                    }
                }
            },
            {
                "toolSpec": {
                    "name": "bridge_trend",
                    "description": "Get the Bridge Health Index trend over time for one bridge.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "bridge_id": {"type": "string"}
                            },
                            "required": ["bridge_id"]
                        }
                    }
                }
            },
            {
                "toolSpec": {
                    "name": "compare_bridges",
                    "description": "Compare two bridges using latest BHI and deterioration slope.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "bridge_id_1": {"type": "string"},
                                "bridge_id_2": {"type": "string"}
                            },
                            "required": ["bridge_id_1", "bridge_id_2"]
                        }
                    }
                }
            },
            {
                "toolSpec": {
                    "name": "cluster_summary",
                    "description": "Summarize a given cluster.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "cluster_id": {"type": "integer"}
                            },
                            "required": ["cluster_id"]
                        }
                    }
                }
            },
            {
                "toolSpec": {
                    "name": "compare_clusters",
                    "description": "Compare two clusters using average bridge characteristics and deterioration behavior.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "cluster_id_1": {"type": "integer"},
                                "cluster_id_2": {"type": "integer"}
                            },
                            "required": ["cluster_id_1", "cluster_id_2"]
                        }
                    }
                }
            },
            {
                "toolSpec": {
                    "name": "cluster_pca_drivers",
                    "description": "Compute PCA for a selected cluster and return the strongest PC1 feature loadings that characterize that cluster.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "cluster_id": {"type": "integer"},
                                "top_n": {"type": "integer"}
                            },
                            "required": ["cluster_id"]
                        }
                    }
                }
            },
            {
                "toolSpec": {
                    "name": "cluster_deep_dive",
                    "description": "Run a deeper analysis for one cluster, including distributions, deterioration extremes, and representative bridges.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "cluster_id": {"type": "integer"}
                            },
                            "required": ["cluster_id"]
                        }
                    }
                }
            },
            {
                "toolSpec": {
                    "name": "bridges_in_cluster",
                    "description": "Return the list of bridge IDs that belong to a specific cluster.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "cluster_id": {"type": "integer"}
                            },
                            "required": ["cluster_id"]
                        }
                    }
                }
            },
            {
                "toolSpec": {
                    "name": "top_deteriorating_bridges",
                    "description": "Return the fastest deteriorating bridges based on linear slope.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "top_n": {"type": "integer"}
                            }
                        }
                    }
                }
            },
            {
                "toolSpec": {
                    "name": "top_best_bridges_year",
                    "description": "Return the best bridges in a given year using overall BHI.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "year": {"type": "integer"},
                                "top_n": {"type": "integer"}
                            },
                            "required": ["year"]
                        }
                    }
                }
            },
            {
                "toolSpec": {
                    "name": "top_worst_bridges_year",
                    "description": "Return the worst bridges in a given year using overall BHI.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "year": {"type": "integer"},
                                "top_n": {"type": "integer"}
                            },
                            "required": ["year"]
                        }
                    }
                }
            },
            {
                "toolSpec": {
                    "name": "dataset_schema",
                    "description": "Show the dataset columns, data types, missing counts, and sample values.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "max_sample_values": {"type": "integer"}
                            }
                        }
                    }
                }
            },
            {
                "toolSpec": {
                    "name": "inspect_column",
                    "description": "Inspect one dataset column and show its type, missing values, summary, and sample or unique values.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "column_name": {"type": "string"},
                                "max_unique": {"type": "integer"}
                            },
                            "required": ["column_name"]
                        }
                    }
                }
            },
            {
                "toolSpec": {
                    "name": "preview_dataset",
                    "description": "Show the first rows of the dataset for inspection.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "n_rows": {"type": "integer"}
                            }
                        }
                    }
                }
            },
            {
                "toolSpec": {
                    "name": "browse_dataset_rows",
                    "description": "Browse rows of the dataset with optional offset, limit, and selected columns.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "offset": {"type": "integer"},
                                "limit": {"type": "integer"},
                                "columns": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            }
                        }
                    }
                }
            },
            {
                "toolSpec": {
                    "name": "forecast_bridge_20_years",
                    "description": "Forecast the next years of overall BHI for a specific bridge using the forecasting methodology.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "bridge_id": {"type": "string"},
                                "forecast_horizon": {"type": "integer"}
                            },
                            "required": ["bridge_id"]
                        }
                    }
                }
            },
            {
                "toolSpec": {
                    "name": "python_analysis",
                    "description": "Generate and run restricted Python analysis on the loaded bridge dataframes when existing tools are insufficient.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "user_request": {"type": "string"}
                            },
                            "required": ["user_request"]
                        }
                    }
                }
            }
        ]
    }

# ---------------------------
# Tool execution
# ---------------------------
def execute_tool(tool_name, tool_input):
    if tool_name == "overall_summary":
        return overall_dataset_summary()

    if tool_name == "bridge_profile":
        bridge_id = tool_input["bridge_id"]
        matched = find_best_bridge_match(bridge_id)
        return {
            "text": get_bridge_profile(bridge_id),
            "bridge_ids": [matched] if matched else None,
            "label": "bridge_profile_single"
        }

    if tool_name == "bridge_trend":
        bridge_id = tool_input["bridge_id"]
        matched = find_best_bridge_match(bridge_id)
        return {
            "text": get_bridge_trend(bridge_id),
            "bridge_id": bridge_id,
            "show_trend_chart": True,
            "bridge_ids": [matched] if matched else None,
            "label": "bridge_trend_single"
        }

    if tool_name == "compare_bridges":
        bridge_id_1 = tool_input["bridge_id_1"]
        bridge_id_2 = tool_input["bridge_id_2"]
        matched1 = find_best_bridge_match(bridge_id_1)
        matched2 = find_best_bridge_match(bridge_id_2)
        return {
            "text": compare_two_bridges(bridge_id_1, bridge_id_2),
            "bridge_id_1": bridge_id_1,
            "bridge_id_2": bridge_id_2,
            "show_compare_chart": True,
            "bridge_ids": [b for b in [matched1, matched2] if b is not None],
            "label": "compare_two_bridges"
        }

    if tool_name == "cluster_summary":
        cluster_id = int(tool_input["cluster_id"])
        return {
            "text": get_cluster_summary(cluster_id),
            "cluster_id": cluster_id,
            "cluster_ids": [cluster_id],
            "label": "cluster_summary",
            "show_cluster_chart": True
        }

    if tool_name == "compare_clusters":
        cluster_id_1 = int(tool_input["cluster_id_1"])
        cluster_id_2 = int(tool_input["cluster_id_2"])
        return {
            "text": compare_two_clusters(cluster_id_1, cluster_id_2),
            "cluster_id_1": cluster_id_1,
            "cluster_id_2": cluster_id_2,
            "show_compare_clusters_chart": True
        }

    if tool_name == "cluster_pca_drivers":
        cluster_id = int(tool_input["cluster_id"])
        top_n = int(tool_input.get("top_n", 8))
        result = get_cluster_pca_drivers(cluster_id, top_n=top_n)
        result["cluster_ids"] = [cluster_id]
        result["label"] = "cluster_pca_drivers"
        return result

    if tool_name == "cluster_deep_dive":
        cluster_id = int(tool_input["cluster_id"])
        return get_cluster_deep_dive(cluster_id)

    if tool_name == "bridges_in_cluster":
        cluster_id = int(tool_input["cluster_id"])
        return get_bridges_in_cluster(cluster_id)

    if tool_name == "top_deteriorating_bridges":
        top_n = int(tool_input.get("top_n", 5))
        return get_top_deteriorating_bridges(top_n=top_n)

    if tool_name == "top_best_bridges_year":
        year = int(tool_input["year"])
        top_n = int(tool_input.get("top_n", 5))
        return get_top_best_bridges(year, top_n=top_n)

    if tool_name == "top_worst_bridges_year":
        year = int(tool_input["year"])
        top_n = int(tool_input.get("top_n", 5))
        return get_top_worst_bridges(year, top_n=top_n)

    if tool_name == "dataset_schema":
        max_sample_values = int(tool_input.get("max_sample_values", 5))
        return get_dataset_schema(max_sample_values=max_sample_values)

    if tool_name == "inspect_column":
        column_name = tool_input["column_name"]
        max_unique = int(tool_input.get("max_unique", 20))
        return inspect_column(column_name=column_name, max_unique=max_unique)

    if tool_name == "preview_dataset":
        n_rows = int(tool_input.get("n_rows", 10))
        return preview_dataset(n_rows=n_rows)

    if tool_name == "browse_dataset_rows":
        offset = int(tool_input.get("offset", 0))
        limit = int(tool_input.get("limit", 25))
        columns = tool_input.get("columns", None)
        return browse_dataset_rows(offset=offset, limit=limit, columns=columns)

    if tool_name == "forecast_bridge_20_years":
        bridge_id = tool_input["bridge_id"]
        forecast_horizon = int(tool_input.get("forecast_horizon", 20))
        return forecast_bridge_20_years(bridge_id=bridge_id, forecast_horizon=forecast_horizon)

    if tool_name == "python_analysis":
        user_request = tool_input["user_request"]
        return run_python_analysis(user_request)

    return {"text": f"Unknown tool: {tool_name}"}

# ---------------------------
# Bedrock conversation
# ---------------------------
def ask_bedrock_with_tools(user_prompt):
    messages = [
        {
            "role": "user",
            "content": [{"text": user_prompt}]
        }
    ]

    pending_chart = None
    pending_summary_df = None
    pending_cluster_df = None
    pending_pc1_table = None
    pending_schema_df = None
    pending_column_df = None
    pending_values_df = None
    pending_preview_df = None
    pending_browse_df = None
    pending_analysis_df = None
    pending_generated_code = None
    pending_execution_steps = None
    pending_stdout = None
    pending_bridge_ids = None
    pending_cluster_ids = None
    pending_label = None

    loops = 0
    max_loops = 6

    try:
        response = bedrock.converse(
            modelId=BEDROCK_MODEL_ID,
            system=[{"text": SYSTEM_PROMPT}],
            messages=messages,
            toolConfig=get_tool_config()
        )
    except (BotoCoreError, ClientError) as e:
        return {
            "text": f"Bedrock request failed: {e}",
            "chart": None,
            "summary_df": None,
            "cluster_df": None,
            "pc1_table": None,
            "schema_df": None,
            "column_df": None,
            "values_df": None,
            "preview_df": None,
            "browse_df": None,
            "analysis_df": None,
            "generated_code": None,
            "execution_steps": None,
            "stdout": None,
            "bridge_ids": None,
            "cluster_ids": None,
            "label": None
        }
    except Exception as e:
        return {
            "text": f"Unexpected Bedrock error: {e}",
            "chart": None,
            "summary_df": None,
            "cluster_df": None,
            "pc1_table": None,
            "schema_df": None,
            "column_df": None,
            "values_df": None,
            "preview_df": None,
            "browse_df": None,
            "analysis_df": None,
            "generated_code": None,
            "execution_steps": None,
            "stdout": None,
            "bridge_ids": None,
            "cluster_ids": None,
            "label": None
        }

    while loops < max_loops:
        loops += 1

        output_message = response["output"]["message"]
        stop_reason = response.get("stopReason", "")
        messages.append(output_message)

        if stop_reason == "end_turn":
            final_text = extract_text_from_content_blocks(output_message["content"])
            if (
                not final_text and
                pending_summary_df is None and
                pending_cluster_df is None and
                pending_pc1_table is None and
                pending_schema_df is None and
                pending_column_df is None and
                pending_values_df is None and
                pending_preview_df is None and
                pending_browse_df is None and
                pending_analysis_df is None and
                pending_generated_code is None and
                pending_execution_steps is None and
                pending_stdout is None
            ):
                final_text = "I could not generate a final answer."

            return {
                "text": final_text,
                "chart": pending_chart,
                "summary_df": pending_summary_df,
                "cluster_df": pending_cluster_df,
                "pc1_table": pending_pc1_table,
                "schema_df": pending_schema_df,
                "column_df": pending_column_df,
                "values_df": pending_values_df,
                "preview_df": pending_preview_df,
                "browse_df": pending_browse_df,
                "analysis_df": pending_analysis_df,
                "generated_code": pending_generated_code,
                "execution_steps": pending_execution_steps,
                "stdout": pending_stdout,
                "bridge_ids": pending_bridge_ids,
                "cluster_ids": pending_cluster_ids,
                "label": pending_label
            }

        if stop_reason == "tool_use":
            tool_result_content = []

            for block in output_message["content"]:
                if "toolUse" not in block:
                    continue

                tool_use = block["toolUse"]
                tool_name = tool_use["name"]
                tool_input = tool_use["input"]
                tool_use_id = tool_use["toolUseId"]

                result = execute_tool(tool_name, tool_input)

                if "pc1_table" in result:
                    pending_pc1_table = result["pc1_table"]

                if "summary_df" in result:
                    pending_summary_df = result["summary_df"]

                if "cluster_df" in result:
                    pending_cluster_df = result["cluster_df"]

                if "schema_df" in result:
                    pending_schema_df = result["schema_df"]

                if "column_df" in result:
                    pending_column_df = result["column_df"]

                if "values_df" in result:
                    pending_values_df = result["values_df"]

                if "preview_df" in result:
                    pending_preview_df = result["preview_df"]

                if "browse_df" in result:
                    pending_browse_df = result["browse_df"]

                if "analysis_df" in result:
                    pending_analysis_df = result["analysis_df"]

                if "generated_code" in result:
                    pending_generated_code = result["generated_code"]

                if "execution_steps" in result:
                    pending_execution_steps = result["execution_steps"]

                if "stdout" in result:
                    pending_stdout = result["stdout"]

                if "bridge_ids" in result:
                    pending_bridge_ids = result["bridge_ids"]

                if "cluster_ids" in result:
                    pending_cluster_ids = result["cluster_ids"]

                if "label" in result:
                    pending_label = result["label"]

                if result.get("show_trend_chart"):
                    pending_chart = {
                        "type": "trend",
                        "bridge_id": result["bridge_id"]
                    }

                if result.get("show_compare_chart"):
                    pending_chart = {
                        "type": "compare",
                        "bridge_id_1": result["bridge_id_1"],
                        "bridge_id_2": result["bridge_id_2"]
                    }

                if result.get("show_cluster_chart"):
                    pending_chart = {
                        "type": "cluster",
                        "cluster_id": result["cluster_id"]
                    }

                if result.get("show_compare_clusters_chart"):
                    pending_chart = {
                        "type": "compare_clusters",
                        "cluster_id_1": result["cluster_id_1"],
                        "cluster_id_2": result["cluster_id_2"]
                    }

                json_safe_result = make_json_safe(result)

                tool_result_content.append({
                    "toolResult": {
                        "toolUseId": tool_use_id,
                        "content": [{"json": json_safe_result}]
                    }
                })

            if not tool_result_content:
                return {
                    "text": "The model requested tool use, but no valid tool call was returned.",
                    "chart": None,
                    "summary_df": None,
                    "cluster_df": None,
                    "pc1_table": None,
                    "schema_df": None,
                    "column_df": None,
                    "values_df": None,
                    "preview_df": None,
                    "browse_df": None,
                    "analysis_df": None,
                    "generated_code": None,
                    "execution_steps": None,
                    "stdout": None,
                    "bridge_ids": None,
                    "label": None
                }

            messages.append({
                "role": "user",
                "content": tool_result_content
            })

            try:
                response = bedrock.converse(
                    modelId=BEDROCK_MODEL_ID,
                    system=[{"text": SYSTEM_PROMPT}],
                    messages=messages,
                    toolConfig=get_tool_config()
                )
            except (BotoCoreError, ClientError) as e:
                fallback_text = "I ran the requested analysis, but the follow-up model response failed."
                if pending_analysis_df is not None or pending_summary_df is not None or pending_pc1_table is not None:
                    fallback_text += " The computed result is still available below."
                else:
                    fallback_text += f" Error: {e}"

                return {
                    "text": fallback_text,
                    "chart": pending_chart,
                    "summary_df": pending_summary_df,
                    "cluster_df": pending_cluster_df,
                    "pc1_table": pending_pc1_table,
                    "schema_df": pending_schema_df,
                    "column_df": pending_column_df,
                    "values_df": pending_values_df,
                    "preview_df": pending_preview_df,
                    "browse_df": pending_browse_df,
                    "analysis_df": pending_analysis_df,
                    "generated_code": pending_generated_code,
                    "execution_steps": pending_execution_steps,
                    "stdout": pending_stdout,
                    "bridge_ids": pending_bridge_ids,
                    "cluster_ids": pending_cluster_ids,
                    "label": pending_label
                }
            except Exception as e:
                return {
                    "text": f"Unexpected Bedrock follow-up error: {e}",
                    "chart": None,
                    "summary_df": pending_summary_df,
                    "cluster_df": pending_cluster_df,
                    "pc1_table": pending_pc1_table,
                    "schema_df": pending_schema_df,
                    "column_df": pending_column_df,
                    "values_df": pending_values_df,
                    "preview_df": pending_preview_df,
                    "browse_df": pending_browse_df,
                    "analysis_df": pending_analysis_df,
                    "generated_code": pending_generated_code,
                    "execution_steps": pending_execution_steps,
                    "stdout": pending_stdout,
                    "bridge_ids": pending_bridge_ids,
                    "cluster_ids": pending_cluster_ids,
                    "label": pending_label
                }

            continue

        return {
            "text": "I could not complete the request.",
            "chart": None,
            "summary_df": pending_summary_df,
            "cluster_df": pending_cluster_df,
            "pc1_table": pending_pc1_table,
            "schema_df": pending_schema_df,
            "column_df": pending_column_df,
            "values_df": pending_values_df,
            "preview_df": pending_preview_df,
            "browse_df": pending_browse_df,
            "analysis_df": pending_analysis_df,
            "generated_code": pending_generated_code,
            "execution_steps": pending_execution_steps,
            "stdout": pending_stdout,
            "bridge_ids": pending_bridge_ids,
            "cluster_ids": pending_cluster_ids,
            "label": pending_label
        }

    return {
        "text": "The Bedrock tool loop reached its limit.",
        "chart": None,
        "summary_df": pending_summary_df,
        "cluster_df": pending_cluster_df,
        "pc1_table": pending_pc1_table,
        "schema_df": pending_schema_df,
        "column_df": pending_column_df,
        "values_df": pending_values_df,
        "preview_df": pending_preview_df,
        "browse_df": pending_browse_df,
        "analysis_df": pending_analysis_df,
        "generated_code": pending_generated_code,
        "execution_steps": pending_execution_steps,
        "stdout": pending_stdout,
        "bridge_ids": pending_bridge_ids,
        "label": pending_label
    }

# ---------------------------
# Main answer router
# ---------------------------
def answer_question(question):
    # 1) Handle general related follow-up to prior bridge subset
    if has_bridge_context() and is_contextual_followup(question):
        prior_bridge_ids = st.session_state.last_result_context.get("bridge_ids")
        subset_result = analyze_bridge_subset(question, prior_bridge_ids)
        if subset_result is not None:
            return {
                "text": subset_result.get("text"),
                "figure": subset_result.get("figure"),
                "summary_df": subset_result.get("summary_df"),
                "cluster_df": subset_result.get("cluster_df"),
                "pc1_table": subset_result.get("pc1_table"),
                "schema_df": subset_result.get("schema_df"),
                "column_df": subset_result.get("column_df"),
                "values_df": subset_result.get("values_df"),
                "preview_df": subset_result.get("preview_df"),
                "browse_df": subset_result.get("browse_df"),
                "analysis_df": subset_result.get("analysis_df"),
                "generated_code": subset_result.get("generated_code"),
                "execution_steps": subset_result.get("execution_steps"),
                "stdout": subset_result.get("stdout"),
                "bridge_ids": subset_result.get("bridge_ids"),
                "cluster_ids": subset_result.get("cluster_ids"),
                "label": subset_result.get("label")
            }

    # 1b) Handle related follow-up to prior cluster
    if has_cluster_context() and is_cluster_followup(question):
        prior_cluster_ids = st.session_state.last_result_context.get("cluster_ids")
        cluster_id = prior_cluster_ids[0]

        intent = resolve_cluster_followup_intent(question)

        if intent == "cluster_median_interpretation":
            fig = make_cluster_median_figure(cluster_id)
            return {
                "text": interpret_cluster_trend(cluster_id),
                "figure": fig,
                "summary_df": None,
                "cluster_df": None,
                "pc1_table": None,
                "schema_df": None,
                "column_df": None,
                "values_df": None,
                "preview_df": None,
                "browse_df": None,
                "analysis_df": None,
                "generated_code": None,
                "execution_steps": None,
                "stdout": None,
                "bridge_ids": None,
                "cluster_ids": [cluster_id],
                "label": "cluster_median_interpretation"
            }

        if intent == "cluster_fluctuation_interpretation":
            fig = make_cluster_median_figure(cluster_id)
            return {
                "text": interpret_cluster_fluctuations(cluster_id),
                "figure": fig,
                "summary_df": None,
                "cluster_df": None,
                "pc1_table": None,
                "schema_df": None,
                "column_df": None,
                "values_df": None,
                "preview_df": None,
                "browse_df": None,
                "analysis_df": None,
                "generated_code": None,
                "execution_steps": None,
                "stdout": None,
                "bridge_ids": None,
                "cluster_ids": [cluster_id],
                "label": "cluster_fluctuation_interpretation"
            }

        if intent == "cluster_pca":
            result = execute_tool("cluster_pca_drivers", {"cluster_id": cluster_id, "top_n": 8})
            fig = None
            return {
                "text": result.get("text"),
                "figure": fig,
                "summary_df": result.get("summary_df"),
                "cluster_df": result.get("cluster_df"),
                "pc1_table": result.get("pc1_table"),
                "schema_df": result.get("schema_df"),
                "column_df": result.get("column_df"),
                "values_df": result.get("values_df"),
                "preview_df": result.get("preview_df"),
                "browse_df": result.get("browse_df"),
                "analysis_df": result.get("analysis_df"),
                "generated_code": result.get("generated_code"),
                "execution_steps": result.get("execution_steps"),
                "stdout": result.get("stdout"),
                "bridge_ids": result.get("bridge_ids"),
                "cluster_ids": result.get("cluster_ids"),
                "label": result.get("label")
            }

        if intent == "cluster_trend":
            result = execute_tool("cluster_summary", {"cluster_id": cluster_id})
            fig = make_cluster_median_figure(cluster_id)
            return {
                "text": result.get("text"),
                "figure": fig,
                "summary_df": result.get("summary_df"),
                "cluster_df": result.get("cluster_df"),
                "pc1_table": result.get("pc1_table"),
                "schema_df": result.get("schema_df"),
                "column_df": result.get("column_df"),
                "values_df": result.get("values_df"),
                "preview_df": result.get("preview_df"),
                "browse_df": result.get("browse_df"),
                "analysis_df": result.get("analysis_df"),
                "generated_code": result.get("generated_code"),
                "execution_steps": result.get("execution_steps"),
                "stdout": result.get("stdout"),
                "bridge_ids": result.get("bridge_ids"),
                "cluster_ids": result.get("cluster_ids"),
                "label": result.get("label")
            }

        if intent == "cluster_summary":
            result = execute_tool("cluster_summary", {"cluster_id": cluster_id})
            fig = make_cluster_median_figure(cluster_id)
            return {
                "text": result.get("text"),
                "figure": fig,
                "summary_df": result.get("summary_df"),
                "cluster_df": result.get("cluster_df"),
                "pc1_table": result.get("pc1_table"),
                "schema_df": result.get("schema_df"),
                "column_df": result.get("column_df"),
                "values_df": result.get("values_df"),
                "preview_df": result.get("preview_df"),
                "browse_df": result.get("browse_df"),
                "analysis_df": result.get("analysis_df"),
                "generated_code": result.get("generated_code"),
                "execution_steps": result.get("execution_steps"),
                "stdout": result.get("stdout"),
                "bridge_ids": result.get("bridge_ids"),
                "cluster_ids": result.get("cluster_ids"),
                "label": result.get("label")
            }

        result = execute_tool("cluster_deep_dive", {"cluster_id": cluster_id})
        return {
            "text": result.get("text"),
            "figure": None,
            "summary_df": result.get("summary_df"),
            "cluster_df": result.get("cluster_df"),
            "pc1_table": result.get("pc1_table"),
            "schema_df": result.get("schema_df"),
            "column_df": result.get("column_df"),
            "values_df": result.get("values_df"),
            "preview_df": result.get("preview_df"),
            "browse_df": result.get("browse_df"),
            "analysis_df": result.get("analysis_df"),
            "generated_code": result.get("generated_code"),
            "execution_steps": result.get("execution_steps"),
            "stdout": result.get("stdout"),
            "bridge_ids": result.get("bridge_ids"),
            "cluster_ids": result.get("cluster_ids"),
            "label": result.get("label"),
            "forecast_explanation": result.get("forecast_explanation")
        }

    # 2) Existing cluster pending compare
    pending_base = st.session_state.pending_compare_cluster
    followup_target = extract_compare_target(question)

    if pending_base is not None and followup_target is not None:
        result = execute_tool(
            "compare_clusters",
            {
                "cluster_id_1": pending_base,
                "cluster_id_2": followup_target
            }
        )
        st.session_state.pending_compare_cluster = None

        fig = None
        if result.get("show_compare_clusters_chart"):
            fig = make_compare_clusters_figure(result["cluster_id_1"], result["cluster_id_2"])

        if result.get("label") == "bridge_20yr_forecast":
            st.session_state.last_forecast_result = {
                "forecast_explanation": result.get("forecast_explanation"),
                "execution_steps": result.get("execution_steps"),
                "bridge_ids": result.get("bridge_ids")
            }

        return {
            "text": result.get("text"),
            "figure": fig,
            "summary_df": result.get("summary_df"),
            "cluster_df": result.get("cluster_df"),
            "pc1_table": result.get("pc1_table"),
            "schema_df": result.get("schema_df"),
            "column_df": result.get("column_df"),
            "values_df": result.get("values_df"),
            "preview_df": result.get("preview_df"),
            "browse_df": result.get("browse_df"),
            "analysis_df": result.get("analysis_df"),
            "generated_code": result.get("generated_code"),
            "execution_steps": result.get("execution_steps"),
            "stdout": result.get("stdout"),
            "bridge_ids": result.get("bridge_ids"),
            "cluster_ids": result.get("cluster_ids"),
            "label": result.get("label")
        }

    # 3) Direct routing
    routed = route_question(question)

    if routed["mode"] == "forecast_explanation":
        last_forecast = st.session_state.get("last_forecast_result")
        if last_forecast:
            return {
                "text": last_forecast.get("forecast_explanation", "No saved forecast explanation is available."),
                "figure": None,
                "summary_df": None,
                "cluster_df": None,
                "pc1_table": None,
                "schema_df": None,
                "column_df": None,
                "values_df": None,
                "preview_df": None,
                "browse_df": None,
                "analysis_df": None,
                "generated_code": None,
                "execution_steps": last_forecast.get("execution_steps", []),
                "stdout": None,
                "bridge_ids": last_forecast.get("bridge_ids"),
                "cluster_ids": None,
                "label": "forecast_explanation"
            }
        return {
            "text": "Please run a forecast first, then I can explain how that projection was executed.",
            "figure": None,
            "summary_df": None,
            "cluster_df": None,
            "pc1_table": None,
            "schema_df": None,
            "column_df": None,
            "values_df": None,
            "preview_df": None,
            "browse_df": None,
            "analysis_df": None,
            "generated_code": None,
            "execution_steps": None,
            "stdout": None,
            "bridge_ids": None,
            "cluster_ids": None,
            "label": "forecast_explanation"
        }

    if routed["mode"] == "direct_text":
        st.session_state.pending_compare_cluster = routed.get("pending_compare_cluster")
        return {
            "text": routed["text"],
            "figure": None,
            "summary_df": None,
            "cluster_df": None,
            "pc1_table": None,
            "schema_df": None,
            "column_df": None,
            "values_df": None,
            "preview_df": None,
            "browse_df": None,
            "analysis_df": None,
            "generated_code": None,
            "execution_steps": None,
            "stdout": None,
            "bridge_ids": None,
            "cluster_ids": routed.get("cluster_ids"),
            "label": routed.get("label")
        }

    if routed["mode"] == "direct_tool":
        st.session_state.pending_compare_cluster = None
        result = execute_tool(routed["tool_name"], routed["tool_input"])

        fig = result.get("figure")
        if fig is None and result.get("show_cluster_chart"):
            fig = make_cluster_median_figure(result["cluster_id"])
        elif fig is None and result.get("show_compare_clusters_chart"):
            fig = make_compare_clusters_figure(result["cluster_id_1"], result["cluster_id_2"])

        return {
            "text": result.get("text"),
            "figure": fig,
            "summary_df": result.get("summary_df"),
            "cluster_df": result.get("cluster_df"),
            "pc1_table": result.get("pc1_table"),
            "schema_df": result.get("schema_df"),
            "column_df": result.get("column_df"),
            "values_df": result.get("values_df"),
            "preview_df": result.get("preview_df"),
            "browse_df": result.get("browse_df"),
            "analysis_df": result.get("analysis_df"),
            "generated_code": result.get("generated_code"),
            "execution_steps": result.get("execution_steps"),
            "stdout": result.get("stdout"),
            "bridge_ids": result.get("bridge_ids"),
            "cluster_ids": result.get("cluster_ids"),
            "label": result.get("label")
        }

    # 4) Bedrock + tool use
    st.session_state.pending_compare_cluster = None
    result = ask_bedrock_with_tools(question)
    fig = None
    chart = result.get("chart")

    if chart:
        if chart["type"] == "trend":
            fig = make_bridge_trend_figure(chart["bridge_id"])
        elif chart["type"] == "compare":
            fig = make_compare_bridges_figure(chart["bridge_id_1"], chart["bridge_id_2"])
        elif chart["type"] == "cluster":
            fig = make_cluster_median_figure(chart["cluster_id"])
        elif chart["type"] == "compare_clusters":
            fig = make_compare_clusters_figure(chart["cluster_id_1"], chart["cluster_id_2"])

    return {
        "text": result.get("text"),
        "figure": fig,
        "summary_df": result.get("summary_df"),
        "cluster_df": result.get("cluster_df"),
        "pc1_table": result.get("pc1_table"),
        "schema_df": result.get("schema_df"),
        "column_df": result.get("column_df"),
        "values_df": result.get("values_df"),
        "preview_df": result.get("preview_df"),
        "browse_df": result.get("browse_df"),
        "analysis_df": result.get("analysis_df"),
        "generated_code": result.get("generated_code"),
        "execution_steps": result.get("execution_steps"),
        "stdout": result.get("stdout"),
        "bridge_ids": result.get("bridge_ids"),
        "label": result.get("label")
    }

# ---------------------------
# Sidebar
# ---------------------------
sample_df = bridge_summary.dropna(subset=["deterioration_slope_per_year"]).copy()

example_1 = sample_df.sample(1)["STRUCTURE_NUMBER_008"].iloc[0]
example_2 = sample_df[
    sample_df["Cluster"] != sample_df[sample_df["STRUCTURE_NUMBER_008"] == example_1]["Cluster"].iloc[0]
].sample(1)["STRUCTURE_NUMBER_008"].iloc[0]
example_3 = sample_df.sort_values("deterioration_slope_per_year").iloc[0]["STRUCTURE_NUMBER_008"]

with st.sidebar:
    st.subheader("Dataset")
    st.write("This dataset contains 1,378 bridges.")
    st.write(f"Years: {min(years_available)}–{max(years_available)}")
    st.caption("Use a bridge ID from STRUCTURE_NUMBER_008")
    st.write("Example questions:")
    st.markdown(f"""
    - Give me an overview of the bridge deterioration dataset
    - Show trend for bridge {example_1}
    - Compare bridge {example_1} and {example_2}
    - Summarize cluster 2
    - Compare cluster 2 and cluster 3
    - What features characterize cluster 5?
    - What columns are in the dataset?
    - Show me the first 10 rows of the dataset
    - Browse dataset rows
    - Show the fastest deteriorating bridges
    - Show the 5 worst bridges in 2020
    - Show the 5 best bridges in 2020
    - Give me the profile for bridge {example_3}
    """)

    open_explorer = st.checkbox("Open dataset explorer")

if open_explorer:
    render_paginated_dataframe(
        static_df,
        key_prefix="main_dataset",
        title="Full Dataset Explorer"
    )

# ---------------------------
# Chat history
# ---------------------------
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message.get("content"):
            st.write(message["content"])

        if "pc1_table" in message and message["pc1_table"] is not None:
            st.subheader("PC1 Loadings")
            st.dataframe(pd.DataFrame(message["pc1_table"]), use_container_width=True)

        if "summary_df" in message and message["summary_df"] is not None:
            st.subheader("Dataset Summary")
            st.dataframe(pd.DataFrame(message["summary_df"]), use_container_width=True)

        if "cluster_df" in message and message["cluster_df"] is not None:
            st.subheader("Cluster Distribution")
            st.dataframe(pd.DataFrame(message["cluster_df"]), use_container_width=True)

        if "schema_df" in message and message["schema_df"] is not None:
            st.subheader("Dataset Schema")
            st.dataframe(pd.DataFrame(message["schema_df"]), use_container_width=True)

        if "column_df" in message and message["column_df"] is not None:
            st.subheader("Column Details")
            st.dataframe(pd.DataFrame(message["column_df"]), use_container_width=True)

        if "values_df" in message and message["values_df"] is not None:
            st.subheader("Column Values")
            st.dataframe(pd.DataFrame(message["values_df"]), use_container_width=True)

        if "preview_df" in message and message["preview_df"] is not None:
            st.subheader("Dataset Preview")
            st.dataframe(pd.DataFrame(message["preview_df"]), use_container_width=True)

        if "browse_df" in message and message["browse_df"] is not None:
            render_paginated_dataframe(
                pd.DataFrame(message["browse_df"]),
                key_prefix=f"history_browse_{idx}",
                title="Dataset Rows"
            )

        if "analysis_df" in message and message["analysis_df"] is not None:
            st.subheader("Analysis Results")
            st.dataframe(pd.DataFrame(message["analysis_df"]), use_container_width=True)

        if "forecast_explanation" in message and message["forecast_explanation"] is not None:
            st.subheader("Forecast Execution Summary")
            st.write(message["forecast_explanation"])

        if "execution_steps" in message and message["execution_steps"] is not None:
            with st.expander("Analysis Steps"):
                for step in message["execution_steps"]:
                    st.write(f"- {step}")

        if "generated_code" in message and message["generated_code"] is not None:
            with st.expander("Generated Python Code"):
                st.code(message["generated_code"], language="python")

        if "stdout" in message and message["stdout"]:
            with st.expander("Execution Log"):
                st.text(message["stdout"])

        if "figure_key" in message and message["figure_key"] in st.session_state:
            st.pyplot(st.session_state[message["figure_key"]])

# ---------------------------
# Chat input
# ---------------------------
user_prompt = st.chat_input("Ask a question about the bridge dataset...")

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    with st.chat_message("user"):
        st.write(user_prompt)

    result = answer_question(user_prompt)

    update_last_result_context(user_prompt, result)

    assistant_message = {
        "role": "assistant",
        "content": result.get("text")
    }

    if result.get("pc1_table") is not None:
        assistant_message["pc1_table"] = result["pc1_table"].to_dict(orient="records")

    if result.get("summary_df") is not None:
        assistant_message["summary_df"] = result["summary_df"].to_dict(orient="records")

    if result.get("cluster_df") is not None:
        assistant_message["cluster_df"] = result["cluster_df"].to_dict(orient="records")

    if result.get("schema_df") is not None:
        assistant_message["schema_df"] = result["schema_df"].to_dict(orient="records")

    if result.get("column_df") is not None:
        assistant_message["column_df"] = result["column_df"].to_dict(orient="records")

    if result.get("values_df") is not None:
        assistant_message["values_df"] = result["values_df"].to_dict(orient="records")

    if result.get("preview_df") is not None:
        assistant_message["preview_df"] = result["preview_df"].to_dict(orient="records")

    if result.get("browse_df") is not None:
        assistant_message["browse_df"] = result["browse_df"].to_dict(orient="records")

    if result.get("analysis_df") is not None:
        assistant_message["analysis_df"] = result["analysis_df"].to_dict(orient="records")

    if result.get("forecast_explanation") is not None:
        assistant_message["forecast_explanation"] = result["forecast_explanation"]

    if result.get("generated_code") is not None:
        assistant_message["generated_code"] = result["generated_code"]

    if result.get("execution_steps") is not None:
        assistant_message["execution_steps"] = result["execution_steps"]

    if result.get("stdout") is not None:
        assistant_message["stdout"] = result["stdout"]

    if result.get("bridge_ids") is not None:
        assistant_message["bridge_ids"] = result["bridge_ids"]

    if result.get("cluster_ids") is not None:
        assistant_message["cluster_ids"] = result["cluster_ids"]

    if result.get("label") is not None:
        assistant_message["label"] = result["label"]

    with st.chat_message("assistant"):
        if result.get("text"):
            st.write(result["text"])

        if result.get("pc1_table") is not None:
            st.subheader("PC1 Loadings")
            st.dataframe(result["pc1_table"], use_container_width=True)

        if result.get("summary_df") is not None:
            st.subheader("Dataset Summary")
            st.dataframe(result["summary_df"], use_container_width=True)

        if result.get("cluster_df") is not None:
            st.subheader("Cluster Distribution")
            st.dataframe(result["cluster_df"], use_container_width=True)

        if result.get("schema_df") is not None:
            st.subheader("Dataset Schema")
            st.dataframe(result["schema_df"], use_container_width=True)

        if result.get("column_df") is not None:
            st.subheader("Column Details")
            st.dataframe(result["column_df"], use_container_width=True)

        if result.get("values_df") is not None:
            st.subheader("Column Values")
            st.dataframe(result["values_df"], use_container_width=True)

        if result.get("preview_df") is not None:
            st.subheader("Dataset Preview")
            st.dataframe(result["preview_df"], use_container_width=True)

        if result.get("browse_df") is not None:
            render_paginated_dataframe(
                result["browse_df"],
                key_prefix="current_browse",
                title="Dataset Rows"
            )

        if result.get("analysis_df") is not None:
            st.subheader("Analysis Results")
            st.dataframe(result["analysis_df"], use_container_width=True)

        if result.get("forecast_explanation") is not None:
            st.subheader("Forecast Execution Summary")
            st.write(result["forecast_explanation"])

        if result.get("execution_steps") is not None:
            with st.expander("Analysis Steps"):
                for step in result["execution_steps"]:
                    st.write(f"- {step}")

        if result.get("generated_code") is not None:
            with st.expander("Generated Python Code"):
                st.code(result["generated_code"], language="python")

        if result.get("stdout"):
            with st.expander("Execution Log"):
                st.text(result["stdout"])

        if result.get("figure") is not None:
            figure_key = f"fig_{len(st.session_state.messages)}"
            st.session_state[figure_key] = result["figure"]
            assistant_message["figure_key"] = figure_key
            st.pyplot(result["figure"])

    st.session_state.messages.append(assistant_message)
