import re
import os
import io
import json
import boto3
import traceback
import contextlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from botocore.exceptions import BotoCoreError, ClientError
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress

# =========================================================
# CONFIG
# =========================================================
N_CLUSTERS = 6
STATIC_FILE = "STEEL_Bridges.csv"
PROJECTION_HORIZON = 20

# =========================================================
# AWS CONFIG
# =========================================================
AWS_REGION = st.secrets.get("AWS_REGION", "us-east-2")
BEDROCK_MODEL_ID = st.secrets.get("BEDROCK_MODEL_ID", "us.amazon.nova-lite-v1:0")
AWS_ACCESS_KEY_ID = st.secrets.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = st.secrets.get("AWS_SECRET_ACCESS_KEY")

# =========================================================
# PAGE SETUP
# =========================================================
st.set_page_config(page_title="Bridge Deterioration Analysis", layout="wide")
st.title("Bridge Deterioration Analysis")
st.caption("Amazon Bedrock + Streamlit + Bridge Health Index time-series clustering + 20-year projection")

# =========================================================
# BEDROCK CLIENT
# =========================================================
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

# =========================================================
# FILE READER
# =========================================================
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

# =========================================================
# LOAD RAW DATA
# =========================================================
@st.cache_data
def load_data():
    if not os.path.exists(STATIC_FILE):
        raise FileNotFoundError(f"Missing required file: {STATIC_FILE}")

    static_df = read_table_file(STATIC_FILE)
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
        raise ValueError(f"Static file is missing required columns: {missing_cols}")

    return static_df

# =========================================================
# HELPERS
# =========================================================
def clean_year_built(series):
    s = pd.to_numeric(series, errors="coerce")
    s[(s < 1800) | (s > 2100)] = np.nan
    return s

def strip_thinking_blocks(text: str):
    if not text:
        return text
    return re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

def extract_text_from_content_blocks(content_blocks):
    parts = []
    for block in content_blocks:
        if isinstance(block, dict):
            if "text" in block:
                parts.append(block["text"])
            elif block.get("type") == "text" and "text" in block:
                parts.append(block["text"])
    return "\n".join(parts).strip()

def find_best_bridge_match(bridge_id: str, bridge_ids):
    if not bridge_id:
        return None

    candidate = str(bridge_id).strip()

    exact_matches = [b for b in bridge_ids if b == candidate]
    if exact_matches:
        return exact_matches[0]

    contains_matches = [b for b in bridge_ids if candidate in b]
    if contains_matches:
        return contains_matches[0]

    reverse_contains = [b for b in bridge_ids if b in candidate]
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

def extract_cluster_ids(text: str):
    matches = re.findall(r"cluster\s+(\d+)", text.lower())
    return [int(x) for x in matches]

def extract_bridge_ids_from_question(question: str):
    # tries to capture common bridge patterns after the word bridge
    matches = re.findall(r"bridge\s+([A-Za-z0-9\-_]+)", question, flags=re.IGNORECASE)
    return matches

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

    st.dataframe(page_df, use_container_width=True, height=500)

# =========================================================
# PREPARE ANALYSIS
# =========================================================
@st.cache_data
def prepare_analysis(static_df, n_clusters=N_CLUSTERS, projection_horizon=PROJECTION_HORIZON):
    data = static_df.copy()
    data.columns = data.columns.str.strip()

    numeric_cols = [
        "Year of Data",
        "Bridge Health Index (Overall)",
        "Bridge Health Index (Deck)",
        "Bridge Health Index (Super)",
        "Bridge Health Index (Sub)",
        "Minimum Recorded Temperature,°F",
        "Minimum Average Temperature,°F",
        "Overall Average Temperature,°F",
        "Maximum Average Temperature,°F",
        "Maximum Recorded Temperature,°F",
        "Temperature Range,°F",
        "Yearly Precipitation Total, In.",
        "MIN_VERT_CLR_010",
        "YEAR_BUILT_027",
        "TRAFFIC_LANES_ON_028A",
        "ADT_029",
        "YEAR_ADT_030",
        "MAIN_UNIT_SPANS_045",
        "MAX_SPAN_LEN_MT_048",
        "STRUCTURE_LEN_MT_049",
        "DECK_WIDTH_MT_052",
        "DECK_COND_058",
        "SUPERSTRUCTURE_COND_059",
        "SUBSTRUCTURE_COND_060",
        "CHANNEL_COND_061",
        "CULVERT_COND_062",
        "OPERATING_RATING_064",
        "INVENTORY_RATING_066",
        "STRUCTURAL_EVAL_067",
        "YEAR_RECONSTRUCTED_106",
        "PERCENT_ADT_TRUCK_109",
        "FUTURE_ADT_114",
        "YEAR_OF_FUTURE_ADT_115",
        "SUFFICIENCY_RATING",
        "LOWEST_RATING",
        "DECK_AREA"
    ]

    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=["STRUCTURE_NUMBER_008", "Year of Data"]).copy()
    data["Year of Data"] = data["Year of Data"].astype(int)
    data = data.sort_values(["STRUCTURE_NUMBER_008", "Year of Data"]).reset_index(drop=True)

    if "Bridge Health Index (Overall)" not in data.columns:
        data["Bridge Health Index (Overall)"] = np.nan

    mask_missing_bhi = data["Bridge Health Index (Overall)"].isna()
    if all(c in data.columns for c in ["Bridge Health Index (Deck)", "Bridge Health Index (Super)", "Bridge Health Index (Sub)"]):
        data.loc[mask_missing_bhi, "Bridge Health Index (Overall)"] = data.loc[
            mask_missing_bhi,
            ["Bridge Health Index (Deck)", "Bridge Health Index (Super)", "Bridge Health Index (Sub)"]
        ].mean(axis=1)

    def complete_timeseries(group):
        group = group.sort_values("Year of Data").copy()
        min_year = int(group["Year of Data"].min())
        max_year = int(group["Year of Data"].max())

        full_years = pd.DataFrame({"Year of Data": np.arange(min_year, max_year + 1)})
        observed_years = set(group["Year of Data"].tolist())

        merged = pd.merge(full_years, group, on="Year of Data", how="left")
        merged["STRUCTURE_NUMBER_008"] = merged["STRUCTURE_NUMBER_008"].ffill().bfill()

        for col in [
            "Bridge Health Index (Overall)",
            "Bridge Health Index (Deck)",
            "Bridge Health Index (Super)",
            "Bridge Health Index (Sub)"
        ]:
            if col in merged.columns:
                merged[col] = merged[col].interpolate(method="linear", limit_direction="both")

        merged["DATA_TYPE"] = merged["Year of Data"].apply(
            lambda y: "Observed" if y in observed_years else "Interpolated"
        )

        return merged

    df_full_interpolated = (
        data.groupby("STRUCTURE_NUMBER_008", group_keys=False)
        .apply(complete_timeseries)
        .reset_index(drop=True)
    )

    bridge_counts = df_full_interpolated.groupby("STRUCTURE_NUMBER_008")["Year of Data"].count()
    structure_ids_20 = bridge_counts[bridge_counts >= 20].index.tolist()

    df_filtered = df_full_interpolated[
        df_full_interpolated["STRUCTURE_NUMBER_008"].isin(structure_ids_20)
    ].copy()

    df_check = df_filtered[
        ["STRUCTURE_NUMBER_008", "Year of Data", "Bridge Health Index (Overall)"]
    ].dropna().copy()

    df_check = df_check.sort_values(["STRUCTURE_NUMBER_008", "Year of Data"])
    unchanged_bridges = []

    for bridge_id, group in df_check.groupby("STRUCTURE_NUMBER_008"):
        group = group.sort_values("Year of Data")
        years = group["Year of Data"].tolist()
        values = group["Bridge Health Index (Overall)"].tolist()

        for i in range(len(values) - 19):
            if all(v == values[i] for v in values[i:i + 20]) and \
               all(y2 - y1 == 1 for y1, y2 in zip(years[i:i + 19], years[i + 1:i + 20])):
                unchanged_bridges.append(bridge_id)
                break

    ts_df = df_filtered[
        ~df_filtered["STRUCTURE_NUMBER_008"].isin(set(unchanged_bridges))
    ].copy()

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
        axis=1, limit_direction="both"
    ).ffill(axis=1).bfill(axis=1)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(pivot_df)

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
        data.sort_values(["STRUCTURE_NUMBER_008", "Year of Data"])
        .groupby("STRUCTURE_NUMBER_008", as_index=False)
        .last()
    )

    bridge_summary = bridge_summary.merge(
        latest_static,
        on="STRUCTURE_NUMBER_008",
        how="left"
    )

    def compute_projection_slope(group):
        temp = group[["Year of Data", "Bridge Health Index (Overall)"]].dropna().sort_values("Year of Data")

        if len(temp) < 2:
            return pd.Series({
                "projection_slope": np.nan,
                "projection_intercept": np.nan,
                "last_year": np.nan,
                "last_bhi": np.nan
            })

        X = temp[["Year of Data"]].values
        y = temp["Bridge Health Index (Overall)"].values

        model = LinearRegression()
        model.fit(X, y)

        return pd.Series({
            "projection_slope": model.coef_[0],
            "projection_intercept": model.intercept_,
            "last_year": temp["Year of Data"].max(),
            "last_bhi": temp.loc[temp["Year of Data"].idxmax(), "Bridge Health Index (Overall)"]
        })

    projection_slopes_df = (
        ts_df.groupby("STRUCTURE_NUMBER_008")
        .apply(compute_projection_slope)
        .reset_index()
    )

    projection_list = []
    for _, row in projection_slopes_df.iterrows():
        bridge_id = row["STRUCTURE_NUMBER_008"]
        slope = row["projection_slope"]
        last_year = row["last_year"]
        last_bhi = row["last_bhi"]

        if pd.isna(slope) or pd.isna(last_year) or pd.isna(last_bhi):
            continue

        future_years = np.arange(int(last_year) + 1, int(last_year) + projection_horizon + 1)
        years_ahead = future_years - int(last_year)
        projected_bhi = last_bhi + slope * years_ahead
        projected_bhi = np.clip(projected_bhi, 0, 100)

        proj_df = pd.DataFrame({
            "STRUCTURE_NUMBER_008": bridge_id,
            "Year of Data": future_years,
            "Bridge Health Index (Overall)": projected_bhi,
            "DATA_TYPE": "Projected"
        })
        projection_list.append(proj_df)

    if projection_list:
        df_projected_only = pd.concat(projection_list, ignore_index=True)
    else:
        df_projected_only = pd.DataFrame(columns=[
            "STRUCTURE_NUMBER_008",
            "Year of Data",
            "Bridge Health Index (Overall)",
            "DATA_TYPE"
        ])

    df_observed_for_projection = ts_df[
        ["STRUCTURE_NUMBER_008", "Year of Data", "Bridge Health Index (Overall)", "DATA_TYPE"]
    ].copy()

    df_combined_projection = pd.concat(
        [df_observed_for_projection, df_projected_only],
        ignore_index=True
    ).sort_values(["STRUCTURE_NUMBER_008", "Year of Data"]).reset_index(drop=True)

    bridge_summary = bridge_summary.merge(
        projection_slopes_df[["STRUCTURE_NUMBER_008", "projection_slope", "last_year", "last_bhi"]],
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
        "final_unique_bridges": int(ts_df["STRUCTURE_NUMBER_008"].nunique()),
        "projected_bridges": int(df_projected_only["STRUCTURE_NUMBER_008"].nunique()),
        "projection_horizon_years": int(projection_horizon)
    }

    return {
        "ts_df": ts_df,
        "pivot_df": pivot_df,
        "clustered_df": clustered,
        "bridge_summary": bridge_summary,
        "years": years,
        "cluster_sizes": cluster_sizes,
        "kmeans": kmeans,
        "preprocessing_summary": preprocessing_summary,
        "df_full_interpolated": df_full_interpolated,
        "projection_slopes_df": projection_slopes_df,
        "df_projected_only": df_projected_only,
        "df_combined_projection": df_combined_projection
    }

# =========================================================
# LOAD PIPELINE
# =========================================================
try:
    static_df = load_data()
    analysis = prepare_analysis(static_df, n_clusters=N_CLUSTERS, projection_horizon=PROJECTION_HORIZON)
except Exception as e:
    st.error(f"Failed to initialize analysis: {e}")
    st.stop()

ts_df = analysis["ts_df"]
pivot_df = analysis["pivot_df"]
clustered_df = analysis["clustered_df"]
bridge_summary = analysis["bridge_summary"]
years_available = analysis["years"]
cluster_sizes = analysis["cluster_sizes"]
preprocessing_summary = analysis["preprocessing_summary"]
df_full_interpolated = analysis["df_full_interpolated"]
projection_slopes_df = analysis["projection_slopes_df"]
df_projected_only = analysis["df_projected_only"]
df_combined_projection = analysis["df_combined_projection"]

bridge_ids = sorted(pivot_df.index.astype(str).tolist())

# =========================================================
# SESSION STATE
# =========================================================
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Ask me about bridge deterioration trends, bridge profiles, clusters, "
                    "Bridge Health Index patterns, dataset inspection, or 20-year projections."
                )
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

initialize_session_state()

# =========================================================
# DATASET INSPECTION
# =========================================================
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

    return {"text": text, "schema_df": schema_df}

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
                return {"text": f"I couldn’t find a column named '{column_name}' in the dataset."}

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

    return {"text": text, "column_df": column_df, "values_df": values_df}

def preview_dataset(n_rows=10):
    df = static_df.copy().head(int(n_rows))
    return {"text": f"Showing the first {len(df)} rows of the dataset.", "preview_df": df}

# =========================================================
# FIGURES
# =========================================================
def make_bridge_trend_figure(bridge_id):
    matched = find_best_bridge_match(bridge_id, bridge_ids)
    if matched is None or matched not in pivot_df.index:
        return None

    row = pivot_df.loc[matched].dropna()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(row.index.astype(int), row.values.astype(float), marker="o")
    ax.set_title(f"BHI Trend for Bridge {matched}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Bridge Health Index (Overall)")
    ax.grid(True)
    return fig

def make_multi_bridge_trend_figure(bridge_id_list):
    fig, ax = plt.subplots(figsize=(11, 6))

    plotted = 0
    for bridge_id in bridge_id_list:
        matched = find_best_bridge_match(bridge_id, bridge_ids)
        if matched is None or matched not in pivot_df.index:
            continue
        row = pivot_df.loc[matched].dropna()
        ax.plot(row.index.astype(int), row.values.astype(float), marker="o", label=str(matched))
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return None

    ax.set_title("BHI Trends for Selected Bridges")
    ax.set_xlabel("Year")
    ax.set_ylabel("Bridge Health Index (Overall)")
    ax.grid(True)
    ax.legend()
    return fig

def make_cluster_trend_figure(cluster_id):
    try:
        cluster_id = int(cluster_id)
    except Exception:
        return None

    if "Cluster" not in clustered_df.columns:
        return None

    subset = clustered_df[clustered_df["Cluster"] == cluster_id].drop(columns=["Cluster"], errors="ignore")
    if subset.empty:
        return None

    years = subset.columns.astype(int)
    median_line = subset.median(axis=0)
    q1 = subset.quantile(0.25, axis=0)
    q3 = subset.quantile(0.75, axis=0)

    fig, ax = plt.subplots(figsize=(11, 6))

    for _, row in subset.iterrows():
        ax.plot(years, row.values.astype(float), alpha=0.15)

    ax.plot(years, median_line.values.astype(float), linewidth=3, label="Median BHI")
    ax.fill_between(years, q1.values.astype(float), q3.values.astype(float), alpha=0.2, label="IQR")

    ax.set_title(f"Cluster {cluster_id} BHI Trend")
    ax.set_xlabel("Year")
    ax.set_ylabel("Bridge Health Index (Overall)")
    ax.grid(True)
    ax.legend()
    return fig

def make_bridge_projection_figure(bridge_id):
    matched = find_best_bridge_match(bridge_id, bridge_ids)
    if matched is None:
        return None

    plot_df = df_combined_projection[
        df_combined_projection["STRUCTURE_NUMBER_008"].astype(str) == str(matched)
    ].copy()

    if plot_df.empty:
        return None

    plot_df = plot_df.sort_values("Year of Data")
    observed_part = plot_df[plot_df["DATA_TYPE"].isin(["Observed", "Interpolated"])]
    projected_part = plot_df[plot_df["DATA_TYPE"] == "Projected"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        observed_part["Year of Data"],
        observed_part["Bridge Health Index (Overall)"],
        marker="o",
        label="Observed / Interpolated"
    )

    if not projected_part.empty:
        ax.plot(
            projected_part["Year of Data"],
            projected_part["Bridge Health Index (Overall)"],
            marker="o",
            linestyle="--",
            label="20-Year Projection"
        )
        ax.axvline(
            observed_part["Year of Data"].max(),
            linestyle=":",
            label="Projection Start"
        )

    ax.set_title(f"Bridge BHI Trend + 20-Year Projection\nBridge ID: {matched}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Bridge Health Index (Overall)")
    ax.set_ylim(0, 100)
    ax.grid(True)
    ax.legend()
    return fig

# =========================================================
# ANALYSIS FUNCTIONS
# =========================================================
def overall_dataset_summary():
    total_bridges = pivot_df.shape[0]
    year_min = min(years_available)
    year_max = max(years_available)

    slopes = bridge_summary["deterioration_slope_per_year"].dropna()
    avg_slope = slopes.mean() if not slopes.empty else np.nan

    cluster_df_local = (
        bridge_summary["Cluster"]
        .value_counts()
        .sort_index()
        .reset_index()
    )
    cluster_df_local.columns = ["Cluster", "Number of Bridges"]

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
            "Final Unique Bridges",
            "Projected Bridges",
            "Projection Horizon (Years)"
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
            preprocessing_summary["final_unique_bridges"],
            preprocessing_summary["projected_bridges"],
            preprocessing_summary["projection_horizon_years"]
        ]
    })

    cluster_lines = [
        f"Cluster {int(row['Cluster'])}: {int(row['Number of Bridges'])}"
        for _, row in cluster_df_local.iterrows()
    ]

    avg_slope_text = f"{avg_slope:.4f}" if pd.notna(avg_slope) else "N/A"

    summary_text = (
        f"Here is the overall summary of the bridge deterioration dataset:\n\n"
        f"Total bridges used for clustering: {total_bridges:,}\n"
        f"Data span: {year_min} to {year_max}\n"
        f"Clusters: {N_CLUSTERS} clusters using KMeans on interpolated BHI trajectories\n"
        f"Bridges with >=20 records: {preprocessing_summary['bridges_with_20plus_records']:,}\n"
        f"Constant 20-year bridges removed: {preprocessing_summary['constant_20year_bridges_removed']:,}\n"
        f"Projected bridges: {preprocessing_summary['projected_bridges']:,}\n"
        f"Projection horizon: {preprocessing_summary['projection_horizon_years']} years\n"
        f"Cluster sizes:\n" + "\n".join(cluster_lines) + "\n"
        f"Average deterioration slope: {avg_slope_text} BHI points per year"
    )

    return {
        "text": summary_text,
        "summary_df": summary_df,
        "cluster_df": cluster_df_local
    }

def get_bridge_profile(bridge_id):
    matched = find_best_bridge_match(bridge_id, bridge_ids)
    if matched is None:
        return f"I could not find a matching bridge for '{bridge_id}'."

    row = bridge_summary[bridge_summary["STRUCTURE_NUMBER_008"].astype(str) == str(matched)]
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
        "Projection Slope": row.get("projection_slope", np.nan),
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
    matched = find_best_bridge_match(bridge_id, bridge_ids)
    if matched is None:
        return f"I could not find a matching bridge for '{bridge_id}'."

    if matched not in pivot_df.index:
        return f"No time-series data found for bridge {matched}."

    row = pivot_df.loc[matched]
    lines = [f"Trend for bridge {matched}:"]
    for year, value in row.items():
        if pd.notna(value):
            lines.append(f"{int(year)}: {float(value):.2f}")

    slope_row = bridge_summary[bridge_summary["STRUCTURE_NUMBER_008"].astype(str) == str(matched)]
    if not slope_row.empty:
        slope = slope_row["deterioration_slope_per_year"].iloc[0]
        if pd.notna(slope):
            direction = "declining" if slope < 0 else "improving" if slope > 0 else "stable"
            lines.append(f"\nEstimated slope: {slope:.3f} BHI points/year ({direction}).")

    return "\n".join(lines)

def compare_two_bridges(bridge_id_1, bridge_id_2):
    matched1 = find_best_bridge_match(bridge_id_1, bridge_ids)
    matched2 = find_best_bridge_match(bridge_id_2, bridge_ids)

    if matched1 is None:
        return f"I could not find a matching bridge for '{bridge_id_1}'."
    if matched2 is None:
        return f"I could not find a matching bridge for '{bridge_id_2}'."

    row1 = bridge_summary[bridge_summary["STRUCTURE_NUMBER_008"].astype(str) == str(matched1)]
    row2 = bridge_summary[bridge_summary["STRUCTURE_NUMBER_008"].astype(str) == str(matched2)]

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
        "avg_latest_overall_bhi": subset["Bridge Health Index (Overall)"].mean(),
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
    interpretation.append(
        f"Cluster {cluster_id_1 if metrics1['avg_latest_overall_bhi'] > metrics2['avg_latest_overall_bhi'] else cluster_id_2} "
        f"has the higher average latest overall BHI."
    )
    interpretation.append(
        f"Cluster {cluster_id_1 if metrics1['avg_adt'] > metrics2['avg_adt'] else cluster_id_2} "
        f"carries higher average daily traffic."
    )
    interpretation.append(
        f"Cluster {cluster_id_1 if metrics1['avg_span_len'] > metrics2['avg_span_len'] else cluster_id_2} "
        f"has longer spans on average."
    )

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
        return {"text": f"I found partial information, but not enough PCA variables remained for cluster {cluster_id}."}

    df_filtered_2 = df_filtered_2.dropna()
    if df_filtered_2.shape[0] < 2:
        return {"text": f"I found partial information, but too few rows remained after cleaning for cluster {cluster_id}."}

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
        f"Interpretation note: larger absolute PC1 loadings indicate stronger contribution to PC1. "
        f"This does not automatically imply causation."
    )

    return {
        "text": text,
        "pc1_table": pc1_df,
        "explained_variance_ratio_pc1": explained_var,
        "cluster_ids": [cluster_id],
        "label": "cluster_pca_drivers"
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

def get_top_best_bridges(year=None, top_n=5):
    subset = bridge_summary.dropna(subset=["Bridge Health Index (Overall)"]).copy()
    subset = subset.sort_values("Bridge Health Index (Overall)", ascending=False).head(top_n)

    lines = [f"Top {top_n} best bridges based on latest overall BHI:"]
    bridge_id_list = []

    for _, row in subset.iterrows():
        bid = str(row["STRUCTURE_NUMBER_008"])
        bridge_id_list.append(bid)
        lines.append(
            f"- Bridge {bid}: latest overall BHI {row['Bridge Health Index (Overall)']:.2f}, "
            f"cluster {int(row['Cluster'])}, slope {row['deterioration_slope_per_year']:.3f}"
        )

    return {
        "text": "\n".join(lines),
        "bridge_ids": bridge_id_list,
        "analysis_df": subset[[
            "STRUCTURE_NUMBER_008",
            "Bridge Health Index (Overall)",
            "deterioration_slope_per_year",
            "Cluster"
        ]].copy(),
        "label": f"top_{top_n}_best_bridges"
    }

def get_bridge_projection(bridge_id, horizon=PROJECTION_HORIZON):
    matched = find_best_bridge_match(bridge_id, bridge_ids)
    if matched is None:
        return {"text": f"I could not find a matching bridge for '{bridge_id}'."}

    observed = df_combined_projection[
        (df_combined_projection["STRUCTURE_NUMBER_008"].astype(str) == str(matched)) &
        (df_combined_projection["DATA_TYPE"].isin(["Observed", "Interpolated"]))
    ].copy()

    projected = df_combined_projection[
        (df_combined_projection["STRUCTURE_NUMBER_008"].astype(str) == str(matched)) &
        (df_combined_projection["DATA_TYPE"] == "Projected")
    ].copy()

    if observed.empty:
        return {"text": f"No observed time-series data found for bridge {matched}."}

    observed = observed.sort_values("Year of Data")
    projected = projected.sort_values("Year of Data")

    last_obs = observed.iloc[-1]
    first_proj = projected.iloc[0] if not projected.empty else None
    last_proj = projected.iloc[-1] if not projected.empty else None

    slope_row = projection_slopes_df[
        projection_slopes_df["STRUCTURE_NUMBER_008"].astype(str) == str(matched)
    ]
    proj_slope = slope_row["projection_slope"].iloc[0] if not slope_row.empty else np.nan

    text = [
        f"20-year projection summary for bridge {matched}:",
        f"- Last observed year: {int(last_obs['Year of Data'])}",
        f"- Last observed overall BHI: {float(last_obs['Bridge Health Index (Overall)']):.2f}",
        f"- Projection slope: {proj_slope:.3f} BHI points/year" if pd.notna(proj_slope) else "- Projection slope: N/A"
    ]

    if first_proj is not None:
        text.append(
            f"- First projected year: {int(first_proj['Year of Data'])}, "
            f"projected BHI: {float(first_proj['Bridge Health Index (Overall)']):.2f}"
        )

    if last_proj is not None:
        text.append(
            f"- Year {int(last_proj['Year of Data'])}: "
            f"projected BHI: {float(last_proj['Bridge Health Index (Overall)']):.2f}"
        )

    return {
        "text": "\n".join(text),
        "analysis_df": pd.concat([observed.tail(5), projected.head(horizon)]).reset_index(drop=True),
        "bridge_ids": [str(matched)],
        "label": "single_bridge_projection"
    }

def get_worst_projected_bridges(top_n=5):
    if df_projected_only.empty:
        return {"text": "No projected bridge data is available."}

    final_proj = (
        df_projected_only.sort_values(["STRUCTURE_NUMBER_008", "Year of Data"])
        .groupby("STRUCTURE_NUMBER_008", as_index=False)
        .last()
    )

    ranked = final_proj.sort_values("Bridge Health Index (Overall)", ascending=True).head(top_n).copy()

    lines = [f"Top {top_n} worst projected bridges based on final projected overall BHI:"]
    bridge_id_list = []

    for _, row in ranked.iterrows():
        bid = str(row["STRUCTURE_NUMBER_008"])
        bridge_id_list.append(bid)
        lines.append(
            f"- Bridge {bid}: projected year {int(row['Year of Data'])}, "
            f"projected BHI {float(row['Bridge Health Index (Overall)']):.2f}"
        )

    return {
        "text": "\n".join(lines),
        "analysis_df": ranked,
        "bridge_ids": bridge_id_list,
        "label": f"top_{top_n}_worst_projected_bridges"
    }

def get_cluster_projection_summary(cluster_id):
    try:
        cluster_id = int(cluster_id)
    except Exception:
        return {"text": f"Invalid cluster id: {cluster_id}"}

    cluster_bridges = bridge_summary[
        bridge_summary["Cluster"] == cluster_id
    ]["STRUCTURE_NUMBER_008"].astype(str).tolist()

    if not cluster_bridges:
        return {"text": f"No bridges found in cluster {cluster_id}."}

    proj_subset = df_projected_only[
        df_projected_only["STRUCTURE_NUMBER_008"].astype(str).isin(cluster_bridges)
    ].copy()

    if proj_subset.empty:
        return {"text": f"No projected data found for cluster {cluster_id}."}

    final_proj = (
        proj_subset.sort_values(["STRUCTURE_NUMBER_008", "Year of Data"])
        .groupby("STRUCTURE_NUMBER_008", as_index=False)
        .last()
    )

    avg_final_bhi = final_proj["Bridge Health Index (Overall)"].mean()
    min_final_bhi = final_proj["Bridge Health Index (Overall)"].min()
    max_final_bhi = final_proj["Bridge Health Index (Overall)"].max()

    text = (
        f"Projected summary for cluster {cluster_id}:\n"
        f"- Bridges with projections: {len(final_proj)}\n"
        f"- Average final projected BHI: {avg_final_bhi:.2f}\n"
        f"- Minimum final projected BHI: {min_final_bhi:.2f}\n"
        f"- Maximum final projected BHI: {max_final_bhi:.2f}"
    )

    return {
        "text": text,
        "analysis_df": final_proj.sort_values("Bridge Health Index (Overall)", ascending=True),
        "cluster_ids": [cluster_id],
        "label": "cluster_projection_summary"
    }

# =========================================================
# FOLLOW-UP CONTEXT
# =========================================================
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

# =========================================================
# SAFE PYTHON ANALYSIS
# =========================================================
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
- df_projected_only
- df_combined_projection
- projection_slopes_df

Rules:
- Use only pandas (pd), numpy (np), and matplotlib.pyplot (plt) if needed.
- Do not import os, sys, subprocess, pathlib, requests, or any network/file libraries.
- Do not read or write files.
- Do not call open().
- Do not use eval() or exec().
- Store the final natural-language answer in a variable named result_text.
- Store a step-by-step trace in a variable named execution_steps as a Python list of strings.
- If returning a table, store it in a variable named result_df.
- If the question cannot be answered from the available dataframes, set:
  result_text = "I couldn’t find this information in the dataset."
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
        "projection_slopes_df": projection_slopes_df.copy(),
        "df_projected_only": df_projected_only.copy(),
        "df_combined_projection": df_combined_projection.copy(),
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

# =========================================================
# ROUTER
# =========================================================
def route_question(question: str):
    q = question.lower().strip()
    cluster_ids_local = extract_cluster_ids(q)
    bridge_ids_local = extract_bridge_ids_from_question(question)

    if any(p in q for p in ["overall summary", "dataset summary", "summary of dataset"]):
        return {"mode": "direct_tool", "tool_name": "overall_dataset_summary", "tool_input": {}}

    if any(p in q for p in ["show schema", "dataset schema", "columns in dataset"]):
        return {"mode": "direct_tool", "tool_name": "dataset_schema", "tool_input": {}}

    if any(p in q for p in ["preview dataset", "show first rows", "show sample rows"]):
        return {"mode": "direct_tool", "tool_name": "preview_dataset", "tool_input": {"n_rows": 10}}

    if any(p in q for p in ["top deteriorating bridges", "worst deteriorating bridges", "fastest deteriorating bridges"]):
        top_n = extract_top_n(question, default=5)
        return {"mode": "direct_tool", "tool_name": "top_deteriorating_bridges", "tool_input": {"top_n": top_n}}

    if any(p in q for p in ["top best bridges", "best bridges"]):
        top_n = extract_top_n(question, default=5)
        return {"mode": "direct_tool", "tool_name": "top_best_bridges", "tool_input": {"top_n": top_n}}

    if any(p in q for p in ["20-year projection", "20 year projection", "projection for bridge", "forecast for bridge", "future bhi"]):
        if bridge_ids_local:
            return {
                "mode": "direct_tool",
                "tool_name": "bridge_projection",
                "tool_input": {"bridge_id": bridge_ids_local[0]}
            }

    if any(p in q for p in ["show projected trend", "plot projected trend", "projection plot", "forecast plot"]):
        if bridge_ids_local:
            return {
                "mode": "direct_tool",
                "tool_name": "bridge_projection_plot",
                "tool_input": {"bridge_id": bridge_ids_local[0]}
            }

    if any(p in q for p in ["worst projected bridges", "lowest projected bhi", "future worst bridges"]):
        top_n = extract_top_n(question, default=5)
        return {
            "mode": "direct_tool",
            "tool_name": "worst_projected_bridges",
            "tool_input": {"top_n": top_n}
        }

    if any(p in q for p in ["projected cluster summary", "cluster projection", "future cluster performance", "projection summary for cluster"]):
        cluster_id = extract_single_cluster_id(question)
        if cluster_id is not None:
            return {
                "mode": "direct_tool",
                "tool_name": "cluster_projection_summary",
                "tool_input": {"cluster_id": cluster_id}
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
            "feature drivers",
            "pca drivers",
            "pc1 loadings"
        ])
    ):
        return {
            "mode": "direct_tool",
            "tool_name": "cluster_pca_drivers",
            "tool_input": {"cluster_id": cluster_ids_local[0]}
        }

    if len(cluster_ids_local) == 1 and any(p in q for p in ["summarize cluster", "cluster summary", "summary of cluster"]):
        return {
            "mode": "direct_tool",
            "tool_name": "cluster_summary",
            "tool_input": {"cluster_id": cluster_ids_local[0]}
        }

    if len(cluster_ids_local) == 2 and any(p in q for p in ["compare cluster", "compare clusters", "compare analytically"]):
        return {
            "mode": "direct_tool",
            "tool_name": "compare_clusters",
            "tool_input": {"cluster_id_1": cluster_ids_local[0], "cluster_id_2": cluster_ids_local[1]}
        }

    if len(cluster_ids_local) == 1 and any(p in q for p in ["show trend", "plot trend", "median bhi", "median line", "interpret the median"]):
        return {
            "mode": "direct_tool",
            "tool_name": "cluster_trend",
            "tool_input": {"cluster_id": cluster_ids_local[0]}
        }

    if len(bridge_ids_local) == 1 and any(p in q for p in ["profile", "bridge profile"]):
        return {
            "mode": "direct_tool",
            "tool_name": "bridge_profile",
            "tool_input": {"bridge_id": bridge_ids_local[0]}
        }

    if len(bridge_ids_local) == 1 and any(p in q for p in ["trend", "show trend", "plot trend"]):
        return {
            "mode": "direct_tool",
            "tool_name": "bridge_trend",
            "tool_input": {"bridge_id": bridge_ids_local[0]}
        }

    if len(bridge_ids_local) >= 2 and any(p in q for p in ["compare", "compare bridges"]):
        return {
            "mode": "direct_tool",
            "tool_name": "compare_bridges",
            "tool_input": {"bridge_id_1": bridge_ids_local[0], "bridge_id_2": bridge_ids_local[1]}
        }

    if any(p in q for p in ["python analysis", "run python", "analyze with python"]):
        return {"mode": "python", "tool_input": {"question": question}}

    return {"mode": "llm", "tool_input": {"question": question}}

# =========================================================
# TOOL EXECUTION
# =========================================================
def execute_tool(tool_name, tool_input):
    if tool_name == "overall_dataset_summary":
        return overall_dataset_summary()

    elif tool_name == "dataset_schema":
        return get_dataset_schema()

    elif tool_name == "preview_dataset":
        return preview_dataset(**tool_input)

    elif tool_name == "top_deteriorating_bridges":
        return get_top_deteriorating_bridges(**tool_input)

    elif tool_name == "top_best_bridges":
        return get_top_best_bridges(**tool_input)

    elif tool_name == "bridge_profile":
        text = get_bridge_profile(**tool_input)
        return {
            "text": text,
            "bridge_ids": [tool_input["bridge_id"]],
            "label": "bridge_profile"
        }

    elif tool_name == "bridge_trend":
        text = get_bridge_trend(**tool_input)
        fig = make_bridge_trend_figure(tool_input["bridge_id"])
        return {
            "text": text,
            "figure": fig,
            "bridge_ids": [tool_input["bridge_id"]],
            "label": "bridge_trend"
        }

    elif tool_name == "compare_bridges":
        text = compare_two_bridges(**tool_input)
        return {
            "text": text,
            "bridge_ids": [tool_input["bridge_id_1"], tool_input["bridge_id_2"]],
            "label": "compare_bridges"
        }

    elif tool_name == "cluster_summary":
        text = get_cluster_summary(**tool_input)
        return {
            "text": text,
            "cluster_ids": [tool_input["cluster_id"]],
            "label": "cluster_summary"
        }

    elif tool_name == "compare_clusters":
        text = compare_two_clusters(**tool_input)
        return {
            "text": text,
            "cluster_ids": [tool_input["cluster_id_1"], tool_input["cluster_id_2"]],
            "label": "compare_clusters"
        }

    elif tool_name == "cluster_trend":
        cluster_id = tool_input["cluster_id"]
        fig = make_cluster_trend_figure(cluster_id)
        if "interpret the median" in st.session_state.get("latest_question", "").lower() or "median line" in st.session_state.get("latest_question", "").lower():
            text = (
                f"The median BHI line in cluster {cluster_id} represents the middle bridge health value "
                f"at each year across all bridges in that cluster. It shows the typical central trend, "
                f"while reducing the influence of extreme outliers."
            )
        else:
            text = f"Showing the BHI trend for cluster {cluster_id}."
        return {
            "text": text,
            "figure": fig,
            "cluster_ids": [cluster_id],
            "label": "cluster_trend"
        }

    elif tool_name == "cluster_pca_drivers":
        return get_cluster_pca_drivers(**tool_input)

    elif tool_name == "bridge_projection":
        return get_bridge_projection(**tool_input)

    elif tool_name == "bridge_projection_plot":
        fig = make_bridge_projection_figure(**tool_input)
        return {
            "text": f"Showing the 20-year projection plot for bridge {tool_input['bridge_id']}.",
            "figure": fig,
            "bridge_ids": [tool_input["bridge_id"]],
            "label": "bridge_projection_plot"
        }

    elif tool_name == "worst_projected_bridges":
        return get_worst_projected_bridges(**tool_input)

    elif tool_name == "cluster_projection_summary":
        return get_cluster_projection_summary(**tool_input)

    return {"text": f"Unknown tool: {tool_name}"}

# =========================================================
# LLM FALLBACK
# =========================================================
def answer_with_llm(question: str):
    system_prompt = f"""
You are a bridge deterioration analysis assistant.
You must stay grounded in the available app outputs and bridge dataset context.
Do not invent bridge IDs or results.
If the answer is not supported, say so plainly.

Available context:
- Bridge summary table with latest BHI, slopes, and clusters
- Clustered BHI time series
- 20-year projected bridge data
- Projection slopes
"""

    try:
        response = bedrock.converse(
            modelId=BEDROCK_MODEL_ID,
            system=[{"text": system_prompt}],
            messages=[{"role": "user", "content": [{"text": question}]}]
        )
        output_message = response["output"]["message"]
        answer_text = extract_text_from_content_blocks(output_message["content"])
        return {"text": strip_thinking_blocks(answer_text)}
    except Exception as e:
        return {"text": f"LLM response failed: {e}"}

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("Examples")
    st.markdown("""
- overall dataset summary
- summarize cluster 2
- compare cluster 0 and cluster 3 analytically
- what features characterize cluster 2
- show trend for bridge 1234567890
- bridge profile for bridge 1234567890
- show the 5 fastest deteriorating bridges
- show the 5 best bridges
- show the 20-year projection for bridge 1234567890
- plot projected trend for bridge 1234567890
- which are the 5 worst projected bridges
- give me the projected cluster summary for cluster 2
- show dataset schema
- preview dataset
""")

# =========================================================
# CHAT HISTORY
# =========================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =========================================================
# MAIN CHAT LOOP
# =========================================================
question = st.chat_input("Ask a question about the bridge dataset...")

if question:
    st.session_state.latest_question = question
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    route = route_question(question)

    with st.chat_message("assistant"):
        result = None

        if route["mode"] == "direct_tool":
            result = execute_tool(route["tool_name"], route["tool_input"])

        elif route["mode"] == "python":
            result = run_python_analysis(route["tool_input"]["question"])

        else:
            result = answer_with_llm(route["tool_input"]["question"])

        result_text = result.get("text", "No response generated.")
        st.markdown(result_text)

        if "summary_df" in result and isinstance(result["summary_df"], pd.DataFrame):
            st.dataframe(result["summary_df"], use_container_width=True)

        if "cluster_df" in result and isinstance(result["cluster_df"], pd.DataFrame):
            st.dataframe(result["cluster_df"], use_container_width=True)

        if "schema_df" in result and isinstance(result["schema_df"], pd.DataFrame):
            render_paginated_dataframe(result["schema_df"], key_prefix="schema_df", title="Dataset Schema")

        if "preview_df" in result and isinstance(result["preview_df"], pd.DataFrame):
            st.dataframe(result["preview_df"], use_container_width=True)

        if "column_df" in result and isinstance(result["column_df"], pd.DataFrame):
            st.dataframe(result["column_df"], use_container_width=True)

        if "values_df" in result and isinstance(result["values_df"], pd.DataFrame):
            st.dataframe(result["values_df"], use_container_width=True)

        if "analysis_df" in result and isinstance(result["analysis_df"], pd.DataFrame):
            render_paginated_dataframe(result["analysis_df"], key_prefix="analysis_df", title="Analysis Output")

        if "pc1_table" in result and isinstance(result["pc1_table"], pd.DataFrame):
            render_paginated_dataframe(result["pc1_table"], key_prefix="pc1_table", title="PC1 Loadings")

        if "figure" in result and result["figure"] is not None:
            st.pyplot(result["figure"])

        if "execution_steps" in result and result["execution_steps"]:
            with st.expander("Execution steps"):
                for step in result["execution_steps"]:
                    st.markdown(f"- {step}")

        if "generated_code" in result and result["generated_code"]:
            with st.expander("Generated Python code"):
                st.code(result["generated_code"], language="python")

        if "stdout" in result and result["stdout"]:
            with st.expander("Python stdout / traceback"):
                st.text(result["stdout"])

        update_last_result_context(question, result)
        st.session_state.messages.append({"role": "assistant", "content": result_text})
