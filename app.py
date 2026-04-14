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

# ---------------------------
# Config
# ---------------------------
N_CLUSTERS = 6
STATIC_FILE = "STEEL_Bridges.csv"

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

initialize_session_state()

# ---------------------------
# Helpers
# ---------------------------
def find_best_bridge_match(bridge_id: str):
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
    return [int(m) for m in matches]


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

# ---------------------------
# Analysis functions
# ---------------------------
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


def interpret_cluster_trend(cluster_id):
    try:
        cluster_id = int(cluster_id)
    except Exception:
        return {"text": "Invalid cluster id."}

    clustered_subset = clustered_df[clustered_df["Cluster"] == cluster_id]

    if clustered_subset.empty:
        return {"text": f"No data for cluster {cluster_id}."}

    bridge_ids_local = clustered_subset.index.tolist()
    cluster_ts = pivot_df.loc[bridge_ids_local].copy()

    if cluster_ts.empty:
        return {"text": f"No time-series data found for cluster {cluster_id}."}

    median_trend = cluster_ts.median(axis=0)
    q1 = cluster_ts.quantile(0.25, axis=0)
    q3 = cluster_ts.quantile(0.75, axis=0)

    x = np.array(median_trend.index, dtype=float)
    y = median_trend.values.astype(float)

    if len(x) >= 2 and np.isfinite(y).sum() >= 2:
        slope, _, _, _, _ = linregress(x, y)
    else:
        slope = np.nan

    if pd.isna(slope):
        direction = "stable"
        slope_text = "not available"
    elif slope < 0:
        direction = "declining"
        slope_text = f"{slope:.3f}"
    elif slope > 0:
        direction = "improving"
        slope_text = f"{slope:.3f}"
    else:
        direction = "stable"
        slope_text = f"{slope:.3f}"

    text = (
        f"The median BHI line represents the typical central deterioration pattern of bridges in Cluster {cluster_id}.\n\n"
        f"Each point on this line is the median overall BHI across all bridges in the cluster for that year.\n\n"
        f"Overall, the median trend is {direction} over time"
        + (f" with a slope of {slope_text} BHI points per year.\n\n" if slope_text != "not available" else ".\n\n")
        + f"This means the typical bridge in this cluster is "
        + (
            "gradually losing condition over time.\n\n" if direction == "declining"
            else "generally improving over time.\n\n" if direction == "improving"
            else "remaining fairly stable over time.\n\n"
        )
        + "If your plot includes a shaded band, that band shows variability around the median, often using the interquartile range.\n"
        + "A wider band means bridges in the cluster behave less consistently.\n"
        + "A narrower band means bridges in the cluster follow a more similar pattern."
    )

    median_df = pd.DataFrame({
        "Year": median_trend.index,
        "Median_BHI": median_trend.values,
        "Q1": q1.values,
        "Q3": q3.values
    })

    return {
        "text": text,
        "analysis_df": median_df,
        "cluster_ids": [cluster_id],
        "label": "cluster_median_trend_interpretation"
    }

# ---------------------------
# Routing
# ---------------------------
def route_question(question: str):
    q = question.lower().strip()
    cluster_ids_local = extract_cluster_ids(q)

    if (
        len(cluster_ids_local) == 1 and
        any(phrase in q for phrase in [
            "median bhi",
            "median line",
            "interpret the median",
            "median bhi line",
            "median trend",
            "median line in this plot",
            "how should i interpret the median",
            "what does the median bhi line mean"
        ])
    ):
        return {
            "mode": "direct_tool",
            "tool_name": "interpret_cluster_trend",
            "tool_input": {"cluster_id": cluster_ids_local[0]}
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

# ---------------------------
# Direct tool executor
# ---------------------------
def execute_direct_tool(tool_name, tool_input):
    if tool_name == "interpret_cluster_trend":
        return interpret_cluster_trend(tool_input["cluster_id"])

    if tool_name == "cluster_pca_drivers":
        return get_cluster_pca_drivers(
            cluster_id=tool_input["cluster_id"],
            top_n=tool_input.get("top_n", 8)
        )

    if tool_name == "compare_clusters":
        text = compare_two_clusters(
            cluster_id_1=tool_input["cluster_id_1"],
            cluster_id_2=tool_input["cluster_id_2"]
        )
        return {
            "text": text,
            "cluster_ids": [tool_input["cluster_id_1"], tool_input["cluster_id_2"]],
            "label": "compare_clusters"
        }

    if tool_name == "cluster_deep_dive":
        return get_cluster_deep_dive(tool_input["cluster_id"])

    if tool_name == "bridges_in_cluster":
        return get_bridges_in_cluster(tool_input["cluster_id"])

    if tool_name == "browse_dataset_rows":
        return browse_dataset_rows(
            offset=tool_input.get("offset", 0),
            limit=tool_input.get("limit", 100)
        )

    return {"text": f"Unknown tool: {tool_name}"}

# ---------------------------
# Main handler
# ---------------------------
def handle_question(question: str):
    route = route_question(question)

    if route["mode"] == "direct_text":
        result = {
            "text": route["text"],
            "label": "direct_text"
        }
        if "pending_compare_cluster" in route:
            st.session_state.pending_compare_cluster = route["pending_compare_cluster"]
        return result

    if route["mode"] == "direct_tool":
        return execute_direct_tool(
            tool_name=route["tool_name"],
            tool_input=route["tool_input"]
        )

    return run_python_analysis(question)
