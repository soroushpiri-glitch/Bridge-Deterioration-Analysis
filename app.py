import re
import os
import boto3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from botocore.exceptions import BotoCoreError, ClientError
from sklearn.cluster import KMeans
from scipy.stats import linregress

# ---------------------------
# Config
# ---------------------------
N_CLUSTERS = 6
STATIC_FILE = "STEEL_Bridges.csv"
TS_FILE = "Steel_Bridges_20_and_Over_20_TimeSeries_Data_revised.xls"

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
    """
    Robust table reader that handles:
    - CSV
    - XLSX via openpyxl
    - XLS via xlrd
    - mislabeled .xls files that are actually .xlsx or text/CSV
    """
    ext = os.path.splitext(file_path)[1].lower()

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if ext == ".csv":
        return pd.read_csv(file_path, low_memory=False)

    if ext == ".xlsx":
        return pd.read_excel(file_path, engine="openpyxl")

    if ext == ".xls":
        # First try true old-style XLS
        try:
            return pd.read_excel(file_path, engine="xlrd")
        except Exception as xls_err:
            # If the file is mislabeled and is actually xlsx, try openpyxl
            try:
                return pd.read_excel(file_path, engine="openpyxl")
            except Exception:
                # If it is actually text/csv with the wrong extension, try CSV readers
                try:
                    return pd.read_csv(file_path, low_memory=False)
                except Exception:
                    try:
                        return pd.read_csv(file_path, sep="\t", low_memory=False)
                    except Exception:
                        raise ValueError(
                            f"Could not read '{file_path}'. "
                            f"It has .xls extension, but it does not appear to be a valid XLS workbook. "
                            f"It may be mislabeled, corrupted, or saved in another format. "
                            f"Original xlrd error: {xls_err}"
                        ) from xls_err

    raise ValueError(f"Unsupported file type: {ext}")

# ---------------------------
# Load data
# ---------------------------
@st.cache_data
def load_data():
    if not os.path.exists(STATIC_FILE):
        st.error(f"Missing required file: {STATIC_FILE}")
        st.stop()

    if not os.path.exists(TS_FILE):
        st.error(f"Missing required file: {TS_FILE}")
        st.stop()

    try:
        static_df = read_table_file(STATIC_FILE)
    except Exception as e:
        st.error(f"Failed to read static file '{STATIC_FILE}': {e}")
        st.stop()

    try:
        ts_df = read_table_file(TS_FILE)
    except Exception as e:
        st.error(f"Failed to read time-series file '{TS_FILE}': {e}")
        st.stop()

    static_df.columns = static_df.columns.str.strip()
    ts_df.columns = ts_df.columns.str.strip()

    required_static_cols = ["STRUCTURE_NUMBER_008", "Year of Data"]
    required_ts_cols = ["STRUCTURE_NUMBER_008", "Year of Data", "Bridge Health Index (Overall)"]

    missing_static = [c for c in required_static_cols if c not in static_df.columns]
    missing_ts = [c for c in required_ts_cols if c not in ts_df.columns]

    if missing_static:
        st.error(f"Static file is missing required columns: {missing_static}")
        st.stop()

    if missing_ts:
        st.error(f"Time-series file is missing required columns: {missing_ts}")
        st.stop()

    numeric_cols_static = [
        "Year of Data",
        "Bridge Health Index (Overall)",
        "Bridge Health Index (Deck)",
        "Bridge Health Index (Super)",
        "Bridge Health Index (Sub)",
        "COUNTY_CODE_003",
        "YEAR_BUILT_027",
        "TRAFFIC_LANES_ON_028A",
        "ADT_029",
        "YEAR_ADT_030",
        "MAIN_UNIT_SPANS_045",
        "MAX_SPAN_LEN_MT_048",
        "STRUCTURE_LEN_MT_049",
        "DECK_WIDTH_MT_052",
        "LAT_016",
        "LONG_017"
    ]

    numeric_cols_ts = [
        "Year of Data",
        "Bridge Health Index (Overall)",
        "Bridge Health Index (Deck)",
        "Bridge Health Index (Super)",
        "Bridge Health Index (Sub)"
    ]

    for col in numeric_cols_static:
        if col in static_df.columns:
            static_df[col] = pd.to_numeric(static_df[col], errors="coerce")

    for col in numeric_cols_ts:
        if col in ts_df.columns:
            ts_df[col] = pd.to_numeric(ts_df[col], errors="coerce")

    static_df["STRUCTURE_NUMBER_008"] = static_df["STRUCTURE_NUMBER_008"].astype(str).str.strip()
    ts_df["STRUCTURE_NUMBER_008"] = ts_df["STRUCTURE_NUMBER_008"].astype(str).str.strip()

    static_df = static_df.dropna(subset=["STRUCTURE_NUMBER_008", "Year of Data"])
    ts_df = ts_df.dropna(subset=["STRUCTURE_NUMBER_008", "Year of Data", "Bridge Health Index (Overall)"])

    static_df["Year of Data"] = static_df["Year of Data"].astype(int)
    ts_df["Year of Data"] = ts_df["Year of Data"].astype(int)

    return static_df, ts_df

# ---------------------------
# Prepare analysis
# ---------------------------
@st.cache_data
def prepare_analysis(static_df, ts_df, n_clusters=N_CLUSTERS):
    df = ts_df[
        ["STRUCTURE_NUMBER_008", "Year of Data", "Bridge Health Index (Overall)"]
    ].dropna()

    pivot_df = df.pivot(
        index="STRUCTURE_NUMBER_008",
        columns="Year of Data",
        values="Bridge Health Index (Overall)"
    )

    pivot_df = pivot_df.sort_index(axis=1)
    pivot_df = pivot_df.interpolate(axis=1, limit_direction="both").ffill(axis=1).bfill(axis=1)

    filtered_df = pivot_df.copy()

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
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

    return {
        "pivot_df": pivot_df,
        "clustered_df": clustered,
        "bridge_summary": bridge_summary,
        "years": years,
        "cluster_sizes": cluster_sizes,
        "kmeans": kmeans
    }

static_df, ts_df = load_data()
analysis = prepare_analysis(static_df, ts_df, n_clusters=N_CLUSTERS)

pivot_df = analysis["pivot_df"]
clustered_df = analysis["clustered_df"]
bridge_summary = analysis["bridge_summary"]
years_available = analysis["years"]
cluster_sizes = analysis["cluster_sizes"]

bridge_ids = sorted(pivot_df.index.astype(str).tolist())

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
    match = re.search(r"top\s+(\d+)", user_query.lower())
    if match:
        return int(match.group(1))
    return default

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
            "Total Bridges",
            "Start Year",
            "End Year",
            "Number of Clusters",
            "Average Deterioration Slope"
        ],
        "Value": [
            total_bridges,
            year_min,
            year_max,
            N_CLUSTERS,
            round(avg_slope, 4) if pd.notna(avg_slope) else np.nan
        ]
    })

    cluster_lines = [
        f"Cluster {int(row['Cluster'])}: {int(row['Number of Bridges'])}"
        for _, row in cluster_df.iterrows()
    ]

    avg_slope_text = f"{avg_slope:.4f}" if pd.notna(avg_slope) else "N/A"

    summary_text = (
        f"Here is the overall summary of the bridge deterioration dataset:\n\n"
        f"Total bridges: {total_bridges:,}\n"
        f"Data span: {year_min} to {year_max}\n"
        f"Clusters: {N_CLUSTERS} clusters using KMeans on BHI trajectories\n"
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

    metrics = {
        "count": len(subset),
        "avg_latest_overall_bhi": subset["Bridge Health Index (Overall)"].mean(),
        "avg_deck_bhi": subset["Bridge Health Index (Deck)"].mean() if "Bridge Health Index (Deck)" in subset.columns else np.nan,
        "avg_super_bhi": subset["Bridge Health Index (Super)"].mean() if "Bridge Health Index (Super)" in subset.columns else np.nan,
        "avg_sub_bhi": subset["Bridge Health Index (Sub)"].mean() if "Bridge Health Index (Sub)" in subset.columns else np.nan,
        "avg_year_built": subset["YEAR_BUILT_027"].mean() if "YEAR_BUILT_027" in subset.columns else np.nan,
        "avg_adt": subset["ADT_029"].mean() if "ADT_029" in subset.columns else np.nan,
        "avg_span_len": subset["MAX_SPAN_LEN_MT_048"].mean() if "MAX_SPAN_LEN_MT_048" in subset.columns else np.nan,
        "avg_slope": subset["deterioration_slope_per_year"].mean(),
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


def get_top_deteriorating_bridges(top_n=5):
    subset = bridge_summary.dropna(subset=["deterioration_slope_per_year"]).copy()
    subset = subset.sort_values("deterioration_slope_per_year", ascending=True).head(top_n)

    lines = [f"Top {top_n} fastest deteriorating bridges based on slope:"]
    for _, row in subset.iterrows():
        lines.append(
            f"- Bridge {row['STRUCTURE_NUMBER_008']}: slope {row['deterioration_slope_per_year']:.3f}, "
            f"latest overall BHI {row['Bridge Health Index (Overall)']:.2f}, cluster {int(row['Cluster'])}"
        )
    return "\n".join(lines)


def get_top_best_bridges(year, top_n=5):
    year = int(year)
    subset = ts_df[ts_df["Year of Data"] == year].copy()
    if subset.empty:
        return f"No data found for year {year}."

    subset = subset.sort_values("Bridge Health Index (Overall)", ascending=False).head(top_n)

    lines = [f"Top {top_n} bridges by overall BHI in {year}:"]
    for _, row in subset.iterrows():
        lines.append(
            f"- Bridge {row['STRUCTURE_NUMBER_008']}: {row['Bridge Health Index (Overall)']:.2f}"
        )
    return "\n".join(lines)


def get_top_worst_bridges(year, top_n=5):
    year = int(year)
    subset = ts_df[ts_df["Year of Data"] == year].copy()
    if subset.empty:
        return f"No data found for year {year}."

    subset = subset.sort_values("Bridge Health Index (Overall)", ascending=True).head(top_n)

    lines = [f"Top {top_n} worst bridges by overall BHI in {year}:"]
    for _, row in subset.iterrows():
        lines.append(
            f"- Bridge {row['STRUCTURE_NUMBER_008']}: {row['Bridge Health Index (Overall)']:.2f}"
        )
    return "\n".join(lines)

# ---------------------------
# Plotting
# ---------------------------
def make_bridge_trend_figure(bridge_id):
    matched = find_best_bridge_match(bridge_id)
    if matched is None or matched not in pivot_df.index:
        return None

    row = pivot_df.loc[matched]
    years = list(row.index)
    values = list(row.values)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(years, values, marker="o", linewidth=2)
    ax.set_title(f"Bridge Trend: {matched}", fontsize=12, pad=10)
    ax.set_xlabel("Year")
    ax.set_ylabel("Bridge Health Index (Overall)")
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

    years = subset.columns.tolist()
    median_trend = subset.median(axis=0)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for _, row in subset.iterrows():
        ax.plot(years, row.values, alpha=0.08, linewidth=1)

    ax.plot(years, median_trend.values, linewidth=3, marker="o")
    ax.set_title(f"Cluster {cluster_id} Median Deterioration Trend", fontsize=12, pad=10)
    ax.set_xlabel("Year")
    ax.set_ylabel("Bridge Health Index (Overall)")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig

# ---------------------------
# Bedrock prompt + tools
# ---------------------------
SYSTEM_PROMPT = f"""
You are a bridge deterioration analysis assistant.

You answer questions about steel bridge Bridge Health Index (BHI) time-series data,
bridge-level deterioration trends, and KMeans clustering results.

Important:
- The clustering was performed using the raw interpolated pivot table of
  'Bridge Health Index (Overall)' by year, without standardization.
- The number of clusters is {N_CLUSTERS}.

Available analysis concepts:
- bridge profile
- bridge trend over time
- comparison between two bridges
- cluster summary
- overall dataset summary
- worst bridges in a specific year
- best bridges in a specific year
- fastest deteriorating bridges based on bridge-level slope

Rules:
- Use tools whenever a user asks about the data.
- Do not invent numeric values.
- Keep answers concise, clear, and grounded in the data.
- If a user asks for a plot, chart, trend, or visualize request, use the matching tool.
- When the overall dataset summary is requested, keep the text short because tables are shown separately.
"""

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
        return {"text": get_bridge_profile(bridge_id)}

    if tool_name == "bridge_trend":
        bridge_id = tool_input["bridge_id"]
        return {
            "text": get_bridge_trend(bridge_id),
            "bridge_id": bridge_id,
            "show_trend_chart": True
        }

    if tool_name == "compare_bridges":
        bridge_id_1 = tool_input["bridge_id_1"]
        bridge_id_2 = tool_input["bridge_id_2"]
        return {
            "text": compare_two_bridges(bridge_id_1, bridge_id_2),
            "bridge_id_1": bridge_id_1,
            "bridge_id_2": bridge_id_2,
            "show_compare_chart": True
        }

    if tool_name == "cluster_summary":
        cluster_id = int(tool_input["cluster_id"])
        return {
            "text": get_cluster_summary(cluster_id),
            "cluster_id": cluster_id,
            "show_cluster_chart": True
        }

    if tool_name == "top_deteriorating_bridges":
        top_n = int(tool_input.get("top_n", 5))
        return {"text": get_top_deteriorating_bridges(top_n=top_n)}

    if tool_name == "top_best_bridges_year":
        year = int(tool_input["year"])
        top_n = int(tool_input.get("top_n", 5))
        return {"text": get_top_best_bridges(year, top_n=top_n)}

    if tool_name == "top_worst_bridges_year":
        year = int(tool_input["year"])
        top_n = int(tool_input.get("top_n", 5))
        return {"text": get_top_worst_bridges(year, top_n=top_n)}

    return {"text": f"Unknown tool: {tool_name}"}

# ---------------------------
# Bedrock conversation
# ---------------------------
def extract_text_from_content_blocks(content_blocks):
    parts = []
    for block in content_blocks:
        if "text" in block:
            parts.append(block["text"])
    return "\n".join(parts).strip()


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
            "cluster_df": None
        }
    except Exception as e:
        return {
            "text": f"Unexpected Bedrock error: {e}",
            "chart": None,
            "summary_df": None,
            "cluster_df": None
        }

    while loops < max_loops:
        loops += 1

        output_message = response["output"]["message"]
        stop_reason = response.get("stopReason", "")
        messages.append(output_message)

        if stop_reason == "end_turn":
            final_text = extract_text_from_content_blocks(output_message["content"])
            if not final_text and pending_summary_df is None and pending_cluster_df is None:
                final_text = "I could not generate a final answer."
            return {
                "text": final_text,
                "chart": pending_chart,
                "summary_df": pending_summary_df,
                "cluster_df": pending_cluster_df
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

                if "summary_df" in result:
                    pending_summary_df = result["summary_df"]

                if "cluster_df" in result:
                    pending_cluster_df = result["cluster_df"]

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

                json_safe_result = {}
                for key, value in result.items():
                    if isinstance(value, pd.DataFrame):
                        json_safe_result[key] = value.to_dict(orient="records")
                    else:
                        json_safe_result[key] = value

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
                    "cluster_df": None
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
                return {
                    "text": f"Bedrock follow-up request failed: {e}",
                    "chart": None,
                    "summary_df": pending_summary_df,
                    "cluster_df": pending_cluster_df
                }
            except Exception as e:
                return {
                    "text": f"Unexpected Bedrock follow-up error: {e}",
                    "chart": None,
                    "summary_df": pending_summary_df,
                    "cluster_df": pending_cluster_df
                }

            continue

        return {
            "text": "I could not complete the request.",
            "chart": None,
            "summary_df": pending_summary_df,
            "cluster_df": pending_cluster_df
        }

    return {
        "text": "The Bedrock tool loop reached its limit.",
        "chart": None,
        "summary_df": pending_summary_df,
        "cluster_df": pending_cluster_df
    }


def answer_question(question):
    result = ask_bedrock_with_tools(question)
    fig = None
    chart = result.get("chart")

    summary_df = result.get("summary_df")
    cluster_df = result.get("cluster_df")

    if chart:
        if chart["type"] == "trend":
            fig = make_bridge_trend_figure(chart["bridge_id"])
        elif chart["type"] == "compare":
            fig = make_compare_bridges_figure(chart["bridge_id_1"], chart["bridge_id_2"])
        elif chart["type"] == "cluster":
            fig = make_cluster_median_figure(chart["cluster_id"])

    return {
        "text": result.get("text"),
        "figure": fig,
        "summary_df": summary_df,
        "cluster_df": cluster_df
    }

# ---------------------------
# Sidebar
# ---------------------------
example_ids = bridge_ids[:3]
example_1 = example_ids[0] if len(example_ids) > 0 else "N/A"
example_2 = example_ids[1] if len(example_ids) > 1 else example_1
example_3 = example_ids[2] if len(example_ids) > 2 else example_1

with st.sidebar:
    st.subheader("Dataset")
    st.write(f"Bridge records: {len(static_df):,}")
    st.write(f"Time-series records: {len(ts_df):,}")
    st.write(f"Usable bridges: {pivot_df.shape[0]:,}")
    st.write(f"Years: {min(years_available)}–{max(years_available)}")
    st.write(f"Time-series file used: {TS_FILE}")
    st.write(f"AWS Region: {AWS_REGION}")
    st.caption("Use a bridge ID from STRUCTURE_NUMBER_008")
    st.write("Example questions:")
    st.markdown(f"""
    - Give me an overview of the bridge deterioration dataset
    - Show trend for bridge {example_1}
    - Compare bridge {example_1} and {example_2}
    - Summarize cluster 2
    - Show the fastest deteriorating bridges
    - Show the 5 worst bridges in 2020
    - Show the 5 best bridges in 2020
    - Give me the profile for bridge {example_3}
    """)

# ---------------------------
# Chat history
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me about bridge deterioration trends, bridge profiles, clusters, or Bridge Health Index patterns."
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("content"):
            st.write(message["content"])

        if "summary_df" in message and message["summary_df"] is not None:
            st.subheader("Dataset Summary")
            st.dataframe(pd.DataFrame(message["summary_df"]), use_container_width=True)

        if "cluster_df" in message and message["cluster_df"] is not None:
            st.subheader("Cluster Distribution")
            st.dataframe(pd.DataFrame(message["cluster_df"]), use_container_width=True)

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

    assistant_message = {
        "role": "assistant",
        "content": result.get("text")
    }

    if result.get("summary_df") is not None:
        assistant_message["summary_df"] = result["summary_df"].to_dict(orient="records")

    if result.get("cluster_df") is not None:
        assistant_message["cluster_df"] = result["cluster_df"].to_dict(orient="records")

    with st.chat_message("assistant"):
        if result.get("text"):
            st.write(result["text"])

        if result.get("summary_df") is not None:
            st.subheader("Dataset Summary")
            st.dataframe(result["summary_df"], use_container_width=True)

        if result.get("cluster_df") is not None:
            st.subheader("Cluster Distribution")
            st.dataframe(result["cluster_df"], use_container_width=True)

        if result.get("figure") is not None:
            figure_key = f"fig_{len(st.session_state.messages)}"
            st.session_state[figure_key] = result["figure"]
            assistant_message["figure_key"] = figure_key
            st.pyplot(result["figure"])

    st.session_state.messages.append(assistant_message)
