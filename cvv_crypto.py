import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Crypto Data Preparation", layout="wide")
st.title("📊 Crypto Volatility Visualizer")

# -----------------------------
# Cached data loading
# -----------------------------
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

file_path = "https://raw.githubusercontent.com/sri133/1000408_SriPrasath.P_FA2_Crypto-volatility-Vistualizer/main/btcusd_1-min_data.csv.crdownload"

try:
    df = load_data(file_path)
except Exception as e:
    st.error(f"❌ Error loading CSV file: {e}")
    st.stop()

# -----------------------------
# Raw preview
# -----------------------------
st.subheader("🔹 Raw Dataset Preview")
st.write(df.head(1000))
st.caption("Showing first 1000 rows for performance")

# -----------------------------
# Dataset overview
# -----------------------------
st.subheader("🔹 Dataset Overview")

col1, col2 = st.columns(2)

with col1:
    st.write("Dataset Shape:", df.shape)
    st.write("Columns:", df.columns.tolist())

with col2:
    st.write("Missing values:")
    st.write(df.isnull().sum())

# -----------------------------
# Data cleaning
# -----------------------------
if "Timestamp" in df.columns:
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.sort_values("Timestamp")

df.rename(columns={
    "Open": "Open_Price",
    "High": "High_Price",
    "Low": "Low_Price",
    "Close": "Close_Price"
}, inplace=True)

df = df.fillna(df.mean(numeric_only=True))
df = df.dropna()

# -----------------------------
# Range selection
# -----------------------------
st.subheader("🔹 Select Data Range to Visualize")

col1, col2 = st.columns(2)

with col1:
    start_row = st.number_input(
        "Start row:",
        min_value=0,
        max_value=len(df) - 1,
        value=0,
        step=1
    )

with col2:
    end_row = st.number_input(
        "End row:",
        min_value=int(start_row) + 1,
        max_value=len(df),
        value=min(500, len(df)),
        step=1
    )

subset_df = df.iloc[int(start_row):int(end_row)]

st.write(f"Showing rows from **{start_row}** to **{end_row}**")

# -----------------------------
# Cleaned subset preview
# -----------------------------
st.subheader("🔹 Cleaned Subset Preview")
st.write(subset_df.head(1000))
st.caption("Showing first 1000 rows for performance")

# -----------------------------
# Statistics
# -----------------------------
if "Close_Price" in subset_df.columns:
    st.subheader("🔹 Close Price Statistics")
    st.write(subset_df["Close_Price"].describe())

st.success("✅ Complete — Data cleaned and ready!")

# =============================
# Pattern Simulator Section
# =============================

st.header("🎛 Crypto Pattern Simulator")

# -----------------------------
# Pattern Selector
# -----------------------------
pattern_type = st.selectbox(
    "Choose price movement pattern:",
    ["Real Data", "Sine Wave", "Cosine Wave", "Random Noise"]
)

# -----------------------------
# Controls
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    amplitude = st.slider(
        "Amplitude (Swing Size)",
        100, 5000, 1000, step=100
    )

    frequency = st.slider(
        "Frequency (Swing Speed)",
        1, 20, 6
    )

with col2:
    drift = st.slider(
        "Drift (Trend Direction)",
        -2000, 2000, 0, step=100
    )

comparison_mode = st.checkbox("Enable Comparison Mode")

# -----------------------------
# Pattern generation
# -----------------------------
pattern_df = subset_df.copy()

if "Timestamp" in pattern_df.columns and "Close_Price" in pattern_df.columns:

    n = len(pattern_df)
    base_price = pattern_df["Close_Price"].mean()

    x = np.linspace(0, frequency * np.pi, n)
    drift_line = np.linspace(0, drift, n)

    if pattern_type == "Sine Wave":
        wave = amplitude * np.sin(x)

    elif pattern_type == "Cosine Wave":
        wave = amplitude * np.cos(x)

    elif pattern_type == "Random Noise":
        wave = np.random.normal(0, amplitude / 2, n)

    else:
        wave = pattern_df["Close_Price"] - base_price

    pattern_df["Synthetic_Price"] = base_price + wave + drift_line

# -----------------------------
# Visualization
# -----------------------------
st.subheader("🔹 Pattern Visualization")

if comparison_mode:

    col1, col2 = st.columns(2)

    stable_wave = (amplitude / 4) * np.sin(x)
    volatile_wave = (amplitude * 2) * np.sin(x)

    stable_price = base_price + stable_wave + drift_line
    volatile_price = base_price + volatile_wave + drift_line

    stable_df = pattern_df.copy()
    volatile_df = pattern_df.copy()

    stable_df["Price"] = stable_price
    volatile_df["Price"] = volatile_price

    with col1:
        st.markdown("### 📉 Stable (Small Swings)")
        fig_stable = px.line(
            stable_df,
            x="Timestamp",
            y="Price"
        )
        st.plotly_chart(fig_stable, use_container_width=True)

    with col2:
        st.markdown("### 📈 Volatile (Large Swings)")
        fig_volatile = px.line(
            volatile_df,
            x="Timestamp",
            y="Price"
        )
        st.plotly_chart(fig_volatile, use_container_width=True)

else:

    fig_pattern = px.line(
        pattern_df,
        x="Timestamp",
        y="Synthetic_Price",
        title=f"{pattern_type} Pattern Simulation"
    )

    st.plotly_chart(fig_pattern, use_container_width=True)

# =============================
# Original Visualizations
# =============================

st.header("📈 Bitcoin Visualizations")

MAX_POINTS = 5000

if len(subset_df) > MAX_POINTS:
    step = max(1, len(subset_df) // MAX_POINTS)
    plot_df = subset_df.iloc[::step]
else:
    plot_df = subset_df.copy()

# Price over time
if "Timestamp" in plot_df.columns and "Close_Price" in plot_df.columns:

    st.subheader("🔹 Bitcoin Close Price Over Time")

    fig_price = px.line(
        plot_df,
        x="Timestamp",
        y="Close_Price"
    )

    st.plotly_chart(fig_price, use_container_width=True)

# High vs Low
if {"Timestamp", "High_Price", "Low_Price"}.issubset(plot_df.columns):

    st.subheader("🔹 High vs Low Price Comparison")

    fig_hl = go.Figure()

    fig_hl.add_trace(go.Scatter(
        x=plot_df["Timestamp"],
        y=plot_df["High_Price"],
        mode="lines",
        name="High Price"
    ))

    fig_hl.add_trace(go.Scatter(
        x=plot_df["Timestamp"],
        y=plot_df["Low_Price"],
        mode="lines",
        name="Low Price"
    ))

    st.plotly_chart(fig_hl, use_container_width=True)

# Volume
if {"Timestamp", "Volume"}.issubset(plot_df.columns):

    st.subheader("🔹 Trading Volume Analysis")

    fig_volume = px.bar(
        plot_df,
        x="Timestamp",
        y="Volume"
    )

    st.plotly_chart(fig_volume, use_container_width=True)

# Volatility
if {"Timestamp", "Close_Price"}.issubset(plot_df.columns):

    st.subheader("🔹 Stable vs Volatile Periods")

    vol_df = plot_df.copy()
    vol_df["Price_Change"] = vol_df["Close_Price"].diff().abs()

    threshold = vol_df["Price_Change"].mean()

    vol_df["Volatility_Label"] = vol_df["Price_Change"].apply(
        lambda x: "Volatile" if x > threshold else "Stable"
    )

    fig_volatility = px.scatter(
        vol_df,
        x="Timestamp",
        y="Close_Price",
        color="Volatility_Label"
    )

    st.plotly_chart(fig_volatility, use_container_width=True)

st.success("✅ Visualizations Generated!")

