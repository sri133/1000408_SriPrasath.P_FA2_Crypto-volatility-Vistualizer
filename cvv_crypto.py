import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Crypto Data Preparation", layout="wide")
st.title("📊 Crypto Volatility Visualizer")

# -----------------------------
# Cached data loading
# -----------------------------
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

file_path = r"C:\Users\LENOVO\Downloads\btcusd_1-min_data.csv.crdownload"

try:
    df = load_data(file_path)
except FileNotFoundError:
    st.error("❌ CSV file not found! Check your file path.")
    st.stop()

# -----------------------------
# Raw preview (limited)
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

st.success("✅Complete — Data cleaned and ready for visualization!")

# =============================
# Stage 5: Visualizations
# =============================

st.header("📈Bitcoin Visualizations")

# -----------------------------
# Downsample for plotting
# -----------------------------
MAX_POINTS = 5000

if len(subset_df) > MAX_POINTS:
    step = max(1, len(subset_df) // MAX_POINTS)
    plot_df = subset_df.iloc[::step]
else:
    plot_df = subset_df.copy()

# -----------------------------
# Price Over Time
# -----------------------------
if "Timestamp" in plot_df.columns and "Close_Price" in plot_df.columns:

    st.subheader("🔹 Bitcoin Close Price Over Time")

    fig_price = px.line(
        plot_df,
        x="Timestamp",
        y="Close_Price",
        title="Bitcoin Price Over Time",
        labels={
            "Timestamp": "Date",
            "Close_Price": "Close Price (USD)"
        }
    )

    st.plotly_chart(fig_price, use_container_width=True)

# -----------------------------
# High vs Low Comparison
# -----------------------------
if (
    "Timestamp" in plot_df.columns and
    "High_Price" in plot_df.columns and
    "Low_Price" in plot_df.columns
):

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

    fig_hl.update_layout(
        title="High vs Low Bitcoin Prices",
        xaxis_title="Date",
        yaxis_title="Price (USD)"
    )

    st.plotly_chart(fig_hl, use_container_width=True)

# -----------------------------
# Volume Analysis
# -----------------------------
if "Timestamp" in plot_df.columns and "Volume" in plot_df.columns:

    st.subheader("🔹 Trading Volume Analysis")

    fig_volume = px.bar(
        plot_df,
        x="Timestamp",
        y="Volume",
        title="Bitcoin Trading Volume",
        labels={
            "Timestamp": "Date",
            "Volume": "Trading Volume"
        }
    )

    st.plotly_chart(fig_volume, use_container_width=True)

# -----------------------------
# Stable vs Volatile Periods
# -----------------------------
if "Timestamp" in plot_df.columns and "Close_Price" in plot_df.columns:

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
        color="Volatility_Label",
        title="Stable vs Volatile Bitcoin Periods",
        labels={
            "Timestamp": "Date",
            "Close_Price": "Close Price (USD)"
        }
    )

    st.plotly_chart(fig_volatility, use_container_width=True)

st.success("✅Visualizations Generated!")





##import streamlit as st
##import pandas as pd
##
##st.set_page_config(page_title="Crypto Data Preparation", layout="wide")
##st.title("📊 Crypto Volatility Visualizer — Stage 4")
##
### Load dataset
##file_path = r"C:\Users\LENOVO\Downloads\btcusd_1-min_data.csv.crdownload"
##
##try:
##    df = pd.read_csv(file_path)
##except FileNotFoundError:
##    st.error("❌ CSV file not found! Check your file path.")
##    st.stop()
##
### Raw preview
##st.subheader("🔹 Raw Dataset Preview")
##st.write(df.head())
##
### Dataset overview
##st.subheader("🔹 Dataset Overview")
##
##col1, col2 = st.columns(2)
##
##with col1:
##    st.write("Dataset Shape:", df.shape)
##    st.write("Columns:", df.columns.tolist())
##
##with col2:
##    st.write("Missing values:")
##    st.write(df.isnull().sum())
##
### Timestamp conversion
##if "Timestamp" in df.columns:
##    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
##    df = df.sort_values("Timestamp")
##
### Rename columns
##df.rename(columns={
##    "Open": "Open_Price",
##    "High": "High_Price",
##    "Low": "Low_Price",
##    "Close": "Close_Price"
##}, inplace=True)
##
### Handle missing values
##df = df.fillna(df.mean(numeric_only=True))
##df = df.dropna()
##
### -----------------------------
### Typing range selection
### -----------------------------
##st.subheader("🔹 Select Data Range to Visualize")
##
##col1, col2 = st.columns(2)
##
##with col1:
##    start_row = st.number_input(
##        "Start row:",
##        min_value=0,
##        max_value=len(df) - 1,
##        value=0,
##        step=1
##    )
##
##with col2:
##    end_row = st.number_input(
##        "End row:",
##        min_value=int(start_row) + 1,
##        max_value=len(df),
##        value=min(500, len(df)),
##        step=1
##    )
##
##subset_df = df.iloc[int(start_row):int(end_row)]
##
##st.write(f"Showing rows from **{start_row}** to **{end_row}**")
##
### Cleaned preview
##st.subheader("🔹 Cleaned Subset Preview")
##st.write(subset_df)
##
### Statistics
##if "Close_Price" in subset_df.columns:
##    st.subheader("🔹 Close Price Statistics")
##    st.write(subset_df["Close_Price"].describe())
##
##st.success("✅ Stage 4 Complete — Data cleaned and ready for visualization!")
