import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import streamlit.components.v1 as components
import base64
import time

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Crypto Volatility Visualizer", layout="wide", page_icon="🚀")

# CSS for vibrant UI and improved visibility
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; border-radius: 10px; padding: 15px; }
    h1, h2, h3 { color: #00f2ff !important; text-shadow: 2px 2px 4px #000000; }
    .stNumberInput, .stSelectbox, .stSlider { background-color: #1f2937; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 Crypto Volatility Visualizer Pro")
st.markdown("---")

# -----------------------------
# Cached data loading
# -----------------------------
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

# -----------------------------
# Multi Crypto Selector
# -----------------------------
st.sidebar.subheader("🪙 Select Cryptocurrency")

crypto_files = {
    "Bitcoin (BTC)": "https://raw.githubusercontent.com/sri133/1000408_SriPrasath.P_FA2_Crypto-volatility-Vistualizer/main/btcusd_1-min_data.csv.crdownload",
    "Ethereum (ETH)": "https://raw.githubusercontent.com/sri133/1000408_SriPrasath.P_FA2_Crypto-volatility-Vistualizer/main/btcusd_1-min_data.csv.crdownload",
    "Litecoin (LTC)": "https://raw.githubusercontent.com/sri133/1000408_SriPrasath.P_FA2_Crypto-volatility-Vistualizer/main/btcusd_1-min_data.csv.crdownload"
}

selected_crypto = st.sidebar.selectbox(
    "Choose Crypto:",
    list(crypto_files.keys())
)

try:
    df = load_data(crypto_files[selected_crypto])
    st.sidebar.success(f"Loaded: {selected_crypto}")
except Exception as e:
    st.error(f"❌ Error loading crypto data: {e}")
    st.stop()


# -----------------------------
# Sidebar Enhancements
# -----------------------------
st.sidebar.header("🛠️ Simulation Dashboard")

st.sidebar.subheader("💎 Volatility Presets")
preset = st.sidebar.selectbox(
    "Select Risk Level:",
    ["Manual Control", "Stable (Institutional)", "Low Risk (Blue Chip)", "High Risk (Degenerate)"]
)

if preset == "Stable (Institutional)":
    amp_val, freq_val, drift_val = 200, 2, 100
elif preset == "Low Risk (Blue Chip)":
    amp_val, freq_val, drift_val = 800, 5, 300
elif preset == "High Risk (Degenerate)":
    amp_val, freq_val, drift_val = 4000, 15, -500
else:
    amp_val, freq_val, drift_val = 1000, 6, 0

st.sidebar.subheader("🎨 Customize Pattern")
amplitude = st.sidebar.slider("📏 Amplitude (Swing)", 100, 5000, amp_val)
frequency = st.sidebar.slider("🔄 Frequency (Speed)", 1, 20, freq_val)
drift = st.sidebar.slider("📉 Drift (Trend)", -2000, 2000, drift_val)

st.sidebar.subheader("⚡ Stress Testing")
add_shock = st.sidebar.toggle("💥 Add Market Shock")
shock_type = st.sidebar.radio("Shock Type:", ["Flash Crash", "Moon Spike"]) if add_shock else None

comparison_mode = st.sidebar.checkbox("⚖️ Enable Comparison Mode")

# -----------------------------
# Data Processing
# -----------------------------
if "Timestamp" in df.columns:
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.sort_values("Timestamp")

df.rename(columns={"Open": "Open_Price", "High": "High_Price", "Low": "Low_Price", "Close": "Close_Price"}, inplace=True)
df = df.fillna(df.mean(numeric_only=True)).dropna()

# Range Selection
st.subheader("🔍 Select Data Range")
col_start, col_end = st.columns(2)
start_row = col_start.number_input("Start row:", 0, len(df)-1, 0)
end_row = col_end.number_input("End row:", int(start_row)+1, len(df), min(500, len(df)))
subset_df = df.iloc[int(start_row):int(end_row)]

# -----------------------------
# Pattern Simulator Section
# -----------------------------
st.header("🎛️ Crypto Pattern Simulator")
pattern_type = st.selectbox("Choose price movement pattern:", ["Real Data", "Sine Wave", "Cosine Wave", "Random Noise"])

pattern_df = subset_df.copy()
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

synthetic_series = base_price + wave + drift_line
if add_shock:
    shock_point = n // 2
    synthetic_series[shock_point:] += (amplitude * 3) if shock_type == "Moon Spike" else -(amplitude * 3)

pattern_df["Synthetic_Price"] = synthetic_series

if comparison_mode:
    c1, c2 = st.columns(2)
    stable_price = base_price + (amplitude / 4) * np.sin(x) + drift_line
    volatile_price = base_price + (amplitude * 2) * np.sin(x) + drift_line
    with c1:
        st.markdown("### 🛡️ Stable (Low Risk)")
        st.plotly_chart(px.line(pattern_df, x="Timestamp", y=stable_price, template="plotly_dark", color_discrete_sequence=['#00ff88']), use_container_width=True)
    with c2:
        st.markdown("### 📈 Volatile (High Risk)")
        st.plotly_chart(px.line(pattern_df, x="Timestamp", y=volatile_price, template="plotly_dark", color_discrete_sequence=['#ff3366']), use_container_width=True)
else:
    # Simulator Line with glow effect logic
    fig_sim = px.line(pattern_df, x="Timestamp", y="Synthetic_Price", title=f"✨ {pattern_type} Simulation", template="plotly_dark", color_discrete_sequence=['#00f2ff'])
    st.plotly_chart(fig_sim, use_container_width=True)

# -----------------------------
# Enhanced Visualizations
# -----------------------------
st.header("📊 Market Insights & Sync'd Analytics")

# 1. Price Trends (Synched with Volume Colors)
st.subheader("💰 Synced Price & Volume Intensity")
fig_sync = go.Figure()
fig_sync.add_trace(go.Scatter(
    x=subset_df["Timestamp"], 
    y=subset_df["Close_Price"],
    mode='lines+markers',
    marker=dict(
        size=4,
        color=subset_df["Volume"], # Color points by volume
        colorscale='Viridis',
        showscale=False
    ),
    line=dict(color='rgba(255,255,255,0.4)', width=1),
    name="Price"
))
fig_sync.update_layout(template="plotly_dark", xaxis_title="Time", yaxis_title="Price (USD)")
st.plotly_chart(fig_sync, use_container_width=True)

# 2. High vs Low
st.subheader("⚖️ High vs Low Price Comparison")
fig_hl = go.Figure()
fig_hl.add_trace(go.Scatter(x=subset_df["Timestamp"], y=subset_df["High_Price"], name="High", line=dict(color='#00ff88', width=2)))
fig_hl.add_trace(go.Scatter(x=subset_df["Timestamp"], y=subset_df["Low_Price"], name="Low", line=dict(color='#ff3366', width=2)))
fig_hl.update_layout(template="plotly_dark")
st.plotly_chart(fig_hl, use_container_width=True)

# 3. Trading Volume Analysis (Glow Neon Style)
st.subheader("🔊 Trading Volume Analysis")

fig_vol_bar = go.Figure()

fig_vol_bar.add_trace(go.Bar(
    x=subset_df["Timestamp"],
    y=subset_df["Volume"],
    marker=dict(
        color=subset_df["Volume"],
        colorscale="Viridis",
        line=dict(width=0),
        showscale=True,
        colorbar=dict(title="Volume")
    ),
    hovertemplate="Time: %{x}<br>Volume: %{y}<extra></extra>"
))

fig_vol_bar.update_layout(
    template="plotly_dark",
    plot_bgcolor="#05070d",
    paper_bgcolor="#05070d",
    xaxis=dict(
        title="",
        showgrid=False,
        zeroline=False,
        color="#00f2ff"
    ),
    yaxis=dict(
        title="Volume",
        gridcolor="rgba(255,255,255,0.15)",
        color="#00f2ff"
    ),
    bargap=0.15,
    showlegend=False
)

st.plotly_chart(fig_vol_bar, use_container_width=True)


# 4. Volatility Scatter
st.subheader("⚡ Stable vs Volatile Periods")
vol_df = subset_df.copy()
vol_df["Price_Change"] = vol_df["Close_Price"].diff().abs()
threshold = vol_df["Price_Change"].mean()
vol_df["Status"] = vol_df["Price_Change"].apply(lambda x: "⚡ Volatile" if x > threshold else "🛡️ Stable")
fig_vol = px.scatter(
    vol_df, x="Timestamp", y="Close_Price", color="Status",
    color_discrete_map={"⚡ Volatile": "#ff00ff", "🛡️ Stable": "#00d4ff"}, # Neon Pink and Cyan
    template="plotly_dark"
)
st.plotly_chart(fig_vol, use_container_width=True)

st.success("✅ Volume visibility improved and charts synchronized!")

# ============================================================
# 🚀 ADVANCED FEATURES MODULE (ADD BELOW EXISTING CODE ONLY)
# ============================================================

st.markdown("---")
st.header("🚀 Advanced Live Trading Features")

# ------------------------------------------------------------
# 1. Live Crypto Data Dashboard (FINAL FIX)
# ------------------------------------------------------------
st.subheader("📡 Live Crypto Market Feed")

live_crypto = st.selectbox(
    "Choose Live Crypto:",
    ["BTC-USD", "ETH-USD", "SOL-USD"]
)

period = st.selectbox(
    "Time Range:",
    ["1d", "7d", "1mo"]
)

if st.button("🔄 Fetch Live Data"):

    try:
        live_data = yf.download(
            live_crypto,
            period=period,
            interval="5m",
            progress=False
        )

        if live_data.empty:
            st.warning("⚠️ No data returned.")
        else:
            # 🔥 Flatten MultiIndex columns
            if isinstance(live_data.columns, pd.MultiIndex):
                live_data.columns = live_data.columns.get_level_values(0)

            live_data = live_data.reset_index()

            # Detect time column automatically
            time_col = "Datetime" if "Datetime" in live_data.columns else "Date"

            fig_live = px.line(
                live_data,
                x=time_col,
                y="Close",
                title=f"Live {live_crypto} Price",
                template="plotly_dark"
            )

            st.plotly_chart(fig_live, use_container_width=True)

    except Exception as e:
        st.error(f"Live data error: {e}")



# ------------------------------------------------------------
# 2. AI Price Prediction Engine
# ------------------------------------------------------------
st.subheader("🤖 AI Price Prediction")

if st.button("Run AI Prediction"):

    prices = subset_df["Close_Price"].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(prices)

    X, y = [], []
    window = 20

    for i in range(window, len(scaled_prices)):
        X.append(scaled_prices[i-window:i])
        y.append(scaled_prices[i])

    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], -1)

    model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500)
    model.fit(X, y.ravel())

    future_input = scaled_prices[-window:].reshape(1, -1)
    prediction = model.predict(future_input)

    predicted_price = scaler.inverse_transform(
        prediction.reshape(-1, 1)
    )[0][0]

    st.metric("🔮 Predicted Next Price", f"${predicted_price:.2f}")

# ------------------------------------------------------------
# 3. TradingView Professional Widget
# ------------------------------------------------------------
st.subheader("📊 Professional Trading Chart")

tradingview_html = f"""
<div class="tradingview-widget-container">
  <iframe
    src="https://s.tradingview.com/widgetembed/?symbol={live_crypto}&interval=1&theme=dark"
    width="100%"
    height="500"
    frameborder="0">
  </iframe>
</div>
"""

components.html(tradingview_html, height=520)

# ------------------------------------------------------------
# 4. Real-Time Simulation Mode
# ------------------------------------------------------------
sim_speed = st.slider("⚡ Simulation Speed", 0.001, 0.1, 0.02)

if st.button("▶ Start Simulation"):

    sim_placeholder = st.empty()

    for i in range(50, len(subset_df), 20):  # bigger step = faster jump

        sim_fig = px.line(
            subset_df.iloc[:i],
            x="Timestamp",
            y="Close_Price",
            template="plotly_dark"
        )

        sim_placeholder.plotly_chart(sim_fig, use_container_width=True)
        time.sleep(sim_speed)


# ------------------------------------------------------------
# 5. Export Data Feature
# ------------------------------------------------------------
st.subheader("💾 Export Current Dataset")

csv = subset_df.to_csv(index=False)
b64 = base64.b64encode(csv.encode()).decode()

download_link = f"""
<a href="data:file/csv;base64,{b64}" download="crypto_data.csv">
📥 Download CSV File
</a>
"""

st.markdown(download_link, unsafe_allow_html=True)

st.success("✅ Advanced features loaded successfully!")

# ============================================================
# 📊 ADVANCED DASHBOARD ANALYTICS PANEL
# ============================================================

st.markdown("---")
st.header("📊 Advanced Dashboard Analytics")

# ------------------------------------------------------------
# 1. Gains Ratio Donut Chart
# ------------------------------------------------------------
st.subheader("💠 Gains Ratio")

# Create synthetic asset groups from existing data
gains_data = {
    "BTC Trend": subset_df["Close_Price"].pct_change().abs().mean(),
    "Volume Strength": subset_df["Volume"].mean(),
    "High Momentum": subset_df["High_Price"].mean(),
    "Low Stability": subset_df["Low_Price"].mean()
}

gains_df = pd.DataFrame({
    "Category": list(gains_data.keys()),
    "Value": list(gains_data.values())
})

fig_donut = px.pie(
    gains_df,
    values="Value",
    names="Category",
    hole=0.65,
    template="plotly_dark",
    color_discrete_sequence=px.colors.sequential.Tealgrn
)

fig_donut.update_traces(
    textinfo="percent+label",
    pull=[0.03]*len(gains_df)
)

st.plotly_chart(fig_donut, use_container_width=True)

# ------------------------------------------------------------
# 2. Smooth Multi-Line Analysis Chart
# ------------------------------------------------------------
st.subheader("📈 Multi-Metric Analysis")

analysis_df = subset_df.copy()

analysis_df["Trend"] = analysis_df["Close_Price"].rolling(20).mean()
analysis_df["Momentum"] = analysis_df["Close_Price"].diff()
analysis_df["Volatility"] = analysis_df["Close_Price"].rolling(20).std()

fig_analysis = go.Figure()

fig_analysis.add_trace(go.Scatter(
    x=analysis_df["Timestamp"],
    y=analysis_df["Trend"],
    mode="lines",
    name="Trend",
    line=dict(width=3, shape="spline")
))

fig_analysis.add_trace(go.Scatter(
    x=analysis_df["Timestamp"],
    y=analysis_df["Momentum"],
    mode="lines",
    name="Momentum",
    line=dict(width=3, shape="spline")
))

fig_analysis.add_trace(go.Scatter(
    x=analysis_df["Timestamp"],
    y=analysis_df["Volatility"],
    mode="lines",
    name="Volatility",
    line=dict(width=3, shape="spline")
))

fig_analysis.update_layout(
    template="plotly_dark",
    xaxis_title="Time",
    yaxis_title="Value",
    hovermode="x unified"
)

st.plotly_chart(fig_analysis, use_container_width=True)

st.success("✅ Advanced dashboard analytics loaded!")

# ============================================================
# 📊 PRO TRADING DASHBOARD PANELS
# ============================================================

st.markdown("---")
st.header("📊 Pro Trading Dashboard")

# ------------------------------------------------------------
# 1. Trade Graph (SAFE VERSION)
# ------------------------------------------------------------
st.subheader("📈 Trade Graph")

# Ensure data exists
if subset_df.empty or len(subset_df) < 5:
    st.warning("⚠️ Not enough data to draw trade graph.")
else:
    trade_df = subset_df.dropna().tail(100).copy()

    # Ensure timestamp exists
    if "Timestamp" not in trade_df.columns:
        st.error("❌ Timestamp column missing.")
    else:
        fig_trade = go.Figure()

        # Volume bars
        fig_trade.add_trace(go.Bar(
            x=trade_df["Timestamp"],
            y=trade_df["Volume"],
            name="Volume",
            opacity=0.6
        ))

        # Smooth close price line
        fig_trade.add_trace(go.Scatter(
            x=trade_df["Timestamp"],
            y=trade_df["Close_Price"],
            mode="lines",
            name="Close Price",
            line=dict(width=3, shape="spline")
        ))

        fig_trade.update_layout(
            template="plotly_dark",
            hovermode="x unified",
            xaxis_title="Time",
            yaxis_title="Market Activity"
        )

        st.plotly_chart(fig_trade, use_container_width=True)


# ------------------------------------------------------------
# 2. Stat Overview KPI Cards
# ------------------------------------------------------------
st.subheader("📊 Stat Overview")

profit = subset_df["Close_Price"].iloc[-1] - subset_df["Close_Price"].iloc[0]
equity = subset_df["Close_Price"].mean() * 10
expectancy = subset_df["High_Price"].mean()
gain_pct = (profit / subset_df["Close_Price"].iloc[0]) * 100
ratio = subset_df["High_Price"].mean() / subset_df["Low_Price"].mean()
avg_trade = subset_df["Volume"].mean()

col1, col2, col3, col4, col5, col6 = st.columns(6)

col1.metric("💰 Profit", f"${profit:,.0f}")
col2.metric("⚖️ Equality", f"${equity:,.0f}")
col3.metric("📊 Expectancy", f"${expectancy:,.0f}")
col4.metric("📈 Gain", f"{gain_pct:.2f}%")
col5.metric("🔁 Ratio", f"{ratio:.2f}")
col6.metric("📦 Avg", f"${avg_trade:,.0f}")

st.success("✅ Pro dashboard panels loaded!")







