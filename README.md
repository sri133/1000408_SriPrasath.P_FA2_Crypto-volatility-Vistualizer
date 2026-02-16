**📊 Crypto Volatility Visualizer**

Simulating Market Swings with Mathematics for AI and Python

📌 Project Overview

The Crypto Volatility Visualizer is an interactive Streamlit dashboard that helps users understand cryptocurrency price behavior through real data and mathematical simulations. This project connects mathematical concepts such as sine waves, cosine waves, drift, and random noise with real-world crypto market volatility.

The app allows users to explore Bitcoin price movements, generate synthetic volatility patterns, and compare stable vs volatile market behavior in an visual, interactive way.

This project was developed as part of the Mathematics for AI coursework (FA-1 & FA-2).

🎯 Objectives

This project aims to:

Analyze real cryptocurrency datasets to understand volatility

Simulate price swings using mathematical functions

Create interactive visualizations for learning financial behavior

Build and deploy a working Streamlit dashboard

Help users compare stable and volatile market patterns

🧠 FA-1: Planning & Design

Before coding, the dashboard was carefully planned using:

✅ Feature Planning

The app includes:

Pattern Selector

Real Data

Sine Wave

Cosine Wave

Random Noise

Amplitude Control

Adjusts swing size of price movements

Frequency Control

Changes speed of price fluctuations

Drift Control

Adds upward or downward trend

Comparison Mode

Displays stable vs volatile patterns side-by-side

✅ Storyboard & Data Flow

The dashboard design includes:

Sidebar with interactive sliders and dropdowns

Main visualization panel for charts

Optional comparison layout with two graphs

Data flow:

User selects parameters

System generates synthetic or real data

Graph updates instantly

💾 Dataset

The project uses a cryptocurrency historical dataset containing:

Timestamp

Open price

High price

Low price

Close price

Trading volume

Data is cleaned and prepared using Pandas before visualization.

Dataset source:
Kaggle Cryptocurrency Price History

⚙️ Features
📊 Data Preparation

CSV dataset loading

Timestamp conversion

Missing value handling

Column renaming for clarity

Subset selection for faster visualization

📈 Visualizations

The app generates:

Bitcoin close price over time

High vs low price comparison

Trading volume bar charts

Stable vs volatile period detection

Synthetic pattern simulations

🎛 Pattern Simulator

Users can simulate crypto price behavior using:

Sine and cosine wave models

Random noise shocks

Adjustable amplitude, frequency, and drift

Side-by-side comparison mode

🛠 Technologies Used

Python

Streamlit

Pandas

NumPy

Plotly 

🌐 Deployment

The app is deployed using Streamlit Cloud.

👉 Live App Link:
(https://1000408sriprasathpfa2crypto-volatility-vistualizer-miapohwnafb.streamlit.app/)

📂 Project Structure
├── app.py                # Main Streamlit application
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation

📊 Learning Outcomes Achieved

This project demonstrates:

Real-world financial data analysis

Mathematical modeling of volatility

Interactive dashboard design

Python data visualization skills

Streamlit app deployment

🔮 Future Improvements

Add more cryptocurrencies

Advanced volatility metrics

Machine learning prediction models

Enhanced UI/UX design

User-uploaded datasets

Credits:

Student Name: Sri Prasath. P

Mentor Name: Syed Ali Beema. S

Course: Mathematics for AI-1

School Name: Jain Vidyalaya IB World School

