from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import heapq
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    return data

# Add features
def add_features(data):
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    data.dropna(subset=['Close'], inplace=True)
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    rolling_std = data['Close'].rolling(window=10).std()
    data['BB_upper'] = data['MA10'] + 2 * rolling_std
    data['BB_lower'] = data['MA10'] - 2 * rolling_std
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    data.dropna(inplace=True)
    return data

# Rank features
def rank_features(model, feature_names):
    feature_importance = model.feature_importances_
    feature_tuples = [(importance, feature) for importance, feature in zip(feature_importance, feature_names)]
    ranked_features = heapq.nlargest(len(feature_tuples), feature_tuples)
    return ranked_features

# Provide recommendation
def provide_insight(data):
    latest_rsi = data['RSI'].iloc[-1]
    recommendation = ""
    if latest_rsi < 30:
        recommendation = "BUY - The stock appears oversold."
    elif latest_rsi > 70:
        recommendation = "SELL - The stock appears overbought."
    else:
        recommendation = "HOLD - The stock is in a neutral range."
    return recommendation, latest_rsi

# Plot graph
def plot_trader_graph(data):
    last_5_days = data.tail(5)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(last_5_days.index, last_5_days['Close'], label="Close Price", marker="o", color="blue", linewidth=2)
    ax.fill_between(last_5_days.index, last_5_days['BB_upper'], last_5_days['BB_lower'], color="orange", alpha=0.2, label="Bollinger Bands")
    ax.plot(last_5_days.index, last_5_days['BB_upper'], linestyle="--", color="orange", linewidth=1)
    ax.plot(last_5_days.index, last_5_days['BB_lower'], linestyle="--", color="orange", linewidth=1)
    ax.plot(last_5_days.index, last_5_days['MA10'], label="10-Day MA", linestyle="-.", color="green", linewidth=2)
    ax.plot(last_5_days.index, last_5_days['MA50'], label="50-Day MA", linestyle=":", color="purple", linewidth=2)
    ax.set_title("Stock Price and Indicators (Last 5 Days)", fontsize=14)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price ($)", fontsize=12)
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    return fig
