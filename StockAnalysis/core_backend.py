from datetime import datetime
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import heapq


def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch stock data using yfinance and standardize column names.
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    return data


def add_features(data):
    """
    Add technical indicators and a Target column to the dataset.
    """
    # Add Moving Averages
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()

    # Add Relative Strength Index (RSI)
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # Add Bollinger Bands
    data['BB_upper'] = data['MA10'] + 2 * data['Close'].rolling(window=10).std()
    data['BB_lower'] = data['MA10'] - 2 * data['Close'].rolling(window=10).std()

    # Add Target column: 1 for Buy, -1 for Sell, 0 for Hold
    data['Target'] = 0  # Default to Hold
    data.loc[data['Close'].shift(-1) > data['Close'], 'Target'] = 1  # Buy
    data.loc[data['Close'].shift(-1) < data['Close'], 'Target'] = -1  # Sell

    # Drop rows with NaN values created by rolling calculations
    data = data.dropna()

    return data


def train_model(data, features):
    """
    Train a Random Forest model on the provided dataset.
    """
    X = data[features]
    y = data['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model


def rank_features(model, features):
    """
    Rank features using Random Forest feature importance and a min heap.
    """
    importances = model.feature_importances_
    ranked_features = []

    # Use a heap to store features by their importance
    for i, feature in enumerate(features):
        heapq.heappush(ranked_features, (-importances[i], feature))  # Negative for max heap behavior

    # Sort ranked features in descending order of importance
    ranked_features = [(-importance, feature) for importance, feature in sorted(ranked_features)]
    return ranked_features


def provide_insight(data):
    """
    Provide explanatory insights about the latest stock data.
    """
    # Fetch the most recent RSI for explanation purposes
    rsi = data['RSI'].iloc[-1]

    # Provide a basic explanation for RSI
    if rsi < 30:
        insight = "The RSI indicates the stock is oversold. Historically, this suggests a potential buying opportunity."
    elif rsi > 70:
        insight = "The RSI indicates the stock is overbought. Historically, this suggests a potential selling opportunity."
    else:
        insight = "The RSI indicates the stock is in a neutral range. No strong trend detected."

    return insight, rsi


def plot_trader_graph(data):
    """
    Generate a matplotlib figure for stock price and technical indicators.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data.index[-5:], data['Close'][-5:], label="Close Price", marker='o', color='blue')
    ax.fill_between(data.index[-5:], data['BB_upper'][-5:], data['BB_lower'][-5:], color='orange', alpha=0.2, label="Bollinger Bands")
    ax.plot(data.index[-5:], data['MA10'][-5:], label="10-Day MA", linestyle='--', color='green')
    ax.plot(data.index[-5:], data['MA50'][-5:], label="50-Day MA", linestyle=':', color='purple')

    ax.set_title("Stock Price and Indicators (Last 5 Days)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def main():
    print("Welcome to the Stock Analysis App!")
    print("Please follow the instructions to analyze a stock.\n")

    # Input Ticker and Date Range
    ticker = input("Enter the stock ticker (e.g., AAPL): ").upper()
    start_date = input("Enter the start date (YYYY-MM-DD): ")
    end_date = input("Enter the end date (YYYY-MM-DD): ")

    try:
        # Fetch and preprocess data
        data = fetch_stock_data(ticker, start_date, end_date)
        data = add_features(data)

        # Plot the graph
        print("\nPlotting stock trends and indicators...")
        plot_trader_graph(data)

        # Train the model and rank features
        features = ['MA10', 'MA50', 'RSI', 'BB_upper', 'BB_lower']
        model = train_model(data, features)
        ranked_features = rank_features(model, features)

        print("\nFeature Importance Rankings:")
        for rank, (importance, feature) in enumerate(ranked_features, start=1):
            print(f"{rank}. {feature}: {importance:.4f}")

        # Provide recommendation and insights
        latest_data = data[features].iloc[-1:].values
        prediction = model.predict(latest_data)[0]
        recommendation = "Hold"
        if prediction == 1:
            recommendation = "Buy - The stock is expected to increase in value."
        elif prediction == -1:
            recommendation = "Sell - The stock is expected to decrease in value."

        print("\nRecommendation:", recommendation)

        insight, rsi = provide_insight(data)
        print(f"RSI Insight: {insight} (RSI: {rsi:.2f})")

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
