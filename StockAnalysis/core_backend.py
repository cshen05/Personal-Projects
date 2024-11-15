import yfinance as yf
import pandas as pd
import heapq
from sklearn.metrics import accuracy_score

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

    # Moving Averages
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()

    # Exponential Moving Averages
    data['EMA10'] = data['Close'].ewm(span=10, adjust=False).mean()
    data['EMA50'] = data['Close'].ewm(span=50, adjust=False).mean()

    # RSI
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    rolling_std = data['Close'].rolling(window=10).std()
    data['BB_upper'] = data['MA10'] + 2 * rolling_std
    data['BB_lower'] = data['MA10'] - 2 * rolling_std

    # Average True Range (ATR)
    data['TR'] = data[['High', 'Low', 'Close']].apply(
        lambda row: max(row['High'] - row['Low'], abs(row['High'] - row['Close']), abs(row['Low'] - row['Close'])),
        axis=1
    )
    data['ATR'] = data['TR'].rolling(window=14).mean()

    # Stochastic Oscillator
    data['Stochastic'] = (
        (data['Close'] - data['Low'].rolling(14).min()) /
        (data['High'].rolling(14).max() - data['Low'].rolling(14).min())
    ) * 100

    # On-Balance Volume (OBV)
    data['OBV'] = (data['Volume'] * (data['Close'] - data['Close'].shift(1)).apply(lambda x: 1 if x > 0 else -1)).cumsum()

    # Average Directional Index (ADX)
    data['DM_plus'] = (data['High'] - data['High'].shift(1)).apply(lambda x: x if x > 0 else 0)
    data['DM_minus'] = (data['Low'].shift(1) - data['Low']).apply(lambda x: x if x > 0 else 0)
    data['ADX'] = (
        (data['DM_plus'] - data['DM_minus']).abs().rolling(window=14).mean() / data['TR'].rolling(window=14).mean()
    )

    # Target Variable
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
def provide_insight(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    # Confidence level for the latest prediction
    last_prediction_confidence = max(probabilities[-1])
    last_prediction = predictions[-1]

    # Generate recommendation
    if last_prediction == 1:
        recommendation = "BUY - The model predicts a rise in stock price."
    else:
        recommendation = "SELL - The model predicts a decline in stock price."

    # Overall model accuracy
    model_accuracy = accuracy_score(y_test, predictions)

    return (
        f"{recommendation} Confidence level: {last_prediction_confidence:.2f}. "
        f"Overall model accuracy: {model_accuracy:.2%}.",
        last_prediction_confidence,
    )
