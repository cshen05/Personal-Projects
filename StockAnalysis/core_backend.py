import yfinance as yf
import pandas as pd
import heapq

def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    return data

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

def rank_features(model, feature_names):
    feature_importance = model.feature_importances_
    feature_tuples = [(importance, feature) for importance, feature in zip(feature_importance, feature_names)]
    ranked_features = heapq.nlargest(len(feature_tuples), feature_tuples)
    return ranked_features

def provide_insight(model, X_test, y_test):
    """
    Provide recommendation based on the model's prediction.
    Returns a recommendation string and the numeric confidence level.
    """
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    # Get the confidence level for the latest prediction
    last_prediction_confidence = max(probabilities[-1])  # Confidence of the last prediction

    # Determine recommendation based on the model's prediction
    last_prediction = predictions[-1]
    recommendation = ""
    if last_prediction == 1:
        recommendation = "BUY - The model predicts a rise in stock price."
    else:
        recommendation = "SELL - The model predicts a decline in stock price."

    # Calculate model accuracy for overall context
    model_accuracy = model.score(X_test, y_test) * 100

    return (
        f"{recommendation} Confidence level: {last_prediction_confidence:.2f}. "
        f"Overall model accuracy: {model_accuracy:.2f}%.",
        last_prediction_confidence,
    )
