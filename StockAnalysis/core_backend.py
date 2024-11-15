import yfinance as yf
import pandas as pd
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import heapq
from numba import jit
from functools import lru_cache

# Optimized fetch_stock_data with caching and interval support
@lru_cache(maxsize=10)
def fetch_stock_data(ticker, start_date, end_date, interval='1d'):
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    return data

# JIT-optimized RSI calculation
@jit(nopython=True)
def calculate_rsi(close_prices, window=14):
    delta = close_prices.diff().fillna(0).values
    gain = (delta > 0) * delta
    loss = (-delta > 0) * -delta

    avg_gain = gain[:window].mean()
    avg_loss = loss[:window].mean()

    rs = avg_gain / avg_loss
    rsi = [100 - (100 / (1 + rs))]

    for i in range(window, len(delta)):
        avg_gain = (avg_gain * (window - 1) + gain[i]) / window
        avg_loss = (avg_loss * (window - 1) + loss[i]) / window
        rs = avg_gain / avg_loss
        rsi.append(100 - (100 / (1 + rs)))

    return rsi

# Add features with optimizations
def add_features(data):
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    data.dropna(subset=['Close'], inplace=True)

    # Exponential Moving Average (Short-Term Trend)
    data['EMA10'] = data['Close'].ewm(span=10, adjust=False).mean()

    # Moving Average (Long-Term Trend)
    data['MA50'] = data['Close'].rolling(window=50).mean()

    # RSI with JIT optimization
    data['RSI'] = calculate_rsi(data['Close'])

    # Bollinger Bands
    rolling_std = data['Close'].rolling(window=10).std()
    data['BB_upper'] = data['EMA10'] + 2 * rolling_std
    data['BB_lower'] = data['EMA10'] - 2 * rolling_std

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

# Provide recommendation with batch processing
def provide_insight(model, X_test, y_test):
    predictions = []
    probabilities = []
    batch_size = 1000
    for i in range(0, len(X_test), batch_size):
        batch = X_test[i:i + batch_size]
        probabilities.extend(model.predict_proba(batch))
        predictions.extend(model.predict(batch))

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

# Train XGBoost with optimized GridSearch
def train_xgboost(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'learning_rate': [0.1],
    }
    grid_search = GridSearchCV(
        XGBClassifier(eval_metric='logloss', random_state=42),
        param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1  # Utilize all CPU cores
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_
