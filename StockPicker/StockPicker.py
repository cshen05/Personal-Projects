import os
import pandas as pd
import numpy as np
import yfinance as yf
import talib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import roc_auc_score, classification_report
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")

# Directory for caching historical data
DATA_DIR = "data"

def fetch_data(ticker, start='2000-01-01', end='2025-12-31'):
    """
    Fetch historical OHLCV data for a given ticker using caching.
    If cached data exists, load it from disk; otherwise, download and save.
    """
    filename = os.path.join(DATA_DIR, f"{ticker}_{start}_{end}.csv")
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename, index_col=0, parse_dates=True)
            if not df.empty:
                return df
        except Exception as e:
            print(f"Error loading cached data for {ticker}: {e}")
    # If no cached data, fetch from yfinance
    df = yf.download(ticker, start=start, end=end)
    if df.empty:
        print(f"Warning: No data for {ticker}.")
        return df
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(filename)
    return df

def create_features(df):
    """
    Compute technical and price-action features.
    Includes:
      - RSI, MACD, ATR, Bollinger Bands
      - 20-day high and percentage difference from it
      - 50-day moving average and percentage difference
      - Future 5-day return and binary target (1 if >3% gain)
    """
    df = df.copy()
    
    # Technical indicators
    df['rsi'] = talib.RSI(df['Close'], timeperiod=14)
    macd, macdsignal, macdhist = talib.MACD(df['Close'])
    df['macd'] = macd
    df['atr'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    bb_upper, bb_middle, bb_lower = talib.BBANDS(df['Close'], timeperiod=20)
    df['bb_upper'] = bb_upper
    df['bb_middle'] = bb_middle
    df['bb_lower'] = bb_lower

    # Price-action features
    df['20d_high'] = df['High'].rolling(window=20).max()
    df['pct_from_20d_high'] = (df['Close'] - df['20d_high']) / df['20d_high']
    df['50d_ma'] = df['Close'].rolling(window=50).mean()
    df['price_to_ma50'] = df['Close'] / df['50d_ma'] - 1

    # Define target: 1 if future 5-day return >3%, else 0
    df['future_return_5d'] = df['Close'].shift(-5) / df['Close'] - 1
    df['target'] = (df['future_return_5d'] > 0.03).astype(int)
    
    df.dropna(inplace=True)
    return df

def process_ticker(ticker, start, end):
    """
    Helper function for parallel processing.
    Downloads data, computes features, and tags with the ticker.
    """
    df = fetch_data(ticker, start, end)
    if df is not None and not df.empty:
        df = create_features(df)
        df['ticker'] = ticker
        return df
    return None

def build_dataset(tickers, start='2000-01-01', end='2025-12-31'):
    """
    Build a combined dataset for all tickers using parallel processing.
    """
    data_list = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {executor.submit(process_ticker, ticker, start, end): ticker for ticker in tickers}
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                result = future.result()
                if result is not None and not result.empty:
                    data_list.append(result)
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
    if data_list:
        data = pd.concat(data_list, axis=0)
        data.dropna(inplace=True)
        return data
    else:
        return pd.DataFrame()

def train_model(data, feature_cols):
    """
    Train an XGBoost classifier using TimeSeriesSplit and GridSearchCV.
    Scales the features and returns the scaler along with the best model.
    """
    X = data[feature_cols]
    y = data['target']
    
    # Standardize features for better performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Time-series aware cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1],
        'colsample_bytree': [0.8, 1],
        'reg_alpha': [0, 0.01, 0.1],
        'reg_lambda': [1, 1.5, 2]
    }
    xgb_clf = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss')
    grid_search = GridSearchCV(xgb_clf, param_grid, cv=tscv, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid_search.fit(X_scaled, y)
    print("Best parameters:", grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # Evaluate model performance on the training set
    y_pred_prob = best_model.predict_proba(X_scaled)[:, 1]
    auc = roc_auc_score(y, y_pred_prob)
    print("Overall AUC:", auc)
    print("Classification Report:")
    print(classification_report(y, best_model.predict(X_scaled)))
    
    return scaler, best_model

def score_ticker(ticker, model, scaler, feature_cols, start='2024-01-01', end='2025-12-31'):
    """
    For a given ticker, fetch recent data, compute features,
    scale them, and return the model's probability prediction.
    """
    df = fetch_data(ticker, start, end)
    if df is None or df.empty:
        return None
    df = create_features(df)
    latest_features = df.iloc[-1][feature_cols].values.reshape(1, -1)
    latest_features_scaled = scaler.transform(latest_features)
    prob = model.predict_proba(latest_features_scaled)[0, 1]
    return prob

if __name__ == '__main__':
    # Load S&P 500 tickers from your CSV file (assumes a column named "Ticker")
    tickers_df = pd.read_csv("sp500_tickers.csv")
    tickers = tickers_df["Ticker"].tolist()
    
    # Build the historical dataset using parallel processing and caching
    print("Building dataset for training...")
    data = build_dataset(tickers, start='2000-01-01', end='2025-12-31')
    if data.empty:
        print("No data available. Exiting.")
        exit()

    # Define the features to be used by the model
    feature_cols = ['rsi', 'macd', 'atr', 'bb_upper', 'bb_middle', 'bb_lower',
                    'pct_from_20d_high', 'price_to_ma50']

    print("Training model with hyperparameter tuning...")
    scaler, best_model = train_model(data, feature_cols)

    # Live scoring: Use recent data (e.g., from 2024 onward) to score each ticker
    live_scores = {}
    for ticker in tickers:
        prob = score_ticker(ticker, best_model, scaler, feature_cols, start='2024-01-01', end='2025-12-31')
        if prob is not None:
            live_scores[ticker] = prob

    # Get the top 5 tickers based on the predicted probability
    top5 = sorted(live_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\nTop 5 Picks for This Week:")
    for ticker, score in top5:
        print(f"{ticker}: Probability of >3% return in 5 days = {score:.2%}")
