import yfinance as yf
import pandas as pd
import heapq
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.over_sampling import SMOTE

# Fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch historical stock data using yfinance.

    Args:
        ticker: Stock ticker symbol.
        start_date: Start date for fetching data.
        end_date: End date for fetching data.

    Returns:
        DataFrame with stock data.
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    return data

# Add features
def add_features(data):
    """
    Add technical indicators to stock data.

    Args:
        data: DataFrame with stock data.

    Returns:
        DataFrame with added features and target variable.
    """
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
    """
    Rank features by their importance.

    Args:
        model: Trained Random Forest model.
        feature_names: List of feature names.

    Returns:
        List of ranked features with their importance scores.
    """
    feature_importance = model.feature_importances_
    feature_tuples = [(importance, feature) for importance, feature in zip(feature_importance, feature_names)]
    ranked_features = heapq.nlargest(len(feature_tuples), feature_tuples)
    return ranked_features

# Train and evaluate Random Forest
def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    """
    Train and evaluate a Random Forest model with regularization and cross-validation.

    Args:
        X_train: Training dataset features.
        y_train: Training dataset targets.
        X_test: Test dataset features.
        y_test: Test dataset targets.

    Returns:
        Trained Random Forest model and evaluation metrics.
    """
    # Initialize Random Forest with regularization
    model = RandomForestClassifier(
        n_estimators=100,         # Reduce number of trees
        max_depth=10,             # Limit tree depth
        min_samples_split=5,      # Require at least 5 samples to split
        min_samples_leaf=2,       # Require at least 2 samples per leaf
        random_state=42
    )

    # Oversample using SMOTE
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Train the model
    model.fit(X_train, y_train)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-Validation Accuracy: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")

    # Training and Test Accuracy
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)

    print(f"Training Accuracy: {train_accuracy:.2%}")
    print(f"Test Accuracy: {test_accuracy:.2%}")

    return model, train_accuracy, test_accuracy, cv_scores.mean(), cv_scores.std()

# Provide recommendation
def provide_insight(model, X_test):
    """
    Generate a recommendation based on the model's predictions and evaluate its performance.

    Args:
        model: Trained Random Forest model.
        X_train: Training dataset features.
        y_train: Training dataset targets.
        X_test: Test dataset features.
        y_test: Test dataset targets.

    Returns:
        A tuple (recommendation, confidence) where:
        - recommendation: "BUY" or "SELL".
        - confidence: Confidence of the prediction (percentage).
    """
    # Predictions for test data
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

    return (recommendation, last_prediction_confidence,)