import yfinance as yf
import pandas as pd
import numpy as np
import heapq
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch the historical stock data using yfinance.
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    return data

def add_features(data):
    """
    Add technical indicators and enhanced features to stock data.
    """
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    data.dropna(subset=['Close'], inplace=True)

    # Moving averages
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()

    # Exponential moving averages
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

    # ATR
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

    # OBV
    data['OBV'] = (data['Volume'] * (data['Close'] - data['Close'].shift(1)).apply(lambda x: 1 if x > 0 else -1)).cumsum()

    # ADX
    data['DM_plus'] = (data['High'] - data['High'].shift(1)).apply(lambda x: x if x > 0 else 0)
    data['DM_minus'] = (data['Low'].shift(1) - data['Low']).apply(lambda x: x if x > 0 else 0)
    data['ADX'] = (
        (data['DM_plus'] - data['DM_minus']).abs().rolling(window=14).mean() / data['TR'].rolling(window=14).mean()
    )

    # New Features
    data['Log_Close'] = np.log1p(data['Close'])
    data['RSI_x_OBV'] = data['RSI'] * data['OBV']
    data['MA50_minus_MA10'] = data['MA50'] - data['MA10']
    data['DayOfWeek'] = data.index.dayofweek
    data['Month'] = data.index.month

    # Lagging features
    for lag in range(1, 4):
        data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)
        data[f'Volume_Lag_{lag}'] = data['Volume'].shift(lag)

    # Target with threshold
    threshold = 0.01  # Meaningful price movement threshold
    data['Target'] = ((data['Close'].shift(-1) / data['Close'] - 1) > threshold).astype(int)

    data.dropna(inplace=True)
    return data

def rank_features(model, feature_names):
    """
    Rank features based on their importance in the trained Random Forest model.
    """
    feature_importances = model.feature_importances_
    ranked_features = sorted(zip(feature_importances, feature_names), reverse=True)
    return ranked_features

def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    """
    Train and evaluate a Random Forest model using provided train-test splits.
    """
    # Handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_balanced),
        columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )


    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5, 7],
        'min_samples_split': [10, 20],
        'min_samples_leaf': [10, 20],
        'max_features': ['sqrt', 0.2],
        'bootstrap': [True]
    }

    # Initialize model
    model = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)

    # Grid search with cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=tscv,
        n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train_balanced)

    best_model = grid_search.best_estimator_
    print("Best Parameters:", grid_search.best_params_)

    # Evaluate the model
    train_predictions = best_model.predict(X_train_scaled)
    test_predictions = best_model.predict(X_test_scaled)

    train_accuracy = accuracy_score(y_train_balanced, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_probabilities = best_model.predict_proba(X_test_scaled)[:, 1]
    test_auc = roc_auc_score(y_test, test_probabilities)

    # Additional metrics
    precision = precision_score(y_test, test_predictions)
    recall = recall_score(y_test, test_predictions)
    f1 = f1_score(y_test, test_predictions)

    print(f"Training Accuracy: {train_accuracy:.2%}")
    print(f"Test Accuracy: {test_accuracy:.2%}")
    print(f"Test AUC: {test_auc:.2f}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1-Score: {f1:.2%}")

    return best_model, train_accuracy, test_accuracy, test_auc, precision, recall, f1

def provide_insight(model, X_test):
    """
    Generate recommendation based on the model's prediction.
    """
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    last_prediction_confidence = max(probabilities[-1])
    last_prediction = predictions[-1]

    if last_prediction == 1:
        recommendation = "BUY - The model predicts a rise in stock price."
    else:
        recommendation = "SELL - The model predicts a decline in stock price."

    return recommendation, last_prediction_confidence