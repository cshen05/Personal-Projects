import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib as plt
import heapq
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    make_scorer,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
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


def check_class_distribution(y):
    """
    Check and print the class distribution.
    """
    print("Class distribution:")
    print(y.value_counts())


def add_features(data):
    """
    Add technical indicators and enhanced features to stock data.
    """
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    data['Close'].fillna(method='ffill', inplace=True)
    data.dropna(subset=['Close'], inplace=True)

    # Calculate features
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    
    data['EMA10'] = data['Close'].ewm(span=10, adjust=False).mean()
    data['EMA50'] = data['Close'].ewm(span=50, adjust=False).mean()
    
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    rolling_std = data['Close'].rolling(window=10).std()
    data['BB_upper'] = data['MA10'] + 2 * rolling_std
    data['BB_lower'] = data['MA10'] - 2 * rolling_std
    
    data['TR'] = data[['High', 'Low', 'Close']].apply(
        lambda row: max(row['High'] - row['Low'], abs(row['High'] - row['Close']), abs(row['Low'] - row['Close'])),
        axis=1
    )
    
    data['ATR'] = data['TR'].rolling(window=14).mean()
    
    data['Stochastic'] = (
        (data['Close'] - data['Low'].rolling(14).min()) /
        (data['High'].rolling(14).max() - data['Low'].rolling(14).min())
    ) * 100
    
    data['OBV'] = (data['Volume'] * (data['Close'] - data['Close'].shift(1)).apply(lambda x: 1 if x > 0 else -1)).cumsum()
    
    data['DM_plus'] = (data['High'] - data['High'].shift(1)).apply(lambda x: x if x > 0 else 0)
    data['DM_minus'] = (data['Low'].shift(1) - data['Low']).apply(lambda x: x if x > 0 else 0)
    
    data['ADX'] = (
        (data['DM_plus'] - data['DM_minus']).abs().rolling(window=14).mean() / data['TR'].rolling(window=14).mean()
    )
    
    data['Log_Close'] = np.log1p(data['Close'])
    
    data['RSI_x_OBV'] = data['RSI'] * data['OBV']
    
    data['MA50_minus_MA10'] = data['MA50'] - data['MA10']
    
    data['DayOfWeek'] = data.index.dayofweek
    
    data['Month'] = data.index.month

    # Lagging features
    for lag in range(1, 4):
        data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)
        data[f'Volume_Lag_{lag}'] = data['Volume'].shift(lag)

    # Calculate Percentage Change
    lookahead_period = 7  # Predict 7 days ahead
    data['Pct_Change'] = (data['Close'].shift(-lookahead_period) / data['Close'] - 1)

    # Dynamically Adjust Threshold Multiplier Based on ATR
    multi = 0.5
    avg_atr = data['ATR'].mean()
    if avg_atr >= 2.0:
        multi = 0.75
    elif avg_atr >= 1.0:
        multi = 0.5
    else:
        multi = 0.25

    # Set Threshold with Minimum Value
    min_threshold = 0.003  # Minimum threshold for very low-volatility stocks
    data['Threshold'] = data['ATR'] * multi
    data['Threshold'] = data['Threshold'].apply(lambda x: max(x, min_threshold))

    # Generate Target
    data['Target'] = 0 # Default Hold
    data.loc[data['Pct_Change'] > data['Threshold'], 'Target'] = 1  # Buy
    data.loc[data['Pct_Change'] < -data['Threshold'], 'Target'] = -1 # Sell

    # Debugging: Check Values
    print(data[['ATR', 'Threshold', 'Pct_Change']].tail(10))  
    print(data['Target'].value_counts())

    data.dropna(inplace=True)
    return data


def rank_features(model, feature_names, top_n=10):
    """
    Rank the top features based on their importance using a min-heap.
    """
    feature_importances = model.feature_importances_
    top_features = heapq.nlargest(top_n, zip(feature_importances, feature_names))
    return sorted(top_features, reverse=True)


def custom_scorer(y_true, y_pred):
    """
    Custom F1-score with higher weights for buy (1) and sell (-1).
    """
    weights = {1: 2, 0: 1, -1: 2}  # Assign higher weights to buy and sell
    class_f1 = f1_score(y_true, y_pred, average=None)
    weighted_f1 = sum(weights[i] * class_f1[i] for i in range(len(class_f1))) / sum(weights.values())
    return weighted_f1

def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    """
    Train and evaluate a Random Forest model using provided train-test splits.
    """
    # Handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print(pd.Series(y_train_balanced.value_counts(), name="y_train_balanced Set"))
    
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
        'bootstrap': [True, False]
    }

    # Initialize model
    model = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)

    # Grid search with cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=make_scorer(custom_scorer),
        cv=tscv,
        n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train_balanced)

    best_model = grid_search.best_estimator_
    print("Best Parameters:", grid_search.best_params_)
    important_features = rank_features(best_model, X_train.columns)
    print(important_features)

    # Evaluate the model
    train_predictions = best_model.predict(X_train_scaled)
    test_predictions = best_model.predict(X_test_scaled)

    # Classification Report
    print("Classification Report (Train):")
    print(classification_report(y_train_balanced, train_predictions, target_names=["Sell (-1)", "Hold (0)", "Buy (1)"]))

    print("Classification Report (Test):")
    print(classification_report(y_test, test_predictions, target_names=["Sell (-1)", "Hold (0)", "Buy (1)"]))

    # Confusion Matrix
    cm = confusion_matrix(y_test, test_predictions, labels=[-1, 0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Sell (-1)", "Hold (0)", "Buy (1)"])
    disp.plot(cmap="Blues")
    plt.show()

    train_accuracy = accuracy_score(y_train_balanced, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_probabilities = best_model.predict_proba(X_test_scaled)[:, 1]

    print(f"Training Accuracy: {train_accuracy:.2%}")
    print(f"Test Accuracy: {test_accuracy:.2%}")

    if len(np.unique(y_test)) > 1:
        test_auc = roc_auc_score(y_test, test_probabilities)
        print(f"Test AUC: {test_auc:.2f}")
    else:
        test_auc = None
        print("Test AUC is undefined as only one class is present in y_test.")

    precision = precision_score(y_test, test_predictions, zero_division=0)
    recall = recall_score(y_test, test_predictions, zero_division=0)
    f1 = f1_score(y_test, test_predictions, zero_division=0)

    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1-Score: {f1:.2%}")

    print(pd.Series(y_test.value_counts(), name="y_test Set"))
    return best_model, train_accuracy, test_accuracy, test_auc, precision, recall, f1

def provide_insight(model, X_test):
    """
    Generate recommendation based on the model's prediction.
    """
    predictions = model.predict(X_test)
    print(predictions)
    probabilities = model.predict_proba(X_test)

    last_prediction_confidence = max(probabilities[-1])
    last_prediction = predictions[-1]
    print(last_prediction)

    if last_prediction == 1:
        recommendation = "BUY - The model predicts a rise in stock price."
    elif last_prediction == -1:
        recommendation = "SELL - The model predicts a decline in stock price."
    else:
        recommendation = "HOLD - The model predicts minimal price movement."

    return recommendation, last_prediction_confidence