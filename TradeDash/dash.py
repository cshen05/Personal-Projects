import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import resample

# --- Indicator Functions ---
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_macd(series):
    exp1 = series.ewm(span=12, adjust=False).mean()
    exp2 = series.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd - signal

def compute_obv(df):
    obv = [0]
    for i in range(1, len(df)):
        if df['Close'][i] > df['Close'][i - 1]:
            obv.append(obv[-1] + df['Volume'][i])
        elif df['Close'][i] < df['Close'][i - 1]:
            obv.append(obv[-1] - df['Volume'][i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=df.index)

def compute_bollinger_bands(series, period=20):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return upper, lower

def compute_market_forecast_proxy(df):
    k = ((df['Close'] - df['Low'].rolling(14).min()) /
         (df['High'].rolling(14).max() - df['Low'].rolling(14).min())) * 100
    d = k.rolling(3).mean()
    ema_slope = df['Close'].ewm(span=13).mean().diff()
    return 0.4 * compute_rsi(df['Close']) + 0.3 * d + 0.3 * ema_slope

def rule_based_recommendation_series(df):
    signals = []
    for _, row in df.iterrows():
        rsi = row['RSI']
        macd = row['MACD']
        close = row['Close']
        bb_upper = row['BB_Upper']
        bb_lower = row['BB_Lower']

        if rsi < 30 and macd > 0:
            signals.append("BUY")
        elif rsi > 70 and macd < 0:
            signals.append("SELL")
        elif close > bb_upper:
            signals.append("SELL")
        elif close < bb_lower:
            signals.append("BUY")
        else:
            signals.append("HOLD")
    return signals

# Streamlit App
st.set_page_config(layout="wide")
st.title("Technical Dashboard")

ticker_input = st.text_input("Enter stock symbols (comma-separated):", "AAPL, MSFT")
stocks = [s.strip().upper() for s in ticker_input.split(",") if s.strip()]

if st.button("Analyze") and stocks:
    for symbol in stocks:
        st.subheader(f"{symbol} Technical Indicators and ML Prediction")
        df = yf.download(symbol, start="2022-01-01")

        if df.empty:
            st.warning(f"No data for {symbol}")
            continue

        df['RSI'] = compute_rsi(df['Close'])
        df['MACD'] = compute_macd(df['Close'])
        df['OBV'] = compute_obv(df)
        upper, lower = compute_bollinger_bands(df['Close'])
        df['BB_Upper'] = upper
        df['BB_Lower'] = lower
        df['MF_Proxy'] = compute_market_forecast_proxy(df)

        # Label creation for ML
        df['Future_Return'] = df['Close'].shift(-5) / df['Close'] - 1
        df['Label'] = df['Future_Return'].apply(lambda x: 'BUY' if x > 0.03 else ('SELL' if x < -0.03 else 'HOLD'))
        df.dropna(inplace=True)

        features = ['RSI', 'MACD', 'OBV', 'BB_Upper', 'BB_Lower', 'MF_Proxy']

        # Class balancing
        df_buy = df[df['Label'] == 'BUY']
        df_sell = df[df['Label'] == 'SELL']
        df_hold = df[df['Label'] == 'HOLD']
        df_buy = resample(df_buy, replace=True, n_samples=len(df_hold), random_state=42)
        df_sell = resample(df_sell, replace=True, n_samples=len(df_hold), random_state=42)
        df_balanced = pd.concat([df_buy, df_sell, df_hold]).sample(frac=1, random_state=42)

        X = df_balanced[features]
        y = df_balanced['Label']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.text("Classification Report")
        st.text(classification_report(y_test, y_pred))

        df_test = df.iloc[-len(y_pred):].copy()
        df_test['ML_Signal'] = y_pred
        df_test['Rule_Signal'] = rule_based_recommendation_series(df_test)

        # --- Chart Plot ---
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close'))
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'))
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD'))
        fig.add_trace(go.Scatter(x=df.index, y=df['OBV'], name='OBV'))
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='Bollinger Upper'))
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='Bollinger Lower'))
        fig.add_trace(go.Scatter(x=df.index, y=df['MF_Proxy'], name='Market Forecast Proxy'))
        fig.update_layout(title=f"{symbol} Indicators", xaxis_title="Date", yaxis_title="Value",
                          autosize=True, height=600)
        st.plotly_chart(fig, use_container_width=True)

        # --- Backtesting ---
        df_test['Market_Return'] = df_test['Close'].pct_change().shift(-1)
        df_test['Strategy_Return'] = np.where(df_test['ML_Signal'] == 'BUY', df_test['Market_Return'], 0)
        df_test['Cumulative_Market'] = (1 + df_test['Market_Return'].fillna(0)).cumprod()
        df_test['Cumulative_Strategy'] = (1 + df_test['Strategy_Return'].fillna(0)).cumprod()

        backtest_fig = go.Figure()
        backtest_fig.add_trace(go.Scatter(x=df_test.index, y=df_test['Cumulative_Market'],
                                          name='Market (Buy & Hold)', line=dict(color='gray')))
        backtest_fig.add_trace(go.Scatter(x=df_test.index, y=df_test['Cumulative_Strategy'],
                                          name='ML Strategy', line=dict(color='blue')))
        backtest_fig.update_layout(title="📈 Backtest: Cumulative Returns",
                                   xaxis_title="Date", yaxis_title="Cumulative Return",
                                   autosize=True, height=400)
        st.plotly_chart(backtest_fig, use_container_width=True)

        # --- Sharpe Ratio ---
        strategy_returns = df_test['Strategy_Return'].dropna()
        if not strategy_returns.empty:
            mean_return = strategy_returns.mean()
            std_return = strategy_returns.std()
            sharpe_ratio = (mean_return / std_return) * np.sqrt(252)
            st.metric(label="📊 Sharpe Ratio (ML Strategy)", value=f"{sharpe_ratio:.2f}")
        else:
            st.info("Not enough ML trades to compute Sharpe Ratio.")

        # --- Recommendations ---
        st.subheader("Latest Recommendations")
        st.markdown(f"**ML Recommendation:** `{df_test['ML_Signal'].iloc[-1]}`")
        st.markdown(f"**Rule-Based Recommendation:** `{df_test['Rule_Signal'].iloc[-1]}`")

        # --- Accuracy Comparison ---
        ml_acc = (df_test['ML_Signal'] == df_test['Label']).mean()
        rule_acc = (df_test['Rule_Signal'] == df_test['Label']).mean()

        st.subheader("📊 Accuracy Comparison")
        st.dataframe(pd.DataFrame({
            "Method": ["ML Model", "Rule-Based"],
            "Accuracy": [f"{ml_acc:.2%}", f"{rule_acc:.2%}"]
        }))
