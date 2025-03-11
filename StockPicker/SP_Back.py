import sys
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import logging
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

# ---------------------------
# Logging and Alert Configuration
# ---------------------------
logging.basicConfig(stream=sys.stdout, 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def send_alert(message):
    logging.warning("ALERT: " + message)

def compute_RSI(series, period=14):
    """
    Compute the Relative Strength Index (RSI) for a pandas Series.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def filter_tickers(tickers, 
                    min_avg_volume=500000, 
                    min_price=5, 
                    sma_short_period=10, 
                    sma_long_period=50,
                    min_momentum=0,
                    rsi_lower=30, 
                    rsi_upper=70, 
                    max_volatility=0.05,
                    target=100):
    """
    Filters a list of tickers based on liquidity, price, trend, momentum, RSI, and volatility.
    
    Parameters:
        tickers (list): List of ticker symbols.
        min_avg_volume (int): Minimum average daily volume.
        min_price (float): Minimum current stock price.
        sma_short_period (int): Period for the short-term moving average.
        sma_long_period (int): Period for the long-term moving average.
        min_momentum (float): Minimum momentum (e.g., difference between current close and close 10 days ago).
        rsi_lower (float): Lower bound for acceptable RSI.
        rsi_upper (float): Upper bound for acceptable RSI.
        max_volatility (float): Maximum allowed volatility (as a decimal).
        target (int): Maximum number of tickers to return.
    
    Returns:
        list: Filtered list of ticker symbols.
    """
    filtered = []
    for ticker in tickers:
        try:
            # Download 6 months of data for screening
            data = yf.download(ticker, period="6mo")
            if data.empty:
                continue

            # Liquidity: average volume must be above threshold
            avg_volume = data['Volume'].mean()
            if avg_volume < min_avg_volume:
                continue

            # Price: current closing price must be above threshold
            current_price = data['Close'].iloc[-1]
            if current_price < min_price:
                continue

            # Trend: short-term SMA > long-term SMA
            data['SMA_short'] = data['Close'].rolling(window=sma_short_period).mean()
            data['SMA_long'] = data['Close'].rolling(window=sma_long_period).mean()
            if data['SMA_short'].iloc[-1] <= data['SMA_long'].iloc[-1]:
                continue

            # Momentum: difference between current close and close 10 days ago
            if len(data) < 11:
                continue  # not enough data
            momentum = data['Close'].iloc[-1] - data['Close'].iloc[-11]
            if momentum < min_momentum:
                continue

            # RSI: must be between rsi_lower and rsi_upper
            data['RSI'] = compute_RSI(data['Close'], period=14)
            current_rsi = data['RSI'].iloc[-1]
            if not (rsi_lower <= current_rsi <= rsi_upper):
                continue

            # Volatility: 14-day rolling std dev of returns must be below max_volatility
            data['Returns'] = data['Close'].pct_change()
            volatility = data['Returns'].rolling(window=14).std().iloc[-1]
            if volatility > max_volatility:
                continue

            # If all criteria are met, save ticker with key metrics (optional: for sorting)
            filtered.append((ticker, momentum, current_price, avg_volume, current_rsi, volatility))
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue

    # Sort by momentum (or choose another metric) in descending order and return top 'target' tickers.
    filtered.sort(key=lambda x: x[1], reverse=True)
    return [ticker for ticker, *_ in filtered[:target]]

# ---------------------------
# TradingSystem Class Definition
# ---------------------------
class TradingSystem:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.daily_data = {}   # Daily data per ticker
        self.weekly_data = {}  # Weekly data per ticker
        self.dataset = pd.DataFrame()
        self.models = {}
        self.best_model = None
        self.features = None
        self.performance_data = []  # Store performance metrics for feedback

    # ---- Data Downloading & Quality Checks ----
    def download_data(self):
        logging.info("Downloading data and ensuring quality...")
        for ticker in self.tickers:
            try:
                logging.info(f"Downloading daily data for {ticker}")
                df = yf.download(ticker, start=self.start_date, end=self.end_date)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)                
            except Exception as e:
                send_alert(f"Error downloading data for {ticker}: {str(e)}")
                continue
            if df.empty:
                send_alert(f"No data returned for {ticker}")
                continue
            try:
                df.dropna(inplace=True)
                # Apply a liquidity filter: remove days with very low volume (below 5th percentile)
                vol_threshold = df['Volume'].quantile(0.05)
                df = df[df['Volume'] > vol_threshold]
            except Exception as e:
                send_alert(f"Error processing data for {ticker}: {str(e)}")
                continue
            self.daily_data[ticker] = df

            # Download weekly data (resample from daily)
            try:
                weekly = df.resample('W').agg({'Open': 'first',
                                                'High': 'max',
                                                'Low': 'min',
                                                'Close': 'last',
                                                'Volume': 'sum'})
                weekly.dropna(inplace=True)
                if isinstance(weekly.columns, pd.MultiIndex):
                    weekly.columns = weekly.columns.get_level_values(0)               
                self.weekly_data[ticker] = weekly
            except Exception as e:
                send_alert(f"Error processing weekly data for {ticker}: {str(e)}")
                continue

    # ---- Feature Engineering with Multiple Time Frames and Momentum Indicator ----
    def compute_technical_indicators(self, df, timeframe='daily'):
        try:
            # Momentum indicator: RSI (14-day)
            delta = df['Close'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Moving Averages: 5-day and 20-day SMA
            df['SMA_5'] = df['Close'].rolling(window=5).mean()
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            
            # Momentum: Price change over the past 10 days
            df['Momentum'] = df['Close'] - df['Close'].shift(10)
            
            # Volatility: 5-day rolling standard deviation of returns
            df['Return'] = df['Close'].pct_change()
            df['Volatility'] = df['Return'].rolling(window=5).std()
            
            df.dropna(inplace=True)
        except Exception as e:
            send_alert(f"Error computing technical indicators: {str(e)}")
        return df

    def build_dataset(self):
        logging.info("Building dataset with daily and weekly features...")
        data_list = []
        for ticker in self.tickers:
            if ticker not in self.daily_data:
                continue
            try:
                # Compute indicators on daily data
                df_daily = self.compute_technical_indicators(self.daily_data[ticker].copy(), timeframe='daily')
                
                # Compute indicators on weekly data and forward-fill into daily data
                if ticker in self.weekly_data:
                    df_weekly = self.compute_technical_indicators(self.weekly_data[ticker].copy(), timeframe='weekly')
                    # Select a subset of weekly features and forward-fill into daily data
                    df_weekly = df_weekly[['RSI', 'SMA_20', 'Momentum']]
                    df_weekly = df_weekly.reindex(df_daily.index, method='ffill')
                    df_daily['RSI_weekly'] = df_weekly['RSI']
                    df_daily['SMA_20_weekly'] = df_weekly['SMA_20']
                    df_daily['Momentum_weekly'] = df_weekly['Momentum']
                df_daily['Ticker'] = ticker
                # Target: next day return
                df_daily['Target'] = df_daily['Return'].shift(-1)
                df_daily.dropna(inplace=True)
                # Save Date for later use
                df_daily= df_daily.reset_index()
                data_list.append(df_daily)
            except Exception as e:
                send_alert(f"Error building dataset for {ticker}: {str(e)}")
                continue
        try:
            if data_list:
                self.dataset = pd.concat(data_list)
                self.dataset.sort_values(by='Date', inplace=True)
            else:
                send_alert("No dataset could be built from the available data.")
        except Exception as e:
            send_alert(f"Error concatenating dataset: {str(e)}")
        return self.dataset

    # ---- Model Selection, Hyperparameter Tuning, and Training ----
    def model_selection_and_training(self):
        logging.info("Starting model selection and hyperparameter tuning...")
        try:
            feature_cols = ['RSI', 'SMA_5', 'SMA_20', 'Momentum', 'Volatility',
                            'RSI_weekly', 'SMA_20_weekly', 'Momentum_weekly']
            self.features = feature_cols

            X = self.dataset[feature_cols]
            y = self.dataset['Target']

            # Time series cross-validation to reduce overfitting risk
            tscv = TimeSeriesSplit(n_splits=5)

            candidates = {
                'RandomForest': {
                    'model': RandomForestRegressor(random_state=42),
                    'params': {'n_estimators': [50, 100], 'max_depth': [3, 5, None]}
                },
                'GradientBoosting': {
                    'model': GradientBoostingRegressor(random_state=42),
                    'params': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}
                },
                'LinearRegression': {
                    'model': LinearRegression(),
                    'params': {}
                }
            }
            best_score = float('inf')
            best_model_name = None
            best_model = None

            for name, candidate in candidates.items():
                logging.info(f"Tuning model: {name}")
                grid = GridSearchCV(candidate['model'], candidate['params'], cv=tscv,
                                    scoring='neg_mean_squared_error')
                grid.fit(X, y)
                score = -grid.best_score_
                logging.info(f"{name} best MSE: {score:.6f}")
                self.models[name] = grid.best_estimator_
                if score < best_score:
                    best_score = score
                    best_model = grid.best_estimator_
                    best_model_name = name

            logging.info(f"Selected best model: {best_model_name} with MSE: {best_score:.6f}")
            self.best_model = best_model
        except Exception as e:
            send_alert(f"Error during model training: {str(e)}")

    # ---- Dynamic Position Sizing and Stop Loss Orders ----
    def dynamic_position_sizing(self, predicted_return, volatility):
        try:
            target_risk = 0.01  # Example: risk 1% of portfolio per trade
            if volatility > 0:
                position_size = min(target_risk / volatility, 1.0)
            else:
                position_size = 0
        except Exception as e:
            send_alert(f"Error in dynamic position sizing: {str(e)}")
            position_size = 0
        return position_size

    # ---- Diversification: Ensure Basket is Not Overly Correlated ----
    def diversify_basket(self, candidate_signals):
        try:
            if len(candidate_signals) < 2:
                return candidate_signals
            tickers = list(candidate_signals.keys())
            recent_returns = []
            for ticker in tickers:
                df = self.daily_data.get(ticker)
                if df is not None:
                    recent_returns.append(df['Return'].tail(30).values)
            returns_matrix = np.array(recent_returns)
            corr_matrix = np.corrcoef(returns_matrix)
            selected = {}
            removed = set()
            for i, ticker in enumerate(tickers):
                if ticker in removed:
                    continue
                selected[ticker] = candidate_signals[ticker]
                for j in range(i + 1, len(tickers)):
                    if tickers[j] in removed:
                        continue
                    if corr_matrix[i, j] > 0.9:
                        removed.add(tickers[j])
            return selected
        except Exception as e:
            send_alert(f"Error during diversification: {str(e)}")
            return candidate_signals

    # ---- Print Today's Basket ----
    def print_basket(self, basket_size=5, portfolio_value=100):
        """
        For the most recent date in our dataset, generate signals for each ticker.
        Then select the top basket_size stocks with positive predicted returns.
        Print out for each:
            - Ticker
            - Predicted Return (as a percentage)
            - Current Price
            - Allocated Cost (given total portfolio_value)
            - Number of Shares (allocated_cost / current price)
            - Expected Dollar Return (allocated_cost * predicted_return)
            - Position Sizing (as a percentage)
        Also print the total portfolio cost allocated.
        
        A stock is only included if its expected dollar return is greater than its current price.
        """
        signals = {}
        for ticker in self.tickers:
            try:
                df_ticker = self.dataset[self.dataset['Ticker'] == ticker]
                if df_ticker.empty:
                    continue
                latest_row = df_ticker.iloc[-1]
                feat_vals = latest_row[self.features].values.reshape(1, -1)
                predicted_return = self.best_model.predict(feat_vals)[0]
                if predicted_return <= 0:
                    continue
                signals[ticker] = {
                    'predicted_return': predicted_return,
                    'volatility': latest_row['Volatility'],
                    'price': latest_row['Close']
                }
            except Exception as e:
                send_alert(f"Error generating basket signal for {ticker}: {str(e)}")
                continue

        sorted_signals = sorted(signals.items(), key=lambda x: x[1]['predicted_return'], reverse=True)
        selected = sorted_signals[:basket_size]
        if not selected:
            print("No stocks with positive predicted return today.")
            return

        # Allocate weights inversely proportional to volatility
        inv_vol = np.array([1/sig['volatility'] if sig['volatility'] > 0 else 0 for _, sig in selected])
        if inv_vol.sum() > 0:
            weights = inv_vol / inv_vol.sum()
        else:
            weights = np.ones(len(inv_vol)) / len(inv_vol)

        print("Today's Basket:")
        total_cost = 0
        basket_details = []
        for (ticker, sig), weight in zip(selected, weights):
            allocated_cost = portfolio_value * weight
            num_shares = allocated_cost / sig['price']
            expected_dollar_return = allocated_cost * sig['predicted_return']
            # Only include if expected dollar return exceeds the stock's current price
            if expected_dollar_return < sig['price']:
                continue
            basket_details.append({
                'Ticker': ticker,
                'Predicted Return (%)': round(sig['predicted_return'] * 100, 2),
                'Current Price': round(sig['price'], 2),
                'Weight': round(weight, 4),
                'Position Sizing (%)': round(weight * 100, 2),
                'Allocated Cost': round(allocated_cost, 2),
                'Number of Shares': int(num_shares),
                'Expected Dollar Return': round(expected_dollar_return, 2)
            })
            total_cost += allocated_cost

        basket_df = pd.DataFrame(basket_details)
        print(basket_df)
        print(f"\nTotal Portfolio Cost Allocated: {round(total_cost, 2)}")

    # ---- Robust Backtesting ----
    def robust_backtest(self):
        logging.info("Starting robust backtest over out-of-sample period...")
        try:
            all_dates = np.sort(self.dataset['Date'].unique())
            split_idx = int(len(all_dates) * 0.8)
            test_dates = all_dates[split_idx:]
        except Exception as e:
            send_alert(f"Error setting up backtest dates: {str(e)}")
            return None, None, None

        portfolio_value = 100  # starting with $100
        portfolio_history = []
        trade_log = []

        for current_date in test_dates:
            daily_signals = {}
            for ticker in self.tickers:
                try:
                    if ticker not in self.daily_data:
                        continue
                    df = self.daily_data[ticker]
                    if current_date not in df.index:
                        continue
                    row = df.loc[current_date]
                    feat_vals = row[self.features].values.reshape(1, -1)
                    predicted_return = self.best_model.predict(feat_vals)[0]
                    if predicted_return <= 0:
                        continue
                    daily_signals[ticker] = {
                        'predicted_return': predicted_return,
                        'volatility': row['Volatility'],
                        'price': row['Close'],
                        'stop_loss': row['Close'] * 0.95  # e.g., 5% stop loss
                    }
                except Exception as e:
                    send_alert(f"Error generating signal for {ticker} on {current_date}: {str(e)}")
                    continue

            daily_signals = self.diversify_basket(daily_signals)
            if not daily_signals:
                portfolio_history.append((current_date, portfolio_value))
                continue

            positions = {}
            for ticker, signal in daily_signals.items():
                try:
                    size = self.dynamic_position_sizing(signal['predicted_return'], signal['volatility'])
                    positions[ticker] = size
                except Exception as e:
                    send_alert(f"Error calculating position for {ticker}: {str(e)}")
                    continue

            daily_return = 0
            for ticker, size in positions.items():
                try:
                    df = self.daily_data[ticker]
                    idx = df.index.get_loc(current_date)
                    if idx + 1 >= len(df):
                        continue
                    next_date = df.index[idx + 1]
                    next_row = df.iloc[idx + 1]
                    stop_loss = daily_signals[ticker]['stop_loss']
                    if next_row['Low'] < stop_loss:
                        execution_price = stop_loss
                    else:
                        execution_price = next_row['Close']
                    trade_return = (execution_price - daily_signals[ticker]['price']) / daily_signals[ticker]['price']
                    daily_return += size * trade_return
                    trade_log.append({
                        'date': next_date,
                        'ticker': ticker,
                        'entry_price': daily_signals[ticker]['price'],
                        'exit_price': execution_price,
                        'position_size': size,
                        'trade_return': trade_return
                    })
                except Exception as e:
                    send_alert(f"Error executing trade for {ticker} on {current_date}: {str(e)}")
                    continue

            portfolio_value *= (1 + daily_return)
            portfolio_history.append((current_date, portfolio_value))

        try:
            portfolio_df = pd.DataFrame(portfolio_history, columns=['Date', 'PortfolioValue'])
            portfolio_df.set_index('Date', inplace=True)
            daily_returns = portfolio_df['PortfolioValue'].pct_change().dropna()
            sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
            plt.figure(figsize=(10, 6))
            plt.plot(portfolio_df.index, portfolio_df['PortfolioValue'], label='Equity Curve')
            plt.xlabel("Date")
            plt.ylabel("Portfolio Value ($)")
            plt.title("Robust Backtest Equity Curve")
            plt.legend()
            plt.show()
            logging.info(f"Robust Backtest Sharpe Ratio: {sharpe:.2f}")
        except Exception as e:
            send_alert(f"Error finalizing backtest results: {str(e)}")
            return None, trade_log, None

        return portfolio_df, trade_log, sharpe

    # ---- A/B Testing: ML-Based Strategy vs. Baseline ----
    def ab_testing(self):
        logging.info("Conducting A/B testing between ML-based strategy and a baseline momentum strategy...")
        ml_portfolio = 100  # starting with $100
        baseline_portfolio = 100
        ml_history = []
        baseline_history = []
        try:
            all_dates = np.sort(self.dataset['Date'].unique())
            split_idx = int(len(all_dates) * 0.8)
            test_dates = all_dates[split_idx:]
        except Exception as e:
            send_alert(f"Error setting up A/B test dates: {str(e)}")
            return

        for current_date in test_dates:
            ml_return = 0
            baseline_return = 0
            for ticker in self.tickers:
                try:
                    if ticker not in self.daily_data:
                        continue
                    df = self.daily_data[ticker]
                    if current_date not in df.index:
                        continue
                    row = df.loc[current_date]
                    feat_vals = row[self.features].values.reshape(1, -1)
                    ml_pred = self.best_model.predict(feat_vals)[0]
                    baseline_signal = row['Momentum'] > 0
                    idx = df.index.get_loc(current_date)
                    if idx + 1 < len(df):
                        next_row = df.iloc[idx + 1]
                        trade_ret = (next_row['Close'] - row['Close']) / row['Close']
                        if ml_pred > 0:
                            size = self.dynamic_position_sizing(ml_pred, row['Volatility'])
                            ml_return += size * trade_ret
                        if baseline_signal:
                            baseline_return += 0.1 * trade_ret  # fixed weight for baseline
                except Exception as e:
                    send_alert(f"Error in A/B test for {ticker} on {current_date}: {str(e)}")
                    continue
            ml_portfolio *= (1 + ml_return)
            baseline_portfolio *= (1 + baseline_return)
            ml_history.append((current_date, ml_portfolio))
            baseline_history.append((current_date, baseline_portfolio))
        try:
            ml_df = pd.DataFrame(ml_history, columns=['Date', 'ML_Portfolio'])
            baseline_df = pd.DataFrame(baseline_history, columns=['Date', 'Baseline_Portfolio'])
            ml_df.set_index('Date', inplace=True)
            baseline_df.set_index('Date', inplace=True)
            plt.figure(figsize=(10, 6))
            plt.plot(ml_df.index, ml_df['ML_Portfolio'], label="ML Strategy")
            plt.plot(baseline_df.index, baseline_df['Baseline_Portfolio'], label="Baseline Strategy")
            plt.xlabel("Date")
            plt.ylabel("Portfolio Value ($)")
            plt.title("A/B Testing: ML vs. Baseline Strategy")
            plt.legend()
            plt.show()
        except Exception as e:
            send_alert(f"Error finalizing A/B test results: {str(e)}")

# ---------------------------
# Main Execution
# ---------------------------
def main():
    try:
        #tickers = pd.read_csv('nyse-listed.csv')['ACT Symbol'].tolist()
        tickers = pd.read_csv('test.csv')['ACT Symbol'].tolist()
    except:
        send_alert(f"Error loading ticker list: {str(e)}")
    
    filtered_tickers = filter_tickers(tickers, target=100)
    if not filtered_tickers:
        send_alert("No tickers passed the filtering criteria")
    else:
        logging.info(f"Filtered to {len(filtered_tickers)} tickers for model training")
    
    start_date = '2020-01-01'
    end_date = datetime.datetime.today().strftime('%Y-%m-%d')
    
    trading_system = TradingSystem(tickers, start_date, end_date)
    
    try:
        trading_system.download_data()
    except Exception as e:
        send_alert(f"Critical error during data download: {str(e)}")
    
    try:
        trading_system.build_dataset()
    except Exception as e:
        send_alert(f"Critical error during dataset build: {str(e)}")
    
    try:
        trading_system.model_selection_and_training()
    except Exception as e:
        send_alert(f"Critical error during model training: {str(e)}")
    
    # Print out the basket of stocks for today
    try:
        trading_system.print_basket(basket_size=5, portfolio_value=100)
    except Exception as e:
        send_alert(f"Critical error while printing basket: {str(e)}")
    
    # Run robust backtest
    try:
        portfolio_df, trade_log, sharpe = trading_system.robust_backtest()
    except Exception as e:
        send_alert(f"Critical error during robust backtest: {str(e)}")
    
    # Run A/B testing
    try:
        trading_system.ab_testing()
    except Exception as e:
        send_alert(f"Critical error during A/B testing: {str(e)}")
    
if __name__ == '__main__':
    main()
