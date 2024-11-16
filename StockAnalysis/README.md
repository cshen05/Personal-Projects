# Stock Analysis Software
- Connor Shen (cs65692)
- Roland Wang (rxw62)

## Overview
The Stock Analysis software is a Python-based program designed to retrieve, analyze, and visualize stock data while providing a BUY/SELL recommendation using a Random Forest, a supervised machine learning model. 

---

## Disclaimer
This software is for informational purposes only, you should not construe such information or other material as financial, or any other advice.

---

## Features
1. Data Retrieval:
	- Fetch historical stock data for any publicly traded company using Yahoo Finance.
2. Technical Analysis:
	- Computes several technical indicators including:
	    - Moving Averages (MA10, MA50)
	    - Exponential Moving Averages (EMA10, EMA50)
	    - Relative Strength Index (RSI)
	    - Bollinger Bands
	    - Average True Range (ATR)
	    - On-Balance Volume (OBV)
	    - Stochastic Oscillator
	    - Average Directional Index (ADX)
3.	Graphical Visualization:
	- Visualizes the stock price trend along with indicators like Bollinger Bands and EMA over the last 5 days.
4.	Feature Importance:
	- Displays ranked importance of technical indicators based on their weighting inside the machine learning model.
    - Contains information on the technical indicators and what they mean
5.	Recommendations:
	- Provides BUY/SELL recommendations with confidence scores, train/test accuracy, and cross-validation accuracy based on predictions from the Random Forest model.

---

## Project Structure

Frontend (StockAnalysis.py)
- The Graphical User Interface (GUI) for interacting with the app
- Built using PyQt5
- Tabs:
	1.	Introduction:
	    - Explains the purpose of the app and provides input fields for stock ticker and date range
	2.	Graph:
	    - Displays stock price trend, Bollinger Bands, and 10 and 50 Day Exponential Moving Averages
	3.	Feature Importance:
	    - Shows how the model weighed the various technical indicators
        - Provides explanations on what the technical indicators are
	4.	Recommendation:
	    - Provides BUY/SELL recommendations with confidence scores, train/test accuracy, and cross-validation accuracy
- Key Functions:
	- fetch_data: Handles how the frontend connects to the backend by fetching stock data, training the machine learning model, and updating the GUI with predictions and visualizations
    - plot_trader_graph: Plots stock trends over the last 5 days

Backend (Engine.py)
- The backend of the application that handles data retrieval, processing, and machine learning
- Key Functions:
	1.	fetch_stock_data(ticker, start_date, end_date):
	    - Fetches the stock data from Yahoo Finance
	2.	add_features(data):
	    - Computes all the technical indicators and adds them as columns to the dataset
	3.	train_and_evaluate_model(X_train, y_train, X_test, y_test):
	    - Trains a Random Forest model using SMOTE for balancing imbalanced classes
	    - Evaluates training, testing, and cross-validation accuracy
	4.	rank_features(model, feature_names, scaler):
	    - Ranks features based on their Random Forest coefficients
        - Used Heapsort to accomplish this
	5.	provide_insight(model, X_test):
	    - Generates the BUY/SELL recommendation and provides confidence levels for predictions
---

## Technical Indicators
The following indicators are calculated and used in the model:
- MA10 & MA50: Moving averages over 10 and 50 days to smooth short-term and long-term price trends
- EMA10 & EMA50: Exponential moving averages that give more weight to recent prices
- RSI: Measures price momentum to detect overbought/oversold conditions
- Bollinger Bands: Indicates price volatility with upper and lower bands
- ATR: Measures market volatility
- OBV: Tracks volume flow to predict price direction
- ADX: Quantifies trend strength
- Stochastic Oscillator: Compares closing price to price range over a period

---

## Interpretation of Results

The software's results indicate how technical indicators impact stock price trends. While the model performs reasonably well, results depend on the quality of stock data and the assumptions underlying the Random Forest model. Confidence scores, as well as training, testing, and cross-validation accuracy provide additional details about the reliability of the softwares predictions.

---

## Improvements
TODO:
1.	Add more technical indicators like MACD, Momentum, or Williams %R.
2.	Include additional data, such as macroeconomic indicators or company fundamentals, to better contextualize the technical indicators
3.	Improve the graph to show various different technical indicators
4.  In general, improve the accuracy of the model

---
## Install Dependencies
pip3 install -r requirements.txt

---

## How to Run
python3 StockAnalysis.py

---

## How to Use
- Open the app and enter a stock ticker (e.g., AAPL).
- Choose a date range for historical data.
- View:
    - Trends: In the Graph tab.
	- Feature Importance: In the Feature Importance tab.
	- Recommendations: In the Recommendation tab.

---

## File Structure
- StockAnalysis.py: Main GUI application script.
- Engine.py: Backend logic for data retrieval, indicator calculation, model training, and prediction.
- requirements.txt: List of dependencies for the app.

---

## Dependencies
- Python 3.8+
- Libraries:
	- PyQt5
	- yfinance
	- pandas
	- matplotlib
	- scikit-learn
	- imbalanced-learn

---

## References
1.	Yahoo Finance API Documentation:
	- For retrieving historical stock data.
	- URL: [Yahoo Finance API](https://pypi.org/project/yfinance/)
2.	Scikit-learn Documentation:
	- For implementing logistic regression, feature scaling, and performance evaluation.
	- URL: [Scikit-learn Documentation](https://scikit-learn.org/stable/)
5.	Python Libraries:
	- PyQt5 for GUI: [PyQt5 Documentation](https://www.riverbankcomputing.com/static/Docs/PyQt5/)
	- Pandas for data manipulation: [Pandas Documentation](https://pandas.pydata.org/docs/)
	- Matplotlib for plotting: [Matplotlib Documentation](https://matplotlib.org/stable/index.html)