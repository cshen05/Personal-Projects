# Stock Analysis App

## Overview
The Stock Analysis App provides users with tools to analyze stock trends using technical indicators and machine learning models. It generates BUY/SELL recommendations with confidence scores based on historical stock data.

---

## Features
- **Retrieve Stock Data**: Fetch historical stock data for any publicly traded company.
- **Technical Analysis**:
  - Moving Averages (MA10, MA50)
  - Exponential Moving Averages (EMA10, EMA50)
  - RSI, Bollinger Bands, ATR, OBV, ADX, and more.
- **Graphical Visualization**: Visualize stock trends over the last 5 days.
- **Feature Importance**: Understand how indicators contribute to predictions.
- **Recommendations**: Receive BUY/SELL advice based on machine learning predictions.

---

## Install Dependencies
pip3 install -r requirements.txt

---

## How to Run
python3 StockAnalysis.py

___

## How to Use
- Open the app and enter a stock ticker (e.g., AAPL).
- Choose a date range for historical data.
- View:
    - Trends: In the Graph tab.
	- Feature Importance: In the Feature Importance tab.
	- Recommendations: In the Recommendation tab.

## File Structure
- StockAnalysis.py: Main GUI application script.
- Engine.py: Backend logic for data retrieval, indicator calculation, model training, and prediction.
- requirements.txt: List of dependencies for the app.

---

## Requirements
- Python 3.8+
- Libraries:
	- PyQt5
	- yfinance
	- pandas
	- matplotlib
	- scikit-learn
	- imbalanced-learn