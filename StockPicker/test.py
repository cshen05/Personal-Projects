import yfinance as yf
data = yf.download("AAPL", start="2020-01-01", end="2025-03-10")
print(data)
