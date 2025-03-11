import yfinance as yf
data = yf.download("MSFT", start="2020-01-01", end="2025-03-10")
print(data.columns)
