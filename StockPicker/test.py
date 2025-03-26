import yfinance as yf
data = yf.download("MSFT", start="2000-01-01", end="2025-03-10")
print(data)
