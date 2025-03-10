import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
import io
import sys
import threading
import logging
import datetime
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Import the trading system from SP_Back.py
from SP_Back import TradingSystem

# ---------------------------
# Custom Logging Handler
# ---------------------------
class TextHandler(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record)
        def append():
            self.text_widget.insert(tk.END, msg + "\n")
            self.text_widget.see(tk.END)
        self.text_widget.after(0, append)

# ---------------------------
# Functions to Update Various Sections
# ---------------------------
def update_basket(trading_system, basket_widget, basket_size, portfolio_value):
    buffer = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buffer
    try:
        trading_system.print_basket(basket_size=basket_size, portfolio_value=portfolio_value)
    except Exception as e:
        print(f"Error updating basket: {e}")
    sys.stdout = old_stdout
    basket_text = buffer.getvalue()
    basket_widget.config(state=tk.NORMAL)
    basket_widget.delete('1.0', tk.END)
    basket_widget.insert(tk.END, basket_text)
    basket_widget.config(state=tk.DISABLED)

def update_backtest(trading_system, notebook, status_var):
    status_var.set("Running robust backtest...")
    def run_backtest():
        try:
            portfolio_df, trade_log, sharpe = trading_system.robust_backtest()
            if portfolio_df is None:
                status_var.set("Backtest failed.")
                return
            
            # Compute drawdowns
            portfolio_df['RunningMax'] = portfolio_df['PortfolioValue'].cummax()
            portfolio_df['Drawdown'] = (portfolio_df['PortfolioValue'] - portfolio_df['RunningMax']) / portfolio_df['RunningMax']
            daily_returns = portfolio_df['PortfolioValue'].pct_change().dropna()
            
            # Create sub-figures for the Backtest tab
            # Equity Curve
            fig1 = Figure(figsize=(5, 3), dpi=100)
            ax1 = fig1.add_subplot(111)
            ax1.plot(portfolio_df.index, portfolio_df['PortfolioValue'])
            ax1.set_title("Equity Curve")
            ax1.set_xlabel("Date")
            ax1.set_ylabel("Portfolio Value ($)")
            
            # Drawdown Chart
            fig2 = Figure(figsize=(5, 3), dpi=100)
            ax2 = fig2.add_subplot(111)
            ax2.plot(portfolio_df.index, portfolio_df['Drawdown'], color='red')
            ax2.set_title("Drawdown")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Drawdown (%)")
            
            # Histogram of Daily Returns
            fig3 = Figure(figsize=(5, 3), dpi=100)
            ax3 = fig3.add_subplot(111)
            ax3.hist(daily_returns, bins=20, color='gray')
            ax3.set_title("Daily Returns Histogram")
            ax3.set_xlabel("Return")
            ax3.set_ylabel("Frequency")
            
            # Performance Metrics Summary (as text in a figure)
            metrics_text = (
                f"Final Portfolio Value: ${portfolio_df['PortfolioValue'].iloc[-1]:.2f}\n"
                f"Sharpe Ratio: {sharpe:.2f}\n"
                f"Max Drawdown: {portfolio_df['Drawdown'].min()*100:.2f}%\n"
                f"Portfolio Volatility: {daily_returns.std()*100:.2f}%"
            )
            fig4 = Figure(figsize=(5, 3), dpi=100)
            ax4 = fig4.add_subplot(111)
            ax4.text(0.5, 0.5, metrics_text, fontsize=12, ha='center', va='center')
            ax4.axis('off')
            ax4.set_title("Performance Metrics")
            
            # Clear existing sub-tabs and add new ones
            for tab in notebook.winfo_children():
                tab.destroy()
            # Create new tabs in the backtest notebook
            sub_nb = ttk.Notebook(notebook)
            sub_nb.pack(fill=tk.BOTH, expand=True)
            for fig, title in zip([fig1, fig2, fig3, fig4],
                                  ["Equity Curve", "Drawdown", "Daily Returns", "Metrics"]):
                frame = ttk.Frame(sub_nb)
                sub_nb.add(frame, text=title)
                canvas = FigureCanvasTkAgg(fig, master=frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            status_var.set("Backtest complete.")
        except Exception as e:
            status_var.set(f"Error in backtest: {e}")
            logging.error(f"Error in backtest: {e}")
    threading.Thread(target=run_backtest, daemon=True).start()

def update_parameter_tuning(trading_system, status_var):
    status_var.set("Retraining model with new parameters...")
    def retrain():
        try:
            # For simplicity, we call model_selection_and_training again.
            # In a more complete implementation, you'd pass updated hyperparameters.
            trading_system.model_selection_and_training()
            status_var.set("Model retraining complete.")
        except Exception as e:
            status_var.set(f"Error retraining model: {e}")
            logging.error(f"Error retraining model: {e}")
    threading.Thread(target=retrain, daemon=True).start()

def update_risk_dashboard(trading_system, risk_frame, status_var):
    status_var.set("Calculating risk metrics...")
    def calc_risk():
        try:
            portfolio_df, trade_log, sharpe = trading_system.robust_backtest()
            if portfolio_df is None:
                status_var.set("Risk metrics unavailable.")
                return
            final_value = portfolio_df['PortfolioValue'].iloc[-1]
            daily_returns = portfolio_df['PortfolioValue'].pct_change().dropna()
            volatility = daily_returns.std()*100
            portfolio_df['RunningMax'] = portfolio_df['PortfolioValue'].cummax()
            portfolio_df['Drawdown'] = (portfolio_df['PortfolioValue'] - portfolio_df['RunningMax']) / portfolio_df['RunningMax']
            max_drawdown = portfolio_df['Drawdown'].min()*100
            metrics = (
                f"Final Portfolio Value: ${final_value:.2f}\n"
                f"Portfolio Volatility: {volatility:.2f}%\n"
                f"Max Drawdown: {max_drawdown:.2f}%\n"
                f"Sharpe Ratio: {sharpe:.2f}"
            )
            for widget in risk_frame.winfo_children():
                widget.destroy()
            label = tk.Label(risk_frame, text=metrics, justify=tk.LEFT, font=("Arial", 12))
            label.pack(anchor=tk.W, padx=10, pady=10)
            status_var.set("Risk metrics updated.")
        except Exception as e:
            status_var.set(f"Error updating risk dashboard: {e}")
            logging.error(f"Error updating risk dashboard: {e}")
    threading.Thread(target=calc_risk, daemon=True).start()

def update_trade_log(trading_system, log_tree, status_var):
    status_var.set("Fetching trade log...")
    def fetch_log():
        try:
            portfolio_df, trade_log, sharpe = trading_system.robust_backtest()
            if trade_log is None:
                status_var.set("No trade log available.")
                return
            # Clear existing log entries
            for row in log_tree.get_children():
                log_tree.delete(row)
            # Insert each trade
            for trade in trade_log:
                log_tree.insert("", "end", values=(
                    trade.get('date', ''),
                    trade.get('ticker', ''),
                    round(trade.get('entry_price', 0),2),
                    round(trade.get('exit_price', 0),2),
                    round(trade.get('position_size', 0),4),
                    round(trade.get('trade_return', 0)*100,2)
                ))
            status_var.set("Trade log updated.")
        except Exception as e:
            status_var.set(f"Error updating trade log: {e}")
            logging.error(f"Error updating trade log: {e}")
    threading.Thread(target=fetch_log, daemon=True).start()

def update_model_performance(trading_system, canvas_frame, status_var):
    status_var.set("Updating model performance visuals...")
    def update_perf():
        try:
            # Use last 20% of the dataset as holdout sample
            df = trading_system.dataset.copy()
            df = df.sort_values(by='Date')
            split_idx = int(len(df) * 0.8)
            holdout = df.iloc[split_idx:]
            if holdout.empty:
                status_var.set("No holdout data available.")
                return
            X_holdout = holdout[trading_system.features]
            y_holdout = holdout['Target']
            predictions = trading_system.best_model.predict(X_holdout)
            mse = np.mean((predictions - y_holdout.values) ** 2)
            
            # Create a scatter plot: predicted vs actual
            fig = Figure(figsize=(5, 4), dpi=100)
            ax = fig.add_subplot(111)
            ax.scatter(y_holdout, predictions, alpha=0.6)
            ax.plot([y_holdout.min(), y_holdout.max()], [y_holdout.min(), y_holdout.max()], 'r--')
            ax.set_title("Predicted vs. Actual Returns")
            ax.set_xlabel("Actual Return")
            ax.set_ylabel("Predicted Return")
            ax.text(0.05, 0.95, f"MSE: {mse:.6f}", transform=ax.transAxes, fontsize=12,
                    verticalalignment='top')
            for widget in canvas_frame.winfo_children():
                widget.destroy()
            canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            status_var.set("Model performance updated.")
        except Exception as e:
            status_var.set(f"Error updating model performance: {e}")
            logging.error(f"Error updating model performance: {e}")
    threading.Thread(target=update_perf, daemon=True).start()

# ---------------------------
# Initialize Trading System
# ---------------------------
def initialize_trading_system(status_var):
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'UNH']
    start_date = '2020-01-01'
    end_date = datetime.datetime.today().strftime('%Y-%m-%d')
    ts = TradingSystem(tickers, start_date, end_date)
    status_var.set("Downloading data...")
    ts.download_data()
    status_var.set("Building dataset...")
    ts.build_dataset()
    status_var.set("Training model...")
    ts.model_selection_and_training()
    status_var.set("Trading system initialized.")
    return ts

# ---------------------------
# Main GUI Setup
# ---------------------------
root = tk.Tk()
root.title("Trading System GUI")
root.geometry("1000x800")

# Status bar and progress bar
status_var = tk.StringVar()
status_var.set("Initializing...")
status_bar = tk.Label(root, textvariable=status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
status_bar.pack(side=tk.BOTTOM, fill=tk.X)
progress = ttk.Progressbar(root, mode="indeterminate")
progress.pack(side=tk.BOTTOM, fill=tk.X)
progress.start(10)

# Create Notebook for main tabs
notebook = ttk.Notebook(root)
notebook.pack(fill=tk.BOTH, expand=True)

# Tab: Basket
basket_tab = ttk.Frame(notebook)
notebook.add(basket_tab, text="Basket")
basket_text = ScrolledText(basket_tab, width=80, height=20, state=tk.DISABLED)
basket_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
basket_control_frame = ttk.Frame(basket_tab)
basket_control_frame.pack(padx=10, pady=5, fill=tk.X)
ttk.Label(basket_control_frame, text="Basket Size:").pack(side=tk.LEFT, padx=5)
basket_size_entry = ttk.Entry(basket_control_frame, width=5)
basket_size_entry.pack(side=tk.LEFT)
basket_size_entry.insert(0, "5")
ttk.Label(basket_control_frame, text="Portfolio Value ($):").pack(side=tk.LEFT, padx=5)
portfolio_value_entry = ttk.Entry(basket_control_frame, width=10)
portfolio_value_entry.pack(side=tk.LEFT)
portfolio_value_entry.insert(0, "100")
ttk.Button(basket_control_frame, text="Refresh Basket",
           command=lambda: update_basket(trading_system, basket_text, int(basket_size_entry.get()),
                                         float(portfolio_value_entry.get()))).pack(side=tk.LEFT, padx=10)

# Tab: Backtest (with sub-tabs for detailed charts)
backtest_tab = ttk.Frame(notebook)
notebook.add(backtest_tab, text="Backtest")
backtest_notebook = ttk.Notebook(backtest_tab)
backtest_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
ttk.Button(backtest_tab, text="Run Backtest",
           command=lambda: update_backtest(trading_system, backtest_notebook, status_var)).pack(padx=10, pady=5)

# Tab: Parameter Tuning
param_tab = ttk.Frame(notebook)
notebook.add(param_tab, text="Parameter Tuning")
param_frame = ttk.Frame(param_tab)
param_frame.pack(padx=10, pady=10, fill=tk.X)
ttk.Label(param_frame, text="(Example) Adjust Model Hyperparameters:").grid(row=0, column=0, columnspan=2, pady=5)
# For example, you can have entries for RandomForest n_estimators and max_depth.
ttk.Label(param_frame, text="RF n_estimators:").grid(row=1, column=0, sticky=tk.W, padx=5)
rf_estimators_entry = ttk.Entry(param_frame, width=5)
rf_estimators_entry.grid(row=1, column=1, padx=5)
rf_estimators_entry.insert(0, "100")
ttk.Label(param_frame, text="RF max_depth:").grid(row=2, column=0, sticky=tk.W, padx=5)
rf_max_depth_entry = ttk.Entry(param_frame, width=5)
rf_max_depth_entry.grid(row=2, column=1, padx=5)
rf_max_depth_entry.insert(0, "5")
ttk.Button(param_frame, text="Retrain Model",
           command=lambda: update_parameter_tuning(trading_system, status_var)).grid(row=3, column=0, columnspan=2, pady=10)

# Tab: Risk Dashboard
risk_tab = ttk.Frame(notebook)
notebook.add(risk_tab, text="Risk Dashboard")
risk_frame = ttk.Frame(risk_tab)
risk_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
ttk.Button(risk_tab, text="Update Risk Metrics",
           command=lambda: update_risk_dashboard(trading_system, risk_frame, status_var)).pack(padx=10, pady=5)

# Tab: Trade Log
trade_log_tab = ttk.Frame(notebook)
notebook.add(trade_log_tab, text="Trade Log")
columns = ("Date", "Ticker", "Entry Price", "Exit Price", "Position Size", "Trade Return (%)")
trade_tree = ttk.Treeview(trade_log_tab, columns=columns, show="headings")
for col in columns:
    trade_tree.heading(col, text=col)
trade_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
ttk.Button(trade_log_tab, text="Update Trade Log",
           command=lambda: update_trade_log(trading_system, trade_tree, status_var)).pack(padx=10, pady=5)

# Tab: Model Performance
model_perf_tab = ttk.Frame(notebook)
notebook.add(model_perf_tab, text="Model Performance")
model_perf_canvas = ttk.Frame(model_perf_tab)
model_perf_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
ttk.Button(model_perf_tab, text="Update Model Performance",
           command=lambda: update_model_performance(trading_system, model_perf_canvas, status_var)).pack(padx=10, pady=5)

# Tab: Alerts
alerts_tab = ttk.Frame(notebook)
notebook.add(alerts_tab, text="Alerts")
alert_text = ScrolledText(alerts_tab, width=80, height=15, state=tk.NORMAL)
alert_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Configure logging to output to the alerts text widget
logger = logging.getLogger()
logger.setLevel(logging.INFO)
text_handler = TextHandler(alert_text)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
text_handler.setFormatter(formatter)
logger.addHandler(text_handler)

# Global variable for trading system
trading_system = None

# Initialize trading system in a separate thread so GUI stays responsive
def init_trading_system():
    global trading_system
    trading_system = initialize_trading_system(status_var)
    update_basket(trading_system, basket_text, int(basket_size_entry.get()), float(portfolio_value_entry.get()))
    status_var.set("Trading system ready.")
    progress.stop()

threading.Thread(target=init_trading_system, daemon=True).start()

root.mainloop()
