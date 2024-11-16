import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QWidget, QLineEdit, QTabWidget, QDateEdit, QTextEdit, QScrollArea
)
from PyQt5.QtCore import QDate, QTimer
from PyQt5.QtGui import QFont

from Engine import fetch_stock_data, add_features, rank_features, provide_insight, train_and_evaluate_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

def plot_trader_graph(data):
    fig, ax = plt.subplots(figsize=(8, 4))  # Adjust these values for better scaling
    ax.plot(data.index[-5:], data['Close'][-5:], label="Close Price", marker='o', color='blue')
    ax.fill_between(data.index[-5:], data['BB_upper'][-5:], data['BB_lower'][-5:], color='orange', alpha=0.2, label="Bollinger Bands")
    ax.plot(data.index[-5:], data['EMA10'][-5:], label="10-Day EMA", linestyle='--', color='green')
    ax.plot(data.index[-5:], data['EMA50'][-5:], label="50-Day EMA", linestyle=':', color='purple')

    ax.set_title("Stock Price and Indicators (Last 5 Days)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

class StockAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stock Analysis App")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("background-color: #1e1e1e; color: #ffffff;")

        self.layout = QVBoxLayout()

        # Header Section
        header_layout = QVBoxLayout()
        header_title = QLabel("<b>Stock Analysis App</b>")
        header_title.setFont(QFont("Arial", 28, QFont.Bold))
        header_title.setStyleSheet("color: #ffa500; padding: 10px;")
        header_subtitle = QLabel("Analyze stocks using advanced technical indicators and machine learning.")
        header_subtitle.setFont(QFont("Arial", 16))
        header_subtitle.setStyleSheet("padding: 5px;")
        header_layout.addWidget(header_title)
        header_layout.addWidget(header_subtitle)
        self.layout.addLayout(header_layout)

        # Tab widget
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        # Add tabs
        self.init_intro_tab()
        self.init_graph_tab()
        self.init_feature_importance_tab()
        self.init_recommendation_tab()

        # Main widget
        main_widget = QWidget()
        main_widget.setLayout(self.layout)
        self.setCentralWidget(main_widget)

    def init_intro_tab(self):
        intro_tab = QWidget()
        layout = QVBoxLayout()

        intro_text = QTextEdit()
        intro_text.setReadOnly(True)
        intro_text.setStyleSheet("background-color: #2e2e2e; border: none; padding: 10px; font-size: 16px;")
        intro_text.setHtml("""
        <h2><b>Welcome to the Stock Analysis App!</b></h2>
        <p>This application analyzes stock price trends using advanced techniques, including:</p>
        <ul>
            <li><b>Moving Averages (10-Day EMA and 50-Day EMA)</b>: React quickly to price trends.</li>
            <li><b>Average True Range (ATR)</b>: Measures market volatility.</li>
            <li><b>Stochastic Oscillator:</b> Indicates price momentum.</li>
            <li><b>On-Balance Volume (OBV):</b> Tracks volume flow to predict price changes.</li>
            <li><b>ADX:</b> Measures the strength of a trend.</li>
        </ul>
        <p><b>How to Use:</b></p>
        <ol>
            <li>Enter the stock ticker and desired date range in the input fields below.</li>
            <li>View stock trends in the <b>Graph</b> tab.</li>
            <li>Check feature importance in the <b>Feature Importance</b> tab.</li>
            <li>Get actionable recommendations in the <b>Recommendation</b> tab.</li>
        </ol>
        """)
        layout.addWidget(intro_text)

        input_layout = QHBoxLayout()
        self.ticker_input = QLineEdit()
        self.ticker_input.setPlaceholderText("Enter Stock Ticker (e.g., AAPL)")
        self.ticker_input.setStyleSheet("padding: 10px; background-color: #ffffff; color: #000000; border-radius: 5px;")

        self.start_date_input = QDateEdit()
        self.start_date_input.setDate(QDate.currentDate().addYears(-5))
        self.start_date_input.setCalendarPopup(True)
        self.start_date_input.setStyleSheet("padding: 10px;")

        self.end_date_input = QDateEdit()
        self.end_date_input.setDate(QDate.currentDate())
        self.end_date_input.setCalendarPopup(True)
        self.end_date_input.setStyleSheet("padding: 10px;")

        self.submit_button = QPushButton("Submit")
        self.submit_button.setStyleSheet("padding: 10px; background-color: #ffa500; color: #000000;")
        self.submit_button.clicked.connect(self.fetch_data)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #ffa500; font-size: 14px; padding: 5px;")

        input_layout.addWidget(QLabel("<b>Stock Ticker:</b>"))
        input_layout.addWidget(self.ticker_input)
        input_layout.addWidget(QLabel("<b>Start Date:</b>"))
        input_layout.addWidget(self.start_date_input)
        input_layout.addWidget(QLabel("<b>End Date:</b>"))
        input_layout.addWidget(self.end_date_input)
        input_layout.addWidget(self.submit_button)

        layout.addLayout(input_layout)
        layout.addWidget(self.status_label)

        intro_tab.setLayout(layout)
        self.tabs.addTab(intro_tab, "Introduction")

    def init_graph_tab(self):
        self.graph_tab = QWidget()
        layout = QVBoxLayout()
        self.graph_canvas = FigureCanvas(plt.figure(figsize=(10, 5)))
        layout.addWidget(self.graph_canvas)
        self.graph_tab.setLayout(layout)
        self.tabs.addTab(self.graph_tab, "Graph")

    def init_feature_importance_tab(self):
        self.feature_tab = QWidget()
        main_layout = QVBoxLayout()

        # fixed container for feature weights (non-scrollable)
        feature_weights_container = QWidget()
        feature_weights_layout = QVBoxLayout()
        feature_weights_container.setLayout(feature_weights_layout)

        feature_importance_explanation = QLabel("""
        <h3><b>Understanding Feature Importance</b></h3>
        <p>Feature importance quantifies the contribution of each indicator to the machine learning model's predictions. 
        It is calculated based on the decrease in model accuracy or variance when a feature is excluded or randomized. 
        In this app, a Random Forest Classifier is used, which evaluates feature importance by measuring how much 
        each feature splits data effectively across decision trees.</p>
        """)
        feature_importance_explanation.setStyleSheet("padding: 10px; font-size: 16px;")
        feature_importance_explanation.setWordWrap(True)
        feature_weights_layout.addWidget(feature_importance_explanation)

        self.feature_text = QTextEdit()
        self.feature_text.setReadOnly(True)
        self.feature_text.setStyleSheet("background-color: #2e2e2e; border: none; padding: 10px; font-size: 16px;")
        self.feature_text.setFixedHeight(245)  # fixed height for feature weights display
        feature_weights_layout.addWidget(self.feature_text)

        # add fixed feature weights container to the main layout
        main_layout.addWidget(feature_weights_container)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        container_widget = QWidget()
        container_layout = QVBoxLayout()

        indicator_explanation = QLabel("""
        <h3><b>Explanation of Indicators</b></h3>
        <p>Below are the key technical indicators used in stock analysis and their significance:</p>
        <ul>
            <li><b>EMA10 (10-Day Exponential Moving Average):</b> A moving average that gives more weight to recent prices, 
            making it responsive to short-term trends.</li>
            <li><b>EMA50 (50-Day Exponential Moving Average):</b> Similar to EMA10 but spans a longer period, helping 
            to identify medium to long-term trends.</li>
            <li><b>ATR (Average True Range):</b> A measure of market volatility, indicating the average price movement 
            (high to low) over a specified period.</li>
            <li><b>Stochastic Oscillator:</b> A momentum indicator comparing the closing price to a range of prices over 
            a specific period. It helps identify overbought or oversold conditions.</li>
            <li><b>OBV (On-Balance Volume):</b> A cumulative volume-based indicator that predicts price direction by 
            analyzing whether volume flows in or out of a stock.</li>
            <li><b>ADX (Average Directional Index):</b> Measures the strength of a trend without indicating its direction. 
            Higher ADX values signify stronger trends.</li>
            <li><b>Bollinger Bands (BB Upper and Lower):</b> Bands plotted two standard deviations above and below a 
            moving average. They show volatility and potential reversal points when prices touch the bands.</li>
            <li><b>RSI (Relative Strength Index):</b> A momentum oscillator that measures the speed and change of price 
            movements to identify overbought or oversold conditions.</li>
            <li><b>MA10 (10-Day Moving Average):</b> A simple moving average over the past 10 days, used to smooth out 
            price fluctuations and identify trends.</li>
            <li><b>MA50 (50-Day Moving Average):</b> Similar to MA10 but over a longer period, providing insights into 
            longer-term trends.</li>
        </ul>
        """)
        indicator_explanation.setStyleSheet("padding: 10px; font-size: 16px;")
        indicator_explanation.setWordWrap(True)
        container_layout.addWidget(indicator_explanation)

        container_widget.setLayout(container_layout)
        scroll_area.setWidget(container_widget)

        main_layout.addWidget(scroll_area)

        self.feature_tab.setLayout(main_layout)
        self.tabs.addTab(self.feature_tab, "Feature Importance")

    def init_recommendation_tab(self):
        self.recommendation_tab = QWidget()
        layout = QVBoxLayout()

        recommendation_label = QLabel("""
        <h3><b>Recommendation</b></h3>
        <p>The recommendation is based on the machine learning model's prediction:</p>
        <ul>
            <li><b>BUY:</b> The stock is predicted to rise.</li>
            <li><b>SELL:</b> The stock is predicted to fall.</li>
        </ul>
        """)
        recommendation_label.setStyleSheet("padding: 10px; font-size: 16px;")
        layout.addWidget(recommendation_label)

        self.recommendation_text = QTextEdit()
        self.recommendation_text.setReadOnly(True)
        self.recommendation_text.setFont(QFont("Arial", 18, QFont.Bold))
        self.recommendation_text.setStyleSheet("background-color: #2e2e2e; border: none; padding: 20px;")
        layout.addWidget(self.recommendation_text)

        self.recommendation_tab.setLayout(layout)
        self.tabs.addTab(self.recommendation_tab, "Recommendation")

    def fetch_data(self):
        ticker = self.ticker_input.text().strip().upper()
        start_date = self.start_date_input.date().toString("yyyy-MM-dd")
        end_date = self.end_date_input.date().toString("yyyy-MM-dd")

        self.status_label.setText("Fetching data, please wait...")

        if not ticker:
            self.status_label.setText("Error: Please enter a valid stock ticker.")
            return

        try:
            data = fetch_stock_data(ticker, start_date, end_date)
            data = add_features(data)

            fig = plot_trader_graph(data)
            self.graph_canvas.figure = fig
            self.graph_canvas.draw()

            features = ['MA10', 'MA50', 'RSI', 'BB_upper', 'BB_lower', 'EMA10', 'EMA50', 'ATR', 'Stochastic', 'OBV', 'ADX']
            X = data[features]
            y = data['Target']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model, train_accuracy, test_accuracy, cross_val_mean, cross_val_std = train_and_evaluate_model(X_train, y_train, X_test, y_test)
            
            try:
                ranked_features = rank_features(model, features)
                formatted_features = "<br>".join(
                    [f"<b>{feature}</b>: {float(importance):.4f}" for importance, feature in ranked_features]
                )
                self.feature_text.setHtml(formatted_features)
            except Exception as e:
                self.feature_text.setHtml(f"<b>Error displaying feature importance:</b> {str(e)}")
            
            try:
                recommendation, confidence = provide_insight(model, X_test)
                self.recommendation_text.setHtml(f"""
                <h3>Recommendation: {recommendation}</h3>
                <p>Confidence: {confidence*100.0:.2f}%</p>
                <p>Training Accuracy: {train_accuracy*100.0:.2f}%</p>
                <p>Test Accuracy: {test_accuracy*100.0:.2f}%</p>
                <p>Cross-Validation Accuracy: {cross_val_mean*100.0:.2f}% ± {cross_val_std*100.0:.2f}%</p>
                <p>The recommendation is based on the model's evaluation of probabilities for <b>BUY</b> and <b>SELL</b>. 
                Confidence reflects the likelihood of this specific prediction being correct, while accuracy measures overall 
                model performance.</p>
                """)
            except Exception as e:
                self.recommendation_text.setHtml(f"<b>Error generating recommendation:</b> {str(e)}")

            self.status_label.setText("Data successfully loaded.")
            QTimer.singleShot(5000, self.clear_status_label)

        except Exception as e:
            print(e)
            self.status_label.setText(f"Error: {str(e)}")

    def clear_status_label(self):
        self.status_label.setText("")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StockAnalysisApp()
    window.show()
    sys.exit(app.exec_())
