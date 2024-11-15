import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QWidget, QLineEdit, QTabWidget, QDateEdit, QTextEdit
)
from PyQt5.QtCore import QDate, Qt, QTimer
from PyQt5.QtGui import QFont
from core_backend import fetch_stock_data, add_features, rank_features, provide_insight
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


def plot_trader_graph(data):
    fig, ax = plt.subplots(figsize=(10, 5))  # Set a fixed figure size
    ax.plot(data.index[-5:], data['Close'][-5:], label="Close Price", marker='o', color='blue')
    ax.fill_between(data.index[-5:], data['BB_upper'][-5:], data['BB_lower'][-5:], color='orange', alpha=0.2, label="Bollinger Bands")
    ax.plot(data.index[-5:], data['MA10'][-5:], label="10-Day MA", linestyle='--', color='green')
    ax.plot(data.index[-5:], data['MA50'][-5:], label="50-Day MA", linestyle=':', color='purple')

    ax.set_title("Stock Price and Indicators (Last 5 Days)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()  # Automatically adjust layout to prevent clipping

    return fig


class StockAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stock Analysis App")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("background-color: #1e1e1e; color: #ffffff;")

        # Initialize the main layout
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

        # Tab widget to hold different pages
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

        # Introduction text
        intro_text = QTextEdit()
        intro_text.setReadOnly(True)
        intro_text.setStyleSheet("background-color: #2e2e2e; border: none; padding: 10px; font-size: 16px;")
        intro_text.setHtml("""
        <h2><b>Welcome to the Stock Analysis App!</b></h2>
        <p>This application helps you analyze stock price trends using advanced techniques, including:</p>
        <ul>
            <li><b>Moving Averages (10-Day and 50-Day)</b>: Smooth price fluctuations to identify trends.</li>
            <li><b>Relative Strength Index (RSI)</b>: Measures momentum to indicate overbought/oversold conditions.</li>
            <li><b>Bollinger Bands</b>: Highlight volatility and typical price ranges.</li>
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

        # Input fields for stock ticker and dates
        input_layout = QHBoxLayout()
        self.ticker_input = QLineEdit()
        self.ticker_input.setPlaceholderText("Enter Stock Ticker (e.g., AAPL)")
        self.ticker_input.setStyleSheet("padding: 10px; background-color: #ffffff; color: #000000; border-radius: 5px;")

        # Adjusting the start date input
        self.start_date_input = QDateEdit()
        self.start_date_input.setDate(QDate.currentDate().addYears(-5))
        self.start_date_input.setCalendarPopup(True)
        self.start_date_input.setStyleSheet(
            "padding: 10px; background-color: #ffffff; color: #000000; border-radius: 5px; "
            "width: 140px; padding-right: 25px; margin-right: 5px; border: none;"
        )
        self.start_date_input.calendarWidget().setStyleSheet(
            "QCalendarWidget { font-size: 10px; width: 200px; height: 200px; }"
        )

        # Adjusting the end date input
        self.end_date_input = QDateEdit()
        self.end_date_input.setDate(QDate.currentDate())
        self.end_date_input.setCalendarPopup(True)
        self.end_date_input.setStyleSheet(
            "padding: 10px; background-color: #ffffff; color: #000000; border-radius: 5px; "
            "width: 140px; padding-right: 25px; margin-right: 5px; border: none;"
        )
        self.end_date_input.calendarWidget().setStyleSheet(
            "QCalendarWidget { font-size: 10px; width: 200px; height: 200px; }"
        )

        self.submit_button = QPushButton("Submit")
        self.submit_button.setStyleSheet("padding: 10px; background-color: #ffa500; color: #000000; border-radius: 5px;")
        self.submit_button.clicked.connect(self.fetch_data)

        # Status message for feedback
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #ffa500; font-size: 14px; padding: 5px;")

        # Layout adjustments
        input_layout.addWidget(QLabel("<b>Stock Ticker:</b>"))
        input_layout.addWidget(self.ticker_input)
        input_layout.addWidget(QLabel("<b>Start Date:</b>"))
        input_layout.addWidget(self.start_date_input)
        input_layout.addWidget(QLabel("<b>End Date:</b>"))
        input_layout.addWidget(self.end_date_input)
        input_layout.addWidget(self.submit_button)

        layout.addLayout(input_layout)
        layout.addWidget(self.status_label)  # Add status label below the inputs

        intro_tab.setLayout(layout)
        self.tabs.addTab(intro_tab, "Introduction")

    def init_graph_tab(self):
        self.graph_tab = QWidget()
        layout = QVBoxLayout()

        # Placeholder for the graph
        self.graph_canvas = FigureCanvas(plt.figure(figsize=(10, 5)))  # Match the figure size
        self.graph_canvas.setStyleSheet("background-color: #2e2e2e; border: none;")  # Ensure clean design
        layout.addWidget(self.graph_canvas)

        self.graph_tab.setLayout(layout)
        self.tabs.addTab(self.graph_tab, "Graph")

    def init_feature_importance_tab(self):
        self.feature_tab = QWidget()
        layout = QVBoxLayout()

        # Explanation and rankings section
        feature_explanation = QLabel("""
        <h3><b>Feature Importance</b></h3>
        <p>This section provides insights into the key technical indicators used in the machine learning model and their impact on predictions.</p>
        <h4><b>What the Technical Indicators Represent:</b></h4>
        <ul>
            <li><b>Moving Average (10-Day and 50-Day):</b> Averages the stock price over 10 or 50 days to smooth out price fluctuations and identify trends.</li>
            <li><b>Relative Strength Index (RSI):</b> A momentum indicator measuring the magnitude of recent price changes to evaluate whether a stock is overbought or oversold.</li>
            <li><b>Bollinger Bands (Upper and Lower):</b> A volatility indicator showing price levels relative to a moving average, indicating overbought or oversold conditions.</li>
        </ul>
        <h4><b>How Feature Importance is Determined:</b></h4>
        <p>The Random Forest model assigns feature importance based on how much each feature improves the prediction accuracy. Features that split data more effectively to reduce uncertainty are weighted higher.</p>
        <p>For example:
            <ul>
                <li><b>RSI:</b> Highly influential if the stock exhibits clear overbought or oversold behavior.</li>
                <li><b>Moving Averages:</b> Useful for identifying short- and long-term price trends, weighted more if trends are stable.</li>
                <li><b>Bollinger Bands:</b> Significant when price volatility is high, as they provide context for price extremes.</li>
            </ul>
        </p>
        """)
        feature_explanation.setStyleSheet("padding: 10px; font-size: 14px;")
        layout.addWidget(feature_explanation)

        # Placeholder for feature importance text
        self.feature_text = QTextEdit()
        self.feature_text.setReadOnly(True)
        self.feature_text.setStyleSheet("background-color: #2e2e2e; border: none; padding: 10px; font-size: 16px;")
        layout.addWidget(self.feature_text)

        self.feature_tab.setLayout(layout)
        self.tabs.addTab(self.feature_tab, "Feature Importance")

    def init_recommendation_tab(self):
        self.recommendation_tab = QWidget()
        layout = QVBoxLayout()

        # Explanation and recommendation
        recommendation_label = QLabel("""
        <h3><b>Recommendation</b></h3>
        <p>The recommendation is based on the <b>Relative Strength Index (RSI)</b>:</p>
        <ul>
            <li><b>RSI < 30</b>: Oversold (BUY).</li>
            <li><b>RSI > 70</b>: Overbought (SELL).</li>
            <li><b>30 ≤ RSI ≤ 70</b>: Neutral range (HOLD).</li>
        </ul>
        """)
        recommendation_label.setStyleSheet("padding: 10px; font-size: 16px;")
        layout.addWidget(recommendation_label)

        # Placeholder for recommendation text
        self.recommendation_text = QTextEdit()
        self.recommendation_text.setReadOnly(True)
        self.recommendation_text.setFont(QFont("Arial", 18, QFont.Bold))
        self.recommendation_text.setStyleSheet("background-color: #2e2e2e; border: none; padding: 20px;")
        self.recommendation_text.setAlignment(Qt.AlignCenter)
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

            # Update the Graph tab
            fig = plot_trader_graph(data)
            self.graph_canvas.figure = fig
            self.graph_canvas.draw()

            # Update the Feature Importance tab
            features = ['MA10', 'MA50', 'RSI', 'BB_upper', 'BB_lower']
            X = data[features]
            y = data['Target']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            ranked_features = rank_features(model, features)

            # Add detailed explanation of ranked features
            feature_text = "<h4>Feature Importance Rankings:</h4>"
            for rank, (importance, feature) in enumerate(ranked_features, start=1):
                explanation = ""
                if feature == 'MA10':
                    explanation = "Indicates short-term trends by smoothing out daily price fluctuations."
                elif feature == 'MA50':
                    explanation = "Provides a longer-term view of price trends and market momentum."
                elif feature == 'RSI':
                    explanation = "Measures recent price changes to assess overbought/oversold conditions."
                elif feature == 'BB_upper' or feature == 'BB_lower':
                    explanation = "Highlights volatility by showing price levels relative to moving averages."
                feature_text += f"<p><b>{rank}. {feature}</b>: {importance:.4f}<br><i>{explanation}</i></p>"
            self.feature_text.setHtml(feature_text)

            # Update the Recommendation tab
            recommendation, rsi = provide_insight(data)
            self.recommendation_text.setHtml(f"<b>{recommendation}</b><br><br>"
                                             f"<b>Latest RSI:</b> {rsi:.2f}")
            self.status_label.setText("Data successfully loaded.")

            # Start QTimer to clear the status label after 30 seconds
            QTimer.singleShot(30000, self.clear_status_label)

        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")

    def clear_status_label(self):
        self.status_label.setText("")


# Run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StockAnalysisApp()
    window.show()
    sys.exit(app.exec_())
