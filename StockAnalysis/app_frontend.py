import sys
import matplotlib.pyplot as plt  # Import added for FigureCanvas
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QWidget, QLineEdit, QTabWidget, QDateEdit, QTextEdit
)
from PyQt5.QtCore import QDate, Qt, QTimer
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from core_backend import fetch_stock_data, add_features, train_model, rank_features, provide_insight, plot_trader_graph


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

        input_layout.addWidget(QLabel("<b>Stock Ticker:</b>"))
        input_layout.addWidget(self.ticker_input)
        input_layout.addWidget(QLabel("<b>Start Date:</b>"))
        input_layout.addWidget(self.start_date_input)
        input_layout.addWidget(QLabel("<b>End Date:</b>"))
        input_layout.addWidget(self.end_date_input)
        input_layout.addWidget(self.submit_button)
        layout.addLayout(input_layout)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #ffa500; font-size: 14px; padding: 5px;")
        layout.addWidget(self.status_label)

        intro_tab.setLayout(layout)
        self.tabs.addTab(intro_tab, "Introduction")

    def init_graph_tab(self):
        self.graph_tab = QWidget()
        layout = QVBoxLayout()

        self.graph_canvas = FigureCanvas(plt.figure(figsize=(10, 5)))  # Use plt to create an empty figure
        layout.addWidget(self.graph_canvas)

        self.graph_tab.setLayout(layout)
        self.tabs.addTab(self.graph_tab, "Graph")

    def init_feature_importance_tab(self):
        self.feature_tab = QWidget()
        layout = QVBoxLayout()

        explanation = QTextEdit()
        explanation.setReadOnly(True)
        explanation.setStyleSheet("padding: 10px; background-color: #2e2e2e; font-size: 16px;")
        explanation.setHtml("""
        <h3>Feature Importance</h3>
        <p>Details about how technical features contribute to predictions.</p>
        """)

        self.feature_text = QTextEdit()
        self.feature_text.setReadOnly(True)
        self.feature_text.setStyleSheet("padding: 10px; background-color: #2e2e2e; font-size: 16px;")

        layout.addWidget(explanation)
        layout.addWidget(self.feature_text)

        self.feature_tab.setLayout(layout)
        self.tabs.addTab(self.feature_tab, "Feature Importance")

    def init_recommendation_tab(self):
        self.recommendation_tab = QWidget()
        layout = QVBoxLayout()

        self.recommendation_text = QTextEdit()
        self.recommendation_text.setReadOnly(True)
        self.recommendation_text.setStyleSheet("padding: 10px; background-color: #2e2e2e; font-size: 16px;")

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
            # Fetch and preprocess data
            data = fetch_stock_data(ticker, start_date, end_date)
            data = add_features(data)

            # Update the Graph tab
            fig = plot_trader_graph(data)
            self.graph_canvas.figure = fig
            self.graph_canvas.draw()

            # Update the Feature Importance tab
            features = ['MA10', 'MA50', 'RSI', 'BB_upper', 'BB_lower']
            model = train_model(data, features)
            ranked_features = rank_features(model, features)

            feature_text = "<h4>Feature Importance Rankings:</h4>"
            for rank, (importance, feature) in enumerate(ranked_features, start=1):
                explanation = ""
                if feature == 'MA10':
                    explanation = "Indicates short-term trends by smoothing out daily price fluctuations."
                elif feature == 'MA50':
                    explanation = "Provides a longer-term view of price trends and market momentum."
                elif feature == 'RSI':
                    explanation = "Measures recent price changes to assess overbought/oversold conditions."
                elif feature in ['BB_upper', 'BB_lower']:
                    explanation = "Highlights volatility by showing price levels relative to moving averages."
                feature_text += f"<p><b>{rank}. {feature}</b>: {importance:.4f}<br><i>{explanation}</i></p>"
            self.feature_text.setHtml(feature_text)

            # Update the Recommendation tab
            latest_data = data[features].iloc[-1:].values
            prediction = model.predict(latest_data)[0]
            recommendation = "Hold"
            if prediction == 1:
                recommendation = "Buy - The stock is expected to increase in value."
            elif prediction == -1:
                recommendation = "Sell - The stock is expected to decrease in value."

            self.recommendation_text.setHtml(f"<b>{recommendation}</b><br><br>"
                                             f"<b>Model Considerations:</b> This recommendation is based on the analysis of features like RSI, Moving Averages, and Bollinger Bands.")
            self.status_label.setText("Data successfully loaded.")

            QTimer.singleShot(30000, self.clear_status_label)

        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")

    def clear_status_label(self):
        self.status_label.setText("")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StockAnalysisApp()
    window.show()
    sys.exit(app.exec_())
