# 📈 Stock Prediction and Portfolio Management Platform

A web-based platform for stock price prediction, portfolio management, and performance analysis. This platform allows users to analyze historical stock data, forecast future stock prices using **Prophet** and **LSTM** models, and manage their investment portfolios. The project is built using **Streamlit** for the user interface and includes various tools for data analysis and deep learning.

## 🚀 Features

- **Stock Prediction**: Analyze and forecast stock prices using historical data and machine learning models (Prophet, LSTM).
- **Portfolio Management**: Keep track of stocks in a personalized portfolio, view real-time updates, and analyze profit/loss.
- **Advanced Data Analysis**: Use moving averages, RSI, volatility, and other indicators to make informed decisions.
- **Interactive Visualizations**: Dynamic plots using Plotly for better insight into stock performance.
- **LSTM and Prophet Integration**: Leverage deep learning and statistical models for accurate predictions.

## 🛠️ Tech Stack

- **Frontend**: Streamlit (for the interactive user interface)
- **Data**: yFinance (real-time stock data)
- **Prediction Models**: Prophet (time-series forecasting), LSTM (deep learning)
- **Data Analysis**: Pandas, NumPy, SciKit-Learn, SciPy
- **Visualization**: Plotly
- **Machine Learning**: Keras, TensorFlow

## 🗂️ File Structure

```bash
stock-prediction-app/
│
├── main.py               # Main code for the app, including stock prediction, portfolio management, and UI components.
├── requirements.txt       # List of required libraries and dependencies.
└── stock.csv              # Example stock data used for analysis and forecasting.
