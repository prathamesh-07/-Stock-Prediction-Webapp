import streamlit as st
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from datetime import date
from scipy import stats
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(page_title="Stock Prediction App", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better UI
st.markdown("""
<style>
/* Global styles */
body {
    font-family: 'Roboto Mono', Courier New;
    line-height: 1.6;
}

/* Container background */
.reportview-container {
    background: linear-gradient(135deg, #FFFFFF, #F0F2F6);
}

/* Sidebar styles */
.sidebar .sidebar-content {
    background: linear-gradient(to bottom, #2C3E50, #4CA1AF);
    padding: 20px;
}

/* Widget labels */
.Widget > label {
    color: white !important;
    font-weight: bold;
    margin-bottom: 5px;
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
}

/* Date input styles */
.stDateInput > div > div > input {
    color: black;
    background-color: white;
    border: 1px solid #4CA1AF;
    border-radius: 4px;
    padding: 5px 10px;
}

/* Button styles */
.stButton > button {
    color: black;
    background-color: white;
    border: 2px solid #4CA1AF;
    border-radius: 5px;
    padding: 8px 16px;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    color: white;
    background-color: #4CA1AF;
    transform: translateY(-2px);
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

/* Font size for specific elements */
.css-145kmo2 {
    font-size: 1.2rem;
    font-weight: bold;
    color: #2C3E50;
}

/* Big font class */
.big-font {
    font-size: 3rem !important;
    font-weight: bold;
    color: #2C3E50;
    text-align: center;
    padding: 30px 0;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    letter-spacing: 1px;
}

/* Responsive design */
@media (max-width: 768px) {
    .big-font {
        font-size: 2rem !important;
    }
}
</style>
""", unsafe_allow_html=True)

# Load initial data for stocks
@st.cache_data
def get_data():
    path = 'stock.csv'
    return pd.read_csv(path, low_memory=False)

df = get_data()
df = df.drop_duplicates(subset="Name", keep="first")

# Initialize session state for portfolio
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=['Symbol', 'Shares', 'Purchase Price (₹)'])

# App layout
st.markdown("<h1 class='big-font'>🎯 InvestInsight : Stock Prediction </h1>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.title("⚙️ Settings")

# Navigation
page = st.sidebar.radio("Navigate", ["Stock Prediction", "Portfolio Management"])

if page == "Stock Prediction":
    stocks = df['Name']
    selected_stock = st.sidebar.selectbox("Select a stock for prediction", stocks)

    if selected_stock:
        index = df[df["Name"] == selected_stock].index.values[0]
        symbol = df["Symbol"][index]

        START = st.sidebar.date_input("Start date", date(2015, 1, 1))
        TODAY = st.sidebar.date_input("End date", date.today())

        n_years = st.sidebar.slider("Years to predict", 1, 5, 2, help="Select the number of years for forecast")
        period = n_years * 365

        if st.sidebar.button("Analyze Stock"):
            # Load the stock data using yfinance
            @st.cache_data
            def load_data(ticker):
                try:
                    data = yf.download(ticker, START, TODAY)
                    data.reset_index(inplace=True)
                    return data
                except Exception as e:
                    st.error(f"Error loading data for {ticker}: {e}")
                    return None

            # Technical Indicators
            def plot_moving_averages(data):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Close Price'))
                fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'].rolling(window=50).mean(), name='50-day MA'))
                fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'].rolling(window=200).mean(), name='200-day MA'))
                fig.layout.update(title_text="Stock Price with Moving Averages", xaxis_rangeslider_visible=True)
                st.plotly_chart(fig)

            def calculate_volatility(data, window=21):
                returns = np.log(data['Close'] / data['Close'].shift(1))
                volatility = returns.rolling(window=window).std() * np.sqrt(window)
                return volatility

            def calculate_rsi(data, periods=14):
                close_delta = data['Close'].diff()
                up = close_delta.clip(lower=0)
                down = -1 * close_delta.clip(upper=0)
                ma_up = up.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
                ma_down = down.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
                rsi = ma_up / (ma_up + ma_down) * 100
                return rsi

            def calculate_momentum(data, period=14):
                return data['Close'].diff(period)

            def calculate_bollinger_bands(data, window=20):
                rolling_mean = data['Close'].rolling(window=window).mean()
                rolling_std = data['Close'].rolling(window=window).std()
                upper_band = rolling_mean + (rolling_std * 2)
                lower_band = rolling_mean - (rolling_std * 2)
                return upper_band, lower_band

            def plot_volume_analysis(data):
                fig = go.Figure()
                fig.add_trace(go.Bar(x=data['Date'], y=data['Volume'], name='Volume'))
                fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Close Price', yaxis='y2'))
                fig.layout.update(title_text="Stock Price and Volume Analysis", yaxis2=dict(overlaying='y', side='right'))
                st.plotly_chart(fig)

            @st.cache_data
            def get_market_index(start_date, end_date, index_symbol='^BSESN'):  # BSE Sensex
                return yf.download(index_symbol, start=start_date, end=end_date)['Close']

            with st.spinner('Loading stock data...'):
                data = load_data(symbol)

            if data is not None:
                st.subheader(f"📊 {selected_stock} Stock Data")
                st.dataframe(data.tail(), use_container_width=True)

                # Plot the raw stock data
                def plot_raw_data():
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Stock Open', line=dict(color='#1f77b4')))
                    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Stock Close', line=dict(color='#2ca02c')))
                    fig.layout.update(
                        title_text=f"{selected_stock} Time Series Data", 
                        xaxis_title="Date",
                        yaxis_title="Price (₹)",
                        xaxis_rangeslider_visible=True
                    )
                    return fig

                st.plotly_chart(plot_raw_data(), use_container_width=True)

                # Forecasting using Prophet
                df_train = data[['Date', 'Close']]
                df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

                with st.spinner('Generating forecast...'):
                    m = Prophet()
                    m.fit(df_train)
                    future = m.make_future_dataframe(periods=period)
                    forecast = m.predict(future)

                st.subheader("🔮 Forecast Data")
                st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(), use_container_width=True)

                fig1 = plot_plotly(m, forecast)
                fig1.update_layout(
                    title=f"{selected_stock} Forecast",
                    xaxis_title="Date",
                    yaxis_title="Price (₹)",
                    autosize=False,
                    width=1000,
                    height=600,
                )
                st.plotly_chart(fig1, use_container_width=True)

                st.subheader("📈 Forecast Components")
                fig2 = m.plot_components(forecast)
                st.pyplot(fig2, use_container_width=True)

                # Download button for forecast data
                csv = forecast.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Forecast CSV",
                    data=csv,
                    file_name=f'{selected_stock}_forecast.csv',
                    mime='text/csv',
                )

                # LSTM Model
                st.subheader("🧠 LSTM Model Predictions")

                # Prepare data for LSTM
                scaler = MinMaxScaler(feature_range=(0,1))
                scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

                prediction_days = 60

                x_train = []
                y_train = []

                for x in range(prediction_days, len(scaled_data)):
                    x_train.append(scaled_data[x-prediction_days:x, 0])
                    y_train.append(scaled_data[x, 0])

                x_train, y_train = np.array(x_train), np.array(y_train)
                x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

                # Build the LSTM model
                model = Sequential()
                model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
                model.add(LSTM(units=50, return_sequences=False))
                model.add(Dense(units=25))
                model.add(Dense(units=1))

                model.compile(optimizer='adam', loss='mean_squared_error')
                model.fit(x_train, y_train, epochs=25, batch_size=32, verbose=0)

                # Test the model
                test_start = date(2020, 1, 1)
                test_end = date.today()
                test_data = yf.download(symbol, start=test_start, end=test_end)
                actual_prices = test_data['Close'].values

                total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

                model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
                model_inputs = model_inputs.reshape(-1, 1)
                model_inputs = scaler.transform(model_inputs)

                x_test = []

                for x in range(prediction_days, len(model_inputs)):
                    x_test.append(model_inputs[x-prediction_days:x, 0])

                x_test = np.array(x_test)
                x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

                predicted_prices = model.predict(x_test)
                predicted_prices = scaler.inverse_transform(predicted_prices)

                # Plot LSTM predictions
                fig_lstm = go.Figure()
                fig_lstm.add_trace(go.Scatter(x=test_data.index, y=actual_prices, name='Actual Price', line=dict(color='#1f77b4')))
                fig_lstm.add_trace(go.Scatter(x=test_data.index, y=predicted_prices.flatten(), name='Predicted Price', line=dict(color='#ff7f0e')))
                fig_lstm.layout.update(
                    title_text=f'{selected_stock} LSTM Predictions vs Actual Prices',
                    xaxis_title='Date',
                    yaxis_title='Price (₹)',
                    xaxis_rangeslider_visible=True
                )
                st.plotly_chart(fig_lstm, use_container_width=True)

                # Comparison Graph
                st.subheader("🔍 Model Comparison: Actual vs LSTM vs Prophet")
                fig_comparison = go.Figure()
                fig_comparison.add_trace(go.Scatter(x=test_data.index, y=actual_prices, name='Actual Prices', line=dict(color='#1f77b4')))
                fig_comparison.add_trace(go.Scatter(x=test_data.index, y=predicted_prices.flatten(), name='LSTM Predictions', line=dict(color='#ff7f0e')))
                fig_comparison.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Prophet Forecast', line=dict(color='#2ca02c')))
                fig_comparison.layout.update(
                    title_text=f'{selected_stock} Stock Price Prediction Comparison',
                    xaxis_title='Date',
                    yaxis_title='Price (₹)',
                    xaxis_rangeslider_visible=True
                )
                st.plotly_chart(fig_comparison, use_container_width=True)

                st.subheader("➕ Additional Technical Indicators")
                
                # Moving Averages
                st.write("Moving Averages")
                plot_moving_averages(data)
                
                # Volatility
                volatility = calculate_volatility(data)
                st.write("21-day Rolling Volatility")
                st.line_chart(volatility)
                
                # RSI
                rsi = calculate_rsi(data)
                st.write("Relative Strength Index (RSI)")
                st.line_chart(rsi)
                
                # Momentum
                momentum = calculate_momentum(data)
                st.write("14-day Momentum")
                st.line_chart(momentum)
                
                # Bollinger Bands
                upper_band, lower_band = calculate_bollinger_bands(data)
                st.write("Bollinger Bands")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Close Price'))
                fig.add_trace(go.Scatter(x=data['Date'], y=upper_band, name='Upper Band'))
                fig.add_trace(go.Scatter(x=data['Date'], y=lower_band, name='Lower Band'))
                st.plotly_chart(fig)
                
                # Volume Analysis
                st.write("Volume Analysis")
                plot_volume_analysis(data)

                # Investment Suggestion
                st.subheader("💡 Investment Suggestion")
                
                # Calculate the average of the last 5 predicted prices from both LSTM and Prophet
                last_5_lstm = predicted_prices[-5:].mean()
                last_5_prophet = forecast['yhat'].tail(5).mean()
                
                # Calculate the current price (last closing price)
                current_price = data['Close'].iloc[-1]
                
                # Calculate the average predicted price
                avg_predicted_price = (last_5_lstm + last_5_prophet) / 2
                
                # Calculate the predicted change
                predicted_change = (avg_predicted_price - current_price) / current_price * 100
                
                if predicted_change > 5:
                    suggestion = f"Based on our models, the stock price of {selected_stock} is predicted to increase by approximately {predicted_change:.2f}%. This suggests a potentially favorable investment opportunity. Consider buying or holding this stock."
                elif predicted_change < -5:
                    suggestion = f"Our models predict a potential decrease of approximately {abs(predicted_change):.2f}% in the stock price of {selected_stock}. You might want to be cautious or consider selling if you currently hold this stock."
                else:
                    suggestion = f"The models predict relatively stable prices for {selected_stock}, with a change of approximately {predicted_change:.2f}%. This might be a good time to hold your current position or watch the stock closely for any new developments."
                
                st.write(suggestion)
                
                st.warning("Please note that this suggestion is based on historical data and model predictions. The stock market can be unpredictable, and past performance doesn't guarantee future results. Always conduct your own research and consider consulting with a financial advisor before making investment decisions.")

            else:
               st.error("Failed to load data. Please try a different stock or date range.")

        else:
         st.info("Please select a stock from the sidebar to begin analysis.")
        
    
elif page == "Portfolio Management":
    st.subheader("🗃️ Portfolio Management")

    # Initialize session state for portfolio if it doesn't exist
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = pd.DataFrame(columns=['Symbol', 'Shares', 'Purchase Price per Share (₹)', 'Total Investment (₹)'])

    # Input for adding stocks to portfolio
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        symbol = st.text_input("Stock Symbol").upper()
    with col2:
        shares = st.number_input("Number of Shares", min_value=1, value=1)
    with col3:
        price_per_share = st.number_input("Purchase Price per Share (₹)", min_value=0.01, value=1.00, step=0.01)
    with col4:
        if st.button("Add to Portfolio"):
            if symbol and shares and price_per_share:
                new_stock = pd.DataFrame({
                    'Symbol': [symbol], 
                    'Shares': [shares], 
                    'Purchase Price per Share (₹)': [price_per_share],
                    'Total Investment (₹)': [shares * price_per_share]
                })
                st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_stock], ignore_index=True)
                st.success(f"Added {shares} shares of {symbol} at ₹{price_per_share} per share to your portfolio.")
            else:
                st.warning("Please fill in all fields before adding to portfolio.")

    # Display and edit portfolio
    st.subheader("Your Portfolio")
    if not st.session_state.portfolio.empty:
        # Calculate "Total Investment (₹)" before passing the DataFrame to st.data_editor
        st.session_state.portfolio['Total Investment (₹)'] = st.session_state.portfolio['Shares'] * st.session_state.portfolio['Purchase Price per Share (₹)']
        
        edited_portfolio = st.data_editor(
            st.session_state.portfolio.drop(columns=['Total Investment (₹)']),  # Drop the column to prevent editing
            num_rows="dynamic",
            column_config={
                "Symbol": st.column_config.TextColumn("Symbol", help="Stock symbol"),
                "Shares": st.column_config.NumberColumn("Shares", help="Number of shares", min_value=0, step=1),
                "Purchase Price per Share (₹)": st.column_config.NumberColumn("Purchase Price per Share (₹)", help="Price per share at purchase", min_value=0.01, format="₹%.2f"),
            },
            hide_index=True,
        )
        
        # Recalculate "Total Investment (₹)" after editing
        edited_portfolio['Total Investment (₹)'] = edited_portfolio['Shares'] * edited_portfolio['Purchase Price per Share (₹)']
        st.session_state.portfolio = edited_portfolio

        # Display the total investment column separately
        st.write("Portfolio with Total Investment:")
        st.write(st.session_state.portfolio)

        # Calculate and display portfolio statistics
        total_investment = st.session_state.portfolio['Total Investment (₹)'].sum()
        
        # Fetch current prices
        current_prices = {}
        with st.spinner("Fetching current stock prices..."):
            for symbol in st.session_state.portfolio['Symbol'].unique():
                try:
                    ticker = yf.Ticker(symbol)
                    current_prices[symbol] = ticker.history(period="1d")['Close'].iloc[-1]
                except Exception as e:
                    st.error(f"Error fetching price for {symbol}: {str(e)}")
                    current_prices[symbol] = None
        
        st.session_state.portfolio['Current Price (₹)'] = st.session_state.portfolio['Symbol'].map(current_prices)
        st.session_state.portfolio['Current Value (₹)'] = st.session_state.portfolio['Shares'] * st.session_state.portfolio['Current Price (₹)']
        st.session_state.portfolio['Profit/Loss (₹)'] = st.session_state.portfolio['Current Value (₹)'] - st.session_state.portfolio['Total Investment (₹)']
        st.session_state.portfolio['Profit/Loss (%)'] = (st.session_state.portfolio['Profit/Loss (₹)'] / st.session_state.portfolio['Total Investment (₹)']) * 100
        
        total_current_value = st.session_state.portfolio['Current Value (₹)'].sum()
        total_profit_loss = st.session_state.portfolio['Profit/Loss (₹)'].sum()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Investment", f"₹{total_investment:.2f}")
        col2.metric("Current Value", f"₹{total_current_value:.2f}")
        col3.metric("Total Profit/Loss", f"₹{total_profit_loss:.2f}", f"{(total_profit_loss/total_investment)*100:.2f}%")

        # Portfolio Composition Pie Chart
        fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
        fig.add_trace(go.Pie(labels=st.session_state.portfolio['Symbol'], values=st.session_state.portfolio['Current Value (₹)'], name="Current Value"), 1, 1)
        fig.add_trace(go.Pie(labels=st.session_state.portfolio['Symbol'], values=st.session_state.portfolio['Profit/Loss (₹)'], name="Profit/Loss"), 1, 2)
        fig.update_layout(title="Portfolio Composition and Performance")
        st.plotly_chart(fig)

        # Historical Performance
        st.subheader("Historical Performance")
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=365)
        
        portfolio_history = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date, freq='D'))
        
        with st.spinner("Calculating historical performance..."):
            for symbol in st.session_state.portfolio['Symbol']:
                stock_data = yf.download(symbol, start=start_date, end=end_date)['Close']
                portfolio_history[symbol] = stock_data * st.session_state.portfolio.loc[st.session_state.portfolio['Symbol'] == symbol, 'Shares'].values[0]
        
        portfolio_history['Total'] = portfolio_history.sum(axis=1)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=portfolio_history.index, y=portfolio_history['Total'], mode='lines', name='Portfolio Value'))
        fig.update_layout(title="Portfolio Historical Performance", xaxis_title="Date", yaxis_title="Value (₹)")
        st.plotly_chart(fig)

        # Risk Analysis
        st.subheader("Risk Analysis")
        returns = portfolio_history['Total'].pct_change().dropna()
        
        col1, col2 = st.columns(2)
        col1.metric("Portfolio Volatility (Annual)", f"{returns.std() * np.sqrt(252):.2%}")
        col2.metric("Sharpe Ratio (Risk-Free Rate: 2%)", f"{(returns.mean() - 0.02/252) / (returns.std() * np.sqrt(252)):.2f}")

        # Display updated portfolio with current prices and profit/loss
        st.subheader("Updated Portfolio")
        st.dataframe(st.session_state.portfolio.style.format({
            'Purchase Price per Share (₹)': '₹{:.2f}',
            'Current Price (₹)': '₹{:.2f}',
            'Current Value (₹)': '₹{:.2f}',
            'Profit/Loss (₹)': '₹{:.2f}',
            'Profit/Loss (%)': '{:.2f}%'
        }).background_gradient(cmap='RdYlGn', subset=['Profit/Loss (%)']), hide_index=True)

    else:
        st.info("Your portfolio is empty. Add some stocks to get started!")

    # Add a back button to return to the main page
    if st.button("Back to Stock Prediction"):
        st.experimental_set_query_params(page="Stock Prediction")
        st.experimental_rerun()
