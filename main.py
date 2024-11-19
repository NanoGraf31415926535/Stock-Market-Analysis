import logging
from data_collection import get_stock_data
from eda import plot_series
from statistical_analysis import calculate_correlation, calculate_volatility, stationarity_test, seasonal_decompose_data
from predictive_model import prepare_data_for_lstm, build_and_train_lstm, plot_predictions, save_model
from utils import print_summary

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
symbol = "AAPL"
start_date = "2024-06-01"
end_date = "2024-11-07"

# Step 1: Data Collection
logging.info(f"Starting data collection for {symbol} from {start_date} to {end_date}.")
data = get_stock_data(symbol, start_date, end_date)

if data is not None and not data.empty:
    print_summary(data)

    # Step 2: Exploratory Data Analysis
    plot_series(data, 'Date', 'Close', "Stock Closing Prices Over Time", "Date", "Close Price", kind='line')
    plot_series(data, 'Date', 'Volume', "Trading Volume Over Time", "Date", "Volume", kind='bar')

    # Step 3: Statistical Analysis
    calculate_correlation(data)
    calculate_volatility(data)
    stationarity_test(data)
    seasonal_decompose_data(data)

    # Step 4: Predictive Modeling
    X_train, y_train, X_test, y_test, scaler = prepare_data_for_lstm(data)
    
    if X_train is None or X_test is None:
        logging.error("Insufficient data for LSTM. Exiting the process.")
    else:
        model = build_and_train_lstm(X_train, y_train, X_test, y_test)
        save_model(model, "lstm_model.h5")
        plot_predictions(model, data, X_test, scaler)
else:
    logging.error("No data available for analysis.")