import requests
import pandas as pd

API_KEY = '92OMMNY9EN7KTSQN'
BASE_URL = 'https://www.alphavantage.co/query?'

def get_stock_data(symbol, start_date=None, end_date=None):
    """
    Fetches stock data from Alpha Vantage API.
    """
    url = f"{BASE_URL}function=TIME_SERIES_DAILY&symbol={symbol}&apikey={API_KEY}"
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Error: Unable to fetch data. Status code: {response.status_code}")
        return None

    data = response.json()
    if 'Time Series (Daily)' not in data:
        print("No data found for this symbol.")
        return None

    timeseries = data['Time Series (Daily)']
    df = pd.DataFrame({
        'Date': pd.to_datetime(list(timeseries.keys())),
        'Open': [float(timeseries[date]['1. open']) for date in timeseries],
        'High': [float(timeseries[date]['2. high']) for date in timeseries],
        'Low': [float(timeseries[date]['3. low']) for date in timeseries],
        'Close': [float(timeseries[date]['4. close']) for date in timeseries],
        'Volume': [int(timeseries[date]['5. volume']) for date in timeseries]
    }).sort_values('Date')

    if start_date:
        df = df[df['Date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['Date'] <= pd.to_datetime(end_date)]

    return df if not df.empty else None