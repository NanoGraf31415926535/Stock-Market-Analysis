import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

def calculate_correlation(data):
    correlation_matrix = data[['Open', 'Close', 'High', 'Low', 'Volume']].corr()
    sns.heatmap(correlation_matrix, annot=True)
    plt.title("Correlation Matrix")
    plt.show()

def calculate_volatility(data):
    data['Daily_Return'] = data['Close'].pct_change()
    volatility = data['Daily_Return'].std()
    print("Daily Volatility:", volatility)
    data['Daily_Return'].hist(bins=50, figsize=(10, 6))
    plt.title("Daily Return Distribution")
    plt.show()

def stationarity_test(data):
    result = adfuller(data['Close'].dropna())
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    if result[1] > 0.05:
        print("Data is non-stationary.")
    else:
        print("Data is stationary.")

def seasonal_decompose_data(data):
    data.set_index('Date', inplace=True)
    result = seasonal_decompose(data['Close'], model='multiplicative', period=30)
    result.plot()
    plt.show()
    data.reset_index(inplace=True)