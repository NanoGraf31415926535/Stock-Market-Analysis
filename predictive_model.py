import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

def prepare_data_for_lstm(data, look_back=60):
    """
    Prepares data for LSTM by scaling and creating time-lagged features.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    # Split the data into training and testing sets
    train_size = int(len(data_scaled) * 0.8)
    train, test = data_scaled[:train_size], data_scaled[train_size:]

    # Create datasets
    def create_dataset(dataset, look_back):
        X, Y = [], []
        for i in range(len(dataset) - look_back - 1):
            X.append(dataset[i:(i + look_back), 0])
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)

    X_train, y_train = create_dataset(train, look_back)
    X_test, y_test = create_dataset(test, look_back)

    # Check for sufficient data
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        return None, None, None, None, None

    # Reshape data for LSTM input
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return X_train, y_train, X_test, y_test, scaler

def build_and_train_lstm(X_train, y_train, X_test, y_test):
    """
    Builds, compiles, and trains an LSTM model for stock price prediction.
    """
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    return model

def save_model(model, filename):
    """
    Saves the trained model to a file.
    """
    model.save(filename)

def plot_predictions(model, data, X_test, scaler):
    """
    Plots the true vs predicted stock prices.
    """
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    plt.figure(figsize=(14, 7))
    plt.plot(data['Date'], data['Close'], label='True Price')
    plt.plot(data['Date'][-len(predictions):], predictions, label='Predicted Price', color='red')
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()