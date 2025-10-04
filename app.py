"""
Stock Price Predictor using LSTM Neural Network

Required Libraries:
- numpy
- pandas
- matplotlib
- yfinance
- scikit-learn
- tensorflow
"""

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ---------------------------
# Function to fetch stock data
# ---------------------------
def fetch_stock_data(ticker, start="2015-01-01", end="2023-12-31"):
    data = yf.download(ticker, start=start, end=end)
    if data.empty:
        print(f"Error: No data found for ticker '{ticker}'. Please try again.")
        return None
    return data

# ---------------------------
# Function to preprocess data
# ---------------------------
def preprocess_data(data, time_step=60):
    close_data = data[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_data)

    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    def create_sequences(dataset):
        X, Y = [], []
        for i in range(len(dataset) - time_step - 1):
            X.append(dataset[i:(i + time_step), 0])
            Y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(Y)

    X_train, y_train = create_sequences(train_data)
    X_test, y_test = create_sequences(test_data)

    # Reshape for LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    return X_train, y_train, X_test, y_test, scaler, scaled_data

# ---------------------------
# Function to build LSTM model
# ---------------------------
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# ---------------------------
# Function to plot results
# ---------------------------
def plot_predictions(actual, predicted, ticker):
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual Price')
    plt.plot(predicted, label='Predicted Price')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# ---------------------------
# Function to predict next day price
# ---------------------------
def predict_next_day(model, scaled_data, scaler, time_step=60):
    last_60_days = scaled_data[-time_step:]
    X_future = last_60_days.reshape(1, time_step, 1)
    future_price = model.predict(X_future)
    future_price = scaler.inverse_transform(future_price)
    return future_price[0][0]

# ---------------------------
# Main workflow
# ---------------------------
def run_stock_predictor():
    ticker = input("Enter the stock ticker (e.g., INFY.NS, HDFCBANK.NS, AAPL): ")
    data = fetch_stock_data(ticker)
    if data is None:
        return

    print(f"\nShowing first 5 rows of {ticker}:")
    print(data.head())

    X_train, y_train, X_test, y_test, scaler, scaled_data = preprocess_data(data)

    model = build_lstm_model((X_train.shape[1], 1))

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=20, batch_size=64, verbose=1)

    # Plot training & validation loss
    plt.figure(figsize=(8,5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f"{ticker} Training vs Validation Loss")
    plt.legend()
    plt.show()

    # Predict test data
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

    plot_predictions(actual_prices, predictions, ticker)

    # Predict next day price
    next_price = predict_next_day(model, scaled_data, scaler)
    print(f"Predicted next day price for {ticker}: {next_price:.2f}")

# ---------------------------
# Run the predictor
# ---------------------------
if __name__ == "__main__":
    while True:
        run_stock_predictor()
        choice = input("\nDo you want to predict another stock? (yes/no): ").strip().lower()
        if choice != 'yes':
            print("Exiting program. Goodbye!")
            break