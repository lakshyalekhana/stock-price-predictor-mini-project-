# ---------------------------
# Stock Price Predictor (LSTM) - Streamlit Version
# ---------------------------

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

st.title("📈 Stock Price Predictor using LSTM")

# ---------------------------
# Function to fetch stock data
# ---------------------------
def fetch_stock_data(ticker, start="2015-01-01", end="2023-12-31"):
    data = yf.download(ticker, start=start, end=end)
    return data

# ---------------------------
# Function to preprocess data
# ---------------------------
def preprocess_data(data, time_step=60):
    close = data[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close)

    train_size = int(len(scaled) * 0.8)
    train = scaled[:train_size]
    test = scaled[train_size:]

    def create_sequences(ds):
        X, Y = [], []
        for i in range(len(ds) - time_step - 1):
            X.append(ds[i:(i + time_step), 0])
            Y.append(ds[i + time_step, 0])
        return np.array(X), np.array(Y)

    X_train, y_train = create_sequences(train)
    X_test, y_test = create_sequences(test)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    return X_train, y_train, X_test, y_test, scaler, scaled

# ---------------------------
# Function to build LSTM model
# ---------------------------
def build_model(input_shape):
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
# Function to plot predictions
# ---------------------------
def plot_predictions(actual, preds, ticker):
    fig, ax = plt.subplots()
    ax.plot(actual, label="Actual")
    ax.plot(preds, label="Predicted")
    ax.set_title(f"{ticker} Stock Price Prediction")
    ax.legend()
    st.pyplot(fig)

# ---------------------------
# Function to predict next day
# ---------------------------
def predict_next_day(model, scaled_data, scaler, time_step=60):
    last = scaled_data[-time_step:]
    X = last.reshape(1, time_step, 1)
    pred = model.predict(X)
    return scaler.inverse_transform(pred)[0][0]

# ---------------------------
# Streamlit UI
# ---------------------------
ticker = st.text_input("Enter Stock Ticker (e.g. INFY.NS, AAPL):")

if ticker:
    data = fetch_stock_data(ticker)
    if data is not None and not data.empty:
        st.subheader(f"Latest Data for {ticker}")
        st.dataframe(data.tail())

        # Preprocess data
        X_train, y_train, X_test, y_test, scaler, scaled_data = preprocess_data(data)

        # Train model
        with st.spinner("Training model... please wait ⏳"):
            model = build_model((X_train.shape[1], 1))
            history = model.fit(X_train, y_train,
                                validation_data=(X_test, y_test),
                                epochs=10, batch_size=64,
                                verbose=0)
        st.success("✅ Model trained successfully!")

        # Predict test data
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)
        actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Plot
        plot_predictions(actual_prices, predictions, ticker)

        # Predict next day price
        next_day_price = predict_next_day(model, scaled_data, scaler)
        st.metric("Predicted Next Day Price", f"₹{next_day_price:.2f}")

    else:
        st.error("No data found. Please check the ticker symbol.")