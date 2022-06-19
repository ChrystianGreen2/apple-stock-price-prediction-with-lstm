import argparse
import numpy as np
import yfinance as yf

from datetime import date, timedelta

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM


def load_data():
    today = date.today()
    d1 = today.strftime("%Y-%m-%d")
    end_date = d1
    d2 = date.today() - timedelta(days=10000)
    d2 = d2.strftime("%Y-%m-%d")
    start_date = d2
    data = yf.download('AAPL', 
                        start=start_date, 
                        end=end_date, 
                        progress=False)
    return data


def preprocess_data(data):
    data = data.dropna()
    data["Date"] = data.index
    data = data[["Date", "Open", "High", "Low", "Close", 
             "Adj Close", "Volume"]]
    data.reset_index(drop=True, inplace=True)

    x = data[["Open", "High", "Low", "Volume"]]
    y = data["Close"]
    x = x.to_numpy()
    y = y.to_numpy()
    y = y.reshape(-1, 1)
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
    return xtrain, ytrain, xtest, ytest
    


def build_lstm(xtrain, ytrain, xtest, ytest, epochs=100, batch_size=32):
    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True, input_shape=(xtrain.shape[1], 1)))
    model.add(LSTM(units=64, return_sequences=False))
    model.add(Dense(units=16, activation="relu"))
    model.add(Dense(units=1))

    model.compile(loss="mean_squared_error", optimizer="adam")
    model.fit(xtrain, ytrain, epochs=epochs, batch_size=batch_size)
    print(f"test loss: {model.evaluate(xtest, ytest)}")
    return model

def save_model(model):
    model.save("model.h5")

def predict(model, xtest):
    return model.predict(xtest)

def main(args):
    data = load_data()
    xtrain, ytrain, xtest, ytest = preprocess_data(data)
    model = build_lstm(xtrain, ytrain, xtest, ytest, args.epochs, args.batch_size)
    save_model(model)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=5)
    
    args=parser.parse_args()
    
    main(args)
