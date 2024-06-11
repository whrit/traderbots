
#pip install pandas yfinance ipywidgets

import yfinance as yf
import pandas as pd
import talib

"""Load the Data"""

def get_stock_data(symbol):
    stock = yf.Ticker(symbol)
    hist = stock.history(period="1mo", interval="15m")
    return hist

symbol = "SPY"
stock_data = get_stock_data(symbol)
print(stock_data.tail())

"""Placing Orders"""

import ipywidgets as widgets
from IPython.display import display

stock_symbol = widgets.Text(
    value='SPY',
    description='Stock Symbol:',
    disabled=False
)

quantity = widgets.IntText(
    value=1,
    description='Quantity:',
    disabled=False
)

order_type = widgets.Dropdown(
    options=['Market', 'Limit'],
    value='Market',
    description='Order Type:',
    disabled=False,
)

product_type = widgets.Dropdown(
    options=['Delivery', 'Intraday'],
    value='Delivery',
    description='Product Type:',
    disabled=False,
)

limit_price = widgets.FloatText(
    value=0.0,
    description='Limit Price:',
    disabled=True,
)

stop_loss = widgets.FloatText(
    value=0.0,
    description='Stop Loss:',
    disabled=False,
)

target_price = widgets.FloatText(
    value=0.0,
    description='Target Price:',
    disabled=False,
)

def update_limit_price(*args):
    limit_price.disabled = order_type.value != 'Limit'

order_type.observe(update_limit_price, 'value')

display(stock_symbol, product_type, quantity, order_type, limit_price, stop_loss, target_price)

"""Order Execution Details"""

class MockTradingBot:
    def __init__(self):
        self.orders = []

    def place_order(self, symbol, quantity, order_type, limit_price, stop_loss, target_price):
        order = {
            'symbol': symbol,
            'quantity': quantity,
            'order_type': order_type,
            'limit_price': limit_price,
            'stop_loss': stop_loss,
            'target_price': target_price,
            'status': 'open'
        }
        self.orders.append(order)
        return order

    def check_orders(self, current_price):
        for order in self.orders:
            if order['status'] == 'open':
                if order['order_type'] == 'Market':
                    order['status'] = 'executed'
                    order['execution_price'] = current_price
                elif order['order_type'] == 'Limit' and current_price <= order['limit_price']:
                    order['status'] = 'executed'
                    order['execution_price'] = current_price

                if 'execution_price' in order:
                    if current_price <= order['stop_loss']:
                        order['status'] = 'stopped out'
                    elif current_price >= order['target_price']:
                        order['status'] = 'target hit'

    def get_orders(self):
        return pd.DataFrame(self.orders)

bot = MockTradingBot()
order = bot.place_order(stock_symbol.value, quantity.value, order_type.value, limit_price.value, stop_loss.value, target_price.value)
print("Order placed:", order)

current_price = stock_data['Close'].iloc[-1]
bot.check_orders(current_price)

orders_df = bot.get_orders()
print(orders_df)

#!pip install ta matplotlib scikit-learn xgboost

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np

def add_indicators(df):
    df['SMA_15'] = talib.SMA(df['Close'], timeperiod=15)
    df['EMA_15'] = talib.EMA(df['Close'], timeperiod=15)
    df['SMA_10'] = talib.SMA(df['Close'], timeperiod=10)
    df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
    df['EMA_20'] = talib.EMA(df['Close'], timeperiod=20)
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    df['STOCH_K'], df['STOCH_D'] = talib.STOCH(df['High'], df['Low'], df['Close'], fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    df['MACD'], df['MACDSIGNAL'], df['MACDHIST'] = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['OBV'] = talib.OBV(df['Close'], df['Volume'])
    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['BBANDS_UPPER'], df['BBANDS_MIDDLE'], df['BBANDS_LOWER'] = talib.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['MOM'] = talib.MOM(df['Close'], timeperiod=10)
    df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['WILLR'] = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['TSF'] = talib.TSF(df['Close'], timeperiod=14)
    df['TRIX'] = talib.TRIX(df['Close'], timeperiod=30)
    df['ULTOSC'] = talib.ULTOSC(df['High'], df['Low'], df['Close'], timeperiod1=7, timeperiod2=14, timeperiod3=28)
    df['ROC'] = talib.ROC(df['Close'], timeperiod=10)
    df['PLUS_DI'] = talib.PLUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['MINUS_DI'] = talib.MINUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['PLUS_DM'] = talib.PLUS_DM(df['High'], df['Low'], timeperiod=14)
    df['MINUS_DM'] = talib.MINUS_DM(df['High'], df['Low'], timeperiod=14)
    return df

stock_data = add_indicators(stock_data)
print(stock_data.tail())

def prepare_data(df, window_size=10):
    df = df.dropna()
    X = []
    y = []

    feature_columns = [
        'SMA_15', 'EMA_15', 'SMA_10', 'SMA_50', 'EMA_20', 'RSI',
        'STOCH_K', 'STOCH_D', 'MACD', 'MACDSIGNAL', 'MACDHIST',
        'ADX', 'OBV', 'ATR', 'BBANDS_UPPER', 'BBANDS_MIDDLE', 'BBANDS_LOWER',
        'MOM', 'CCI', 'WILLR', 'TSF', 'TRIX', 'ULTOSC', 'ROC',
        'PLUS_DI', 'MINUS_DI', 'PLUS_DM', 'MINUS_DM'
    ]

    for i in range(window_size, len(df)):
        X.append(df[feature_columns].iloc[i-window_size:i].values)
        y.append(df['Close'].iloc[i])

    X = np.array(X)
    y = np.array(y)

    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = prepare_data(stock_data)

def plot_stock_data(df, symbol):
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Close'], label='Close Price')
    plt.plot(df.index, df['SMA_15'], label='SMA (15)')
    plt.plot(df.index, df['EMA_15'], label='EMA (15)')
    plt.plot(df.index, df['SMA_10'], label='SMA (10)')
    plt.plot(df.index, df['SMA_50'], label='SMA (50)')
    plt.plot(df.index, df['BBANDS_UPPER'], label='Bollinger Upper Band')
    plt.plot(df.index, df['BBANDS_MIDDLE'], label='Bollinger Middle Band')
    plt.plot(df.index, df['BBANDS_LOWER'], label='Bollinger Lower Band')
    plt.plot(df.index, df['RSI'], label='RSI')
    plt.plot(df.index, df['MACD'], label='MACD')
    plt.plot(df.index, df['ROC'], label='ROC')
    plt.plot(df.index, df['STOCH_K'], label='STOCH_K')
    plt.plot(df.index, df['STOCH_D'], label='STOCH_D')
    plt.plot(df.index, df['ADX'], label='ADX')
    plt.title(f'{symbol} Price with Indicators')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

plot_stock_data(stock_data, symbol)

def train_and_predict(X_train, y_train, X_test):
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return model, predictions

model, predictions = train_and_predict(X_train, y_train, X_test)
print(predictions)

def predict_next(model, df, window_size=10):
    latest_data = df[[
        'SMA_15', 'EMA_15', 'SMA_10', 'SMA_50', 'EMA_20', 'RSI',
        'STOCH_K', 'STOCH_D', 'MACD', 'MACDSIGNAL', 'MACDHIST',
        'ADX', 'OBV', 'ATR', 'BBANDS_UPPER', 'BBANDS_MIDDLE', 'BBANDS_LOWER',
        'MOM', 'CCI', 'WILLR', 'TSF', 'TRIX', 'ULTOSC', 'ROC',
        'PLUS_DI', 'MINUS_DI', 'PLUS_DM', 'MINUS_DM'
    ]].tail(window_size).values
    latest_data = latest_data.reshape(1, -1)
    next_prediction = model.predict(latest_data)
    return next_prediction

next_prediction = predict_next(model, stock_data)
print(f'Next predicted price: {next_prediction[0]}')

def plot_predictions(y_test, predictions, next_prediction, stock_data):
    plt.figure(figsize=(14, 7))
    plt.plot(stock_data.index[-len(y_test):], y_test, label='Actual Price')
    plt.plot(stock_data.index[-len(y_test):], predictions, label='Predicted Price', linestyle='dashed')
    plt.axhline(y=next_prediction, color='r', linestyle='--', label='Next Predicted Price')
    plt.title('Actual vs Predicted Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

plot_predictions(y_test, predictions, next_prediction, stock_data)

"""Random Forest Regressor"""

from sklearn.ensemble import RandomForestRegressor

def add_indicators(df):
    df['SMA_15'] = talib.SMA(df['Close'], timeperiod=15)
    df['EMA_15'] = talib.EMA(df['Close'], timeperiod=15)
    df['SMA_10'] = talib.SMA(df['Close'], timeperiod=10)
    df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
    df['EMA_20'] = talib.EMA(df['Close'], timeperiod=20)
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    df['STOCH_K'], df['STOCH_D'] = talib.STOCH(df['High'], df['Low'], df['Close'], fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    df['MACD'], df['MACDSIGNAL'], df['MACDHIST'] = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['OBV'] = talib.OBV(df['Close'], df['Volume'])
    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['BBANDS_UPPER'], df['BBANDS_MIDDLE'], df['BBANDS_LOWER'] = talib.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['MOM'] = talib.MOM(df['Close'], timeperiod=10)
    df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['WILLR'] = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['TSF'] = talib.TSF(df['Close'], timeperiod=14)
    df['TRIX'] = talib.TRIX(df['Close'], timeperiod=30)
    df['ULTOSC'] = talib.ULTOSC(df['High'], df['Low'], df['Close'], timeperiod1=7, timeperiod2=14, timeperiod3=28)
    df['ROC'] = talib.ROC(df['Close'], timeperiod=10)
    df['PLUS_DI'] = talib.PLUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['MINUS_DI'] = talib.MINUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['PLUS_DM'] = talib.PLUS_DM(df['High'], df['Low'], timeperiod=14)
    df['MINUS_DM'] = talib.MINUS_DM(df['High'], df['Low'], timeperiod=14)
    return df

stock_data = add_indicators(stock_data)
print(stock_data.tail())

def prepare_data(df, window_size=10):
    df = df.dropna()
    X = []
    y = []

    feature_columns = [
        'SMA_15', 'EMA_15', 'SMA_10', 'SMA_50', 'EMA_20', 'RSI',
        'STOCH_K', 'STOCH_D', 'MACD', 'MACDSIGNAL', 'MACDHIST',
        'ADX', 'OBV', 'ATR', 'BBANDS_UPPER', 'BBANDS_MIDDLE', 'BBANDS_LOWER',
        'MOM', 'CCI', 'WILLR', 'TSF', 'TRIX', 'ULTOSC', 'ROC',
        'PLUS_DI', 'MINUS_DI', 'PLUS_DM', 'MINUS_DM'
    ]

    for i in range(window_size, len(df)):
        X.append(df[feature_columns].iloc[i-window_size:i].values)
        y.append(df['Close'].iloc[i])

    X = np.array(X)
    y = np.array(y)

    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = prepare_data(stock_data)

def train_and_predict(X_train, y_train, X_test):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return model, predictions

model, predictions = train_and_predict(X_train, y_train, X_test)
print(predictions)

def predict_next(model, df, window_size=10):
    latest_data = df[[
        'SMA_15', 'EMA_15', 'SMA_10', 'SMA_50', 'EMA_20', 'RSI',
        'STOCH_K', 'STOCH_D', 'MACD', 'MACDSIGNAL', 'MACDHIST',
        'ADX', 'OBV', 'ATR', 'BBANDS_UPPER', 'BBANDS_MIDDLE', 'BBANDS_LOWER',
        'MOM', 'CCI', 'WILLR', 'TSF', 'TRIX', 'ULTOSC', 'ROC',
        'PLUS_DI', 'MINUS_DI', 'PLUS_DM', 'MINUS_DM'
    ]].tail(window_size).values
    latest_data = latest_data.reshape(1, -1)
    next_prediction = model.predict(latest_data)
    return next_prediction

next_prediction = predict_next(model, stock_data)
print(f'Next predicted price: {next_prediction[0]}')

def plot_predictions(y_test, predictions, next_prediction, stock_data):
    plt.figure(figsize=(14, 7))
    plt.plot(stock_data.index[-len(y_test):], y_test, label='Actual Price')
    plt.plot(stock_data.index[-len(y_test):], predictions, label='Predicted Price', linestyle='dashed')
    plt.axhline(y=next_prediction, color='r', linestyle='--', label='Next Predicted Price')
    plt.title('Actual vs Predicted Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

plot_predictions(y_test, predictions, next_prediction, stock_data)

"""SVR Model"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

def add_indicators(df):
    df['SMA_15'] = talib.SMA(df['Close'], timeperiod=15)
    df['EMA_15'] = talib.EMA(df['Close'], timeperiod=15)
    df['SMA_10'] = talib.SMA(df['Close'], timeperiod=10)
    df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
    df['EMA_20'] = talib.EMA(df['Close'], timeperiod=20)
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    df['STOCH_K'], df['STOCH_D'] = talib.STOCH(df['High'], df['Low'], df['Close'], fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    df['MACD'], df['MACDSIGNAL'], df['MACDHIST'] = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['OBV'] = talib.OBV(df['Close'], df['Volume'])
    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['BBANDS_UPPER'], df['BBANDS_MIDDLE'], df['BBANDS_LOWER'] = talib.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['MOM'] = talib.MOM(df['Close'], timeperiod=10)
    df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['WILLR'] = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['TSF'] = talib.TSF(df['Close'], timeperiod=14)
    df['TRIX'] = talib.TRIX(df['Close'], timeperiod=30)
    df['ULTOSC'] = talib.ULTOSC(df['High'], df['Low'], df['Close'], timeperiod1=7, timeperiod2=14, timeperiod3=28)
    df['ROC'] = talib.ROC(df['Close'], timeperiod=10)
    df['PLUS_DI'] = talib.PLUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['MINUS_DI'] = talib.MINUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['PLUS_DM'] = talib.PLUS_DM(df['High'], df['Low'], timeperiod=14)
    df['MINUS_DM'] = talib.MINUS_DM(df['High'], df['Low'], timeperiod=14)
    return df

def prepare_data(df, window_size=10):
    df = df.dropna()
    X = []
    y = []

    feature_columns = [
        'SMA_15', 'EMA_15', 'SMA_10', 'SMA_50', 'EMA_20', 'RSI',
        'STOCH_K', 'STOCH_D', 'MACD', 'MACDSIGNAL', 'MACDHIST',
        'ADX', 'OBV', 'ATR', 'BBANDS_UPPER', 'BBANDS_MIDDLE', 'BBANDS_LOWER',
        'MOM', 'CCI', 'WILLR', 'TSF', 'TRIX', 'ULTOSC', 'ROC',
        'PLUS_DI', 'MINUS_DI', 'PLUS_DM', 'MINUS_DM'
    ]

    for i in range(window_size, len(df)):
        X.append(df[feature_columns].iloc[i-window_size:i].values)
        y.append(df['Close'].iloc[i])

    X = np.array(X)
    y = np.array(y)

    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_predict(X_train, y_train, X_test):
    model = SVR(kernel='rbf')
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return model, predictions

def predict_next(model, df, window_size=10):
    latest_data = df[[
        'SMA_15', 'EMA_15', 'SMA_10', 'SMA_50', 'EMA_20', 'RSI',
        'STOCH_K', 'STOCH_D', 'MACD', 'MACDSIGNAL', 'MACDHIST',
        'ADX', 'OBV', 'ATR', 'BBANDS_UPPER', 'BBANDS_MIDDLE', 'BBANDS_LOWER',
        'MOM', 'CCI', 'WILLR', 'TSF', 'TRIX', 'ULTOSC', 'ROC',
        'PLUS_DI', 'MINUS_DI', 'PLUS_DM', 'MINUS_DM'
    ]].tail(window_size).values
    latest_data = latest_data.reshape(1, -1)
    next_prediction = model.predict(latest_data)
    return next_prediction

def plot_predictions(y_test, predictions, next_prediction, stock_data):
    plt.figure(figsize=(14, 7))
    plt.plot(stock_data.index[-len(y_test):], y_test, label='Actual Price')
    plt.plot(stock_data.index[-len(y_test):], predictions, label='Predicted Price', linestyle='dashed')
    plt.axhline(y=next_prediction, color='r', linestyle='--', label='Next Predicted Price')
    plt.title('Actual vs Predicted Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

stock_data = get_stock_data(stock_symbol.value)
stock_data = add_indicators(stock_data)

X_train, X_test, y_train, y_test = prepare_data(stock_data)

next_prediction = predict_next(model, stock_data)
print(f'Next predicted price: {next_prediction[0]}')

plot_predictions(y_test, predictions, next_prediction, stock_data)

"""CNN"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

def add_indicators(df):
    df['SMA_15'] = talib.SMA(df['Close'], timeperiod=15)
    df['EMA_15'] = talib.EMA(df['Close'], timeperiod=15)
    df['SMA_10'] = talib.SMA(df['Close'], timeperiod=10)
    df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
    df['EMA_20'] = talib.EMA(df['Close'], timeperiod=20)
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    df['STOCH_K'], df['STOCH_D'] = talib.STOCH(df['High'], df['Low'], df['Close'], fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    df['MACD'], df['MACDSIGNAL'], df['MACDHIST'] = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['OBV'] = talib.OBV(df['Close'], df['Volume'])
    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['BBANDS_UPPER'], df['BBANDS_MIDDLE'], df['BBANDS_LOWER'] = talib.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['MOM'] = talib.MOM(df['Close'], timeperiod=10)
    df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['WILLR'] = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['TSF'] = talib.TSF(df['Close'], timeperiod=14)
    df['TRIX'] = talib.TRIX(df['Close'], timeperiod=30)
    df['ULTOSC'] = talib.ULTOSC(df['High'], df['Low'], df['Close'], timeperiod1=7, timeperiod2=14, timeperiod3=28)
    df['ROC'] = talib.ROC(df['Close'], timeperiod=10)
    df['PLUS_DI'] = talib.PLUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['MINUS_DI'] = talib.MINUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['PLUS_DM'] = talib.PLUS_DM(df['High'], df['Low'], timeperiod=14)
    df['MINUS_DM'] = talib.MINUS_DM(df['High'], df['Low'], timeperiod=14)
    return df

def prepare_data(df, window_size=10):
    df = df.dropna()
    X = []
    y = []

    feature_columns = [
        'SMA_15', 'EMA_15', 'SMA_10', 'SMA_50', 'EMA_20', 'RSI',
        'STOCH_K', 'STOCH_D', 'MACD', 'MACDSIGNAL', 'MACDHIST',
        'ADX', 'OBV', 'ATR', 'BBANDS_UPPER', 'BBANDS_MIDDLE', 'BBANDS_LOWER',
        'MOM', 'CCI', 'WILLR', 'TSF', 'TRIX', 'ULTOSC', 'ROC',
        'PLUS_DI', 'MINUS_DI', 'PLUS_DM', 'MINUS_DM'
    ]

    for i in range(window_size, len(df)):
        X.append(df[feature_columns].iloc[i-window_size:i].values)
        y.append(df['Close'].iloc[i])

    X = np.array(X)
    y = np.array(y)

    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_predict(X_train, y_train, X_test):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

    predictions = model.predict(X_test)
    return model, predictions

def predict_next(model, df, window_size=10):
    latest_data = df[[
        'SMA_15', 'EMA_15', 'SMA_10', 'SMA_50', 'EMA_20', 'RSI',
        'STOCH_K', 'STOCH_D', 'MACD', 'MACDSIGNAL', 'MACDHIST',
        'ADX', 'OBV', 'ATR', 'BBANDS_UPPER', 'BBANDS_MIDDLE', 'BBANDS_LOWER',
        'MOM', 'CCI', 'WILLR', 'TSF', 'TRIX', 'ULTOSC', 'ROC',
        'PLUS_DI', 'MINUS_DI', 'PLUS_DM', 'MINUS_DM'
    ]].tail(window_size).values
    latest_data = latest_data.reshape(1, -1)
    next_prediction = model.predict(latest_data)
    return next_prediction

def plot_predictions(y_test, predictions, next_prediction, stock_data):
    plt.figure(figsize=(14, 7))
    plt.plot(stock_data.index[-len(y_test):], y_test, label='Actual Price')
    plt.plot(stock_data.index[-len(y_test):], predictions, label='Predicted Price', linestyle='dashed')
    plt.axhline(y=next_prediction, color='r', linestyle='--', label='Next Predicted Price')
    plt.title('Actual vs Predicted Price with CNN')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

stock_data = get_stock_data(stock_symbol.value)
stock_data = add_indicators(stock_data)

X_train, X_test, y_train, y_test = prepare_data(stock_data)

model, predictions = train_and_predict(X_train, y_train, X_test)

next_prediction = predict_next(model, stock_data)
print(f'Next predicted price: {next_prediction[0]}')

plot_predictions(y_test, predictions, next_prediction, stock_data)

"""ARIMA Model"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

def add_indicators(df):
    df['SMA_15'] = talib.SMA(df['Close'], timeperiod=15)
    df['EMA_15'] = talib.EMA(df['Close'], timeperiod=15)
    df['SMA_10'] = talib.SMA(df['Close'], timeperiod=10)
    df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
    df['EMA_20'] = talib.EMA(df['Close'], timeperiod=20)
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    df['STOCH_K'], df['STOCH_D'] = talib.STOCH(df['High'], df['Low'], df['Close'], fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    df['MACD'], df['MACDSIGNAL'], df['MACDHIST'] = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['OBV'] = talib.OBV(df['Close'], df['Volume'])
    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['BBANDS_UPPER'], df['BBANDS_MIDDLE'], df['BBANDS_LOWER'] = talib.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['MOM'] = talib.MOM(df['Close'], timeperiod=10)
    df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['WILLR'] = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['TSF'] = talib.TSF(df['Close'], timeperiod=14)
    df['TRIX'] = talib.TRIX(df['Close'], timeperiod=30)
    df['ULTOSC'] = talib.ULTOSC(df['High'], df['Low'], df['Close'], timeperiod1=7, timeperiod2=14, timeperiod3=28)
    df['ROC'] = talib.ROC(df['Close'], timeperiod=10)
    df['PLUS_DI'] = talib.PLUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['MINUS_DI'] = talib.MINUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['PLUS_DM'] = talib.PLUS_DM(df['High'], df['Low'], timeperiod=14)
    df['MINUS_DM'] = talib.MINUS_DM(df['High'], df['Low'], timeperiod=14)
    return df

def prepare_data_arima(df):
    df = df.dropna()
    return df['Close']

def train_and_predict_arima(data, order=(5, 1, 0)):
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    start_index = len(data) - int(len(data) * 0.2)
    end_index = len(data) - 1
    predictions = model_fit.predict(start=start_index, end=end_index, typ='levels')
    return model_fit, predictions

def predict_next_arima(model_fit, data):
    next_prediction = model_fit.forecast(steps=1, exog=data.iloc[[-1]])[0]
    return next_prediction

def plot_predictions_arima(data, predictions, next_prediction):
    plt.figure(figsize=(14, 7))
    plt.plot(data.index[-len(predictions):], data.values[-len(predictions):], label='Actual Price')
    plt.plot(data.index[-len(predictions):], predictions, label='Predicted Price', linestyle='dashed')
    plt.axhline(y=next_prediction, color='r', linestyle='--', label='Next Predicted Price')
    plt.title('Actual vs Predicted Price with ARIMA')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

stock_data = get_stock_data(stock_symbol.value)
stock_data = add_indicators(stock_data)

data = prepare_data_arima(stock_data)
data.index = pd.date_range(start='2024-01-01', periods=len(data), freq='D')

model_fit, predictions = train_and_predict_arima(data)

next_prediction = predict_next_arima(model_fit, data)
print(f'Next predicted price: {next_prediction}')

plot_predictions_arima(data, predictions, next_prediction)

"""Transformer Series Model"""

#pip install pandas yfinance matplotlib scikit-learn torch ta

import yfinance as yf
import pandas as pd

def get_stock_data(symbol):
    stock = yf.Ticker(symbol)
    hist = stock.history(period="1mo", interval="15m")
    return hist

def add_indicators(df):
    df['SMA_15'] = talib.SMA(df['Close'], timeperiod=15)
    df['EMA_15'] = talib.EMA(df['Close'], timeperiod=15)
    df['SMA_10'] = talib.SMA(df['Close'], timeperiod=10)
    df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
    df['EMA_20'] = talib.EMA(df['Close'], timeperiod=20)
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    df['STOCH_K'], df['STOCH_D'] = talib.STOCH(df['High'], df['Low'], df['Close'], fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    df['MACD'], df['MACDSIGNAL'], df['MACDHIST'] = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['OBV'] = talib.OBV(df['Close'], df['Volume'])
    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['BBANDS_UPPER'], df['BBANDS_MIDDLE'], df['BBANDS_LOWER'] = talib.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['MOM'] = talib.MOM(df['Close'], timeperiod=10)
    df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['WILLR'] = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['TSF'] = talib.TSF(df['Close'], timeperiod=14)
    df['TRIX'] = talib.TRIX(df['Close'], timeperiod=30)
    df['ULTOSC'] = talib.ULTOSC(df['High'], df['Low'], df['Close'], timeperiod1=7, timeperiod2=14, timeperiod3=28)
    df['ROC'] = talib.ROC(df['Close'], timeperiod=10)
    df['PLUS_DI'] = talib.PLUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['MINUS_DI'] = talib.MINUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['PLUS_DM'] = talib.PLUS_DM(df['High'], df['Low'], timeperiod=14)
    df['MINUS_DM'] = talib.MINUS_DM(df['High'], df['Low'], timeperiod=14)
    return df

symbol = "SPY"
stock_data = get_stock_data(symbol)
stock_data = add_indicators(stock_data)

print(stock_data.tail())

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class TimeSeriesDataset(Dataset):
    def __init__(self, data, window_size=10):
        self.data = data
        self.window_size = window_size
        self.scaler = MinMaxScaler()
        self.data_scaled = self.scaler.fit_transform(data.values.reshape(-1, 1)).flatten()

    def __len__(self):
        return len(self.data_scaled) - self.window_size

    def __getitem__(self, idx):
        idx_end = idx + self.window_size
        x = self.data_scaled[idx:idx_end]
        y = self.data_scaled[idx_end]
        return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)

window_size = 10

dataset = TimeSeriesDataset(stock_data['Close'], window_size=window_size)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

import torch
import torch.nn as nn

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, num_layers, num_heads, hidden_size, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        self.embedding = nn.Linear(input_size, hidden_size)

        self.transformer_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout),
            num_layers=num_layers
        )

        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.embedding(x)

        x = self.transformer_layers(x)

        x = self.output_layer(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.unsqueeze(1))
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs.unsqueeze(1))
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item() * inputs.size(0)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader.dataset)}, Val Loss: {val_loss/len(val_loader.dataset)}')

input_size = window_size
num_layers = 2
num_heads = 2
hidden_size = 64
dropout = 0.1
learning_rate = 0.001
num_epochs = 10

model = TimeSeriesTransformer(input_size, num_layers, num_heads, hidden_size, dropout=dropout)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs)

def predict_next(model, data, window_size=10):
    model.eval()
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.values.reshape(-1, 1)).flatten()
    with torch.no_grad():
        inputs = torch.tensor(data_scaled[-window_size:], dtype=torch.float).unsqueeze(0).unsqueeze(1)
        prediction = model(inputs)
    return scaler.inverse_transform(prediction.squeeze().numpy().reshape(-1, 1))

# Make predictions
next_prediction_ts = predict_next(model, stock_data['Close'])

# Print the next predicted price
print(f'Next predicted price: {next_prediction_ts[0][-1]}')

def plot_predictions_transformer(data, predictions, next_prediction):
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data.values, label='Actual Price')

    # Plot predicted prices
    for i in range(len(predictions)):
        plt.plot(data.index[-len(predictions) + i:], predictions[i], linestyle='dashed', label='Predicted Price' if i == 0 else None)

    # Plot next predicted price
    plt.axhline(y=next_prediction, color='r', linestyle='--', label='Next Predicted Price')

    plt.title('Actual vs Predicted Price with Transformer Model')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
plot_predictions_transformer(stock_data['Close'], next_prediction_ts.flatten(), next_prediction_ts[-1][-1])