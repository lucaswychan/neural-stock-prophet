import pandas as pd
import numpy as np
from dataset import TimeSeriesDataset
from model import create_attention_lstm_model
from arima import arima_forecast
from decompose import multiplicative_decompose

class NeuralStockProphet:
    def __init__(self, stock_names, time_steps, scaler, train_date, test_date, window_length, factor):
        self.stock_names = stock_names
        self.time_steps = time_steps
        self.scaler = scaler
        self.train_start_date, self.train_end_date = train_date
        self.test_start_date, self.test_end_date = test_date
        
        self.window_length = window_length
        self.factor = factor
        
        model = create_attention_lstm_model(self.time_steps, self.n_features)
        
    def load_data(self):
        self.train_data = {}
        self.test_data = {}
        for stock_name in self.stock_names:
            self.train_data[stock_name] = TimeSeriesDataset(stock_name, self.time_steps, self.scaler, self.train_start_date, self.train_end_date)
            self.test_data[stock_name] = TimeSeriesDataset(stock_name, self.time_steps, self.scaler, self.test_start_date, self.test_end_date)
        self.n_features = self.train_data[self.stock_names[0]].n_features
    
    def forecast(self, save_fig=False):
        self.load_data()
        
        forecasts = {}
        
        for stock_name in self.stock_names:
            trend, seasonal, residual = multiplicative_decompose(self.train_data[stock_name].labels, self.window_length)
            yhat = arima_forecast(residual, len(self.test_data[stock_name].labels), config=((1, 0, 6), "ct"))
            forecasts[stock_name] = trend * seasonal * yhat
        
        return forecasts
    