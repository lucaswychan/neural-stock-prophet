import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Union

import numpy as np

from .arima import arima_forecast
from .dataset import TimeSeriesDataset
from .decompose import multiplicative_decompose
from .model import AttentionLSTM
from .portfolio import BasePortfolio
from .utils import dict_to_matrix, get_prediction_trend, visualize_results

warnings.simplefilter(action="ignore", category=FutureWarning)

__all__ = ["NeuralStockProphet"]


class NSPValidator:
    @staticmethod
    def validate_stock_names(stock_names: List[str]) -> List[str]:
        if len(stock_names) < 2:
            raise ValueError("At least 2 stock names are required")
        return stock_names

    @staticmethod
    def validate_date_range(
        start_date: Union[str, datetime], end_date: Union[str, datetime], date_type: str
    ) -> Tuple[datetime, datetime]:
        start = start_date if isinstance(start_date, datetime) else datetime.strptime(start_date, "%Y-%m-%d")
        end = end_date if isinstance(end_date, datetime) else datetime.strptime(end_date, "%Y-%m-%d")
        if start >= end:
            raise ValueError(f"Invalid {date_type} date range")
        return start, end

    @staticmethod
    def validate_range(value: float, min_val: float, max_val: float, param_name: str) -> float:
        if not min_val <= value <= max_val:
            raise ValueError(f"{param_name} should be in [{min_val}, {max_val}]")
        return value

    @staticmethod
    def validate_positive_int(value: int, param_name: str) -> int:
        if value <= 0:
            raise ValueError(f"{param_name} should be a positive integer")
        return value

    @staticmethod
    def validate_positive_float(value: float, param_name: str) -> float:
        if value <= 0:
            raise ValueError(f"{param_name} should be larger than 0")
        return value

    @staticmethod
    def validate_arima_order(arima_order: Tuple[int, int, int]) -> Tuple[int, int, int]:
        if len(arima_order) != 3:
            raise ValueError("arima_order should have 3 integers")
        return arima_order

    @staticmethod
    def validate_arima_trend(arima_trend: str) -> str:
        valid_trends = ["n", "c", "t", "ct"]
        if arima_trend not in valid_trends:
            raise ValueError(f"Invalid arima_trend. Must be one of {valid_trends}")
        return arima_trend


class NeuralStockProphet:
    def __init__(
        self,
        stock_names: List[str],
        scaler_func: callable,
        train_start_date: Union[str, datetime],
        train_end_date: Union[str, datetime],
        test_start_date: Union[str, datetime],
        test_end_date: Union[str, datetime],
        keep_ratio: float = 0.8,
        time_steps: int = 60,
        window_length: int = 48,
        factor: float = 0.9,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        arima_order: Tuple[int, int, int] = (1, 0, 6),
        arima_trend: str = "ct",
    ):
        validator = NSPValidator()

        # Dataset parameters
        self.stock_names = validator.validate_stock_names(stock_names)
        self.scaler_func = scaler_func
        self.train_start_date, self.train_end_date = validator.validate_date_range(
            train_start_date, train_end_date, "training"
        )
        self.test_start_date, self.test_end_date = validator.validate_date_range(
            test_start_date, test_end_date, "testing"
        )
        self.keep_ratio = validator.validate_range(keep_ratio, 0, 1, "keep_ratio")
        self.time_steps = validator.validate_positive_int(time_steps, "time_steps")

        # Multiplicative decomposition parameter
        self.window_length = validator.validate_positive_int(window_length, "window_length")

        # LSTM model parameters
        self.epochs = validator.validate_positive_int(epochs, "epochs")
        self.batch_size = validator.validate_positive_int(batch_size, "batch_size")
        self.lr = validator.validate_positive_float(lr, "lr")

        # ARIMA parameters
        self.arima_order = validator.validate_arima_order(arima_order)
        self.arima_trend = validator.validate_arima_trend(arima_trend)

        # Combine model parameter
        self.factor = validator.validate_range(factor, 0, 1, "factor")

        self.forecasts = None
        self.real_vals = None
        self.train_data = {}
        self.test_data = {}
        self.model = None
        self.n_features = None

    def load_data(self) -> None:
        for stock_name in self.stock_names:
            self.train_data[stock_name] = TimeSeriesDataset(
                stock_name,
                self.time_steps,
                self.scaler_func(),
                self.train_start_date,
                self.train_end_date,
                self.keep_ratio,
            )
            self.test_data[stock_name] = TimeSeriesDataset(
                stock_name,
                self.time_steps,
                self.scaler_func(),
                self.test_start_date,
                self.test_end_date,
                self.keep_ratio,
            )
        self.n_features = self.train_data[self.stock_names[0]].n_features

    def _forecast_stock(self, stock_name: str, verbose: bool) -> Tuple[np.ndarray, np.ndarray]:
        if verbose:
            separator = "=" * 15
            print(f"\n{separator}Forecasting for {stock_name}{separator}")

        train_data = self.train_data[stock_name]
        test_data = self.test_data[stock_name]

        # Fit the model
        self.model.fit(
            train_data.X,
            train_data.y.ravel(),
            epochs=self.epochs,
            verbose=2 if verbose else 0,
        )

        lstm_trend = get_prediction_trend(self.model, test_data)
        y_true = test_data.scaler.inverse_transform(test_data.y.reshape(-1, 1))

        if verbose:
            self._visualize_prediction(
                test_data, y_true, lstm_trend, "LSTM Stock Price Prediction", f"{stock_name}_lstm"
            )

        # Perform multiplicative decomposition
        data = train_data.labels
        trend, seasonal, residual = multiplicative_decompose(data, self.window_length)

        # Get the complete LSTM prediction
        lstm_signal = lstm_trend.reshape(-1) * seasonal[-len(lstm_trend) :] * 1

        if verbose:
            self._visualize_prediction(
                test_data, y_true, lstm_signal, "Adding Seasonal Component", f"{stock_name}_lstm_seasonal"
            )

        # ARIMA model
        arima_trend = arima_forecast(
            train_data.labels.values,
            len(y_true),
            config=(self.arima_order, self.arima_trend),
        )

        if verbose:
            self._visualize_prediction(
                test_data, y_true, arima_trend, "ARIMA Stock Price Prediction", f"{stock_name}_arima"
            )

        arima_signal = arima_trend.reshape(-1) * seasonal[-len(lstm_trend) :] * 1

        # Combine the LSTM and ARIMA forecast
        weighted_signal = self.factor * lstm_signal + (1 - self.factor) * arima_signal

        if verbose:
            self._visualize_prediction(
                test_data, y_true, weighted_signal.reshape((-1, 1)), "Combined Model", f"{stock_name}_combine"
            )

        return weighted_signal, y_true.reshape(-1)

    def _visualize_prediction(self, test_data, y_true, prediction, title, filename):
        visualize_results(
            test_data.df.index[test_data.time_steps :],
            y_true,
            prediction,
            title,
            filename,
        )

    def forecast(self, verbose=False):
        self.load_data()
        self.model = AttentionLSTM(self.time_steps, self.n_features, self.lr)

        forecasts = {}
        real_vals = {}

        for stock_name in self.stock_names:
            forecast, real_val = self._forecast_stock(stock_name, verbose)
            forecasts[stock_name] = forecast
            real_vals[stock_name] = real_val

        self.forecasts = forecasts
        self.real_vals = real_vals

        return dict_to_matrix(forecasts), dict_to_matrix(real_vals)

    def evaluate_performance(self, portfolio: BasePortfolio, **kwargs) -> None:
        return portfolio.evaluate(prices=self.forecasts, **kwargs)
