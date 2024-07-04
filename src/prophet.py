import warnings
from datetime import datetime
from typing import Dict

from src.arima import arima_forecast
from src.dataset import TimeSeriesDataset
from src.decompose import multiplicative_decompose
from src.model import AttentionLSTM
from src.utils import get_prediction_trend, visualize_results

warnings.simplefilter(action="ignore", category=FutureWarning)


class NeuralStockProphet:
    def __init__(
        self,
        stock_names,
        scaler_func,
        train_start_date,
        train_end_date,
        test_start_date,
        test_end_date,
        keep_ratio=0.8,
        time_steps=60,
        window_length=48,
        factor=0.9,
        epochs=50,
        batch_size=32,
        lr=1e-3,
        arima_order=(1, 0, 6),
        arima_trend="ct",
    ):
        # dataset parameters
        assert len(stock_names) >= 2, "At least 2 stock names are required"
        self.stock_names = stock_names
        self.scaler_func = scaler_func

        self.train_start_date = datetime.strptime(train_start_date, "%Y-%m-%d")
        self.train_end_date = datetime.strptime(train_end_date, "%Y-%m-%d")
        self.test_start_date = datetime.strptime(test_start_date, "%Y-%m-%d")
        self.test_end_date = datetime.strptime(test_end_date, "%Y-%m-%d")
        assert self.train_start_date < self.train_end_date, "Invalid training date"
        assert self.test_start_date < self.test_end_date, "Invalid testing date"

        assert time_steps > 0, "time_steps should be a positive integer"
        self.time_steps = time_steps
        assert 0 <= keep_ratio <= 1, "keep_ratio should be in [0, 1]"
        self.keep_ratio = keep_ratio

        # multiplicative decomposition parameter
        assert window_length > 0, "window_length should be a positive integer"
        self.window_length = window_length

        # LSTM model parameters
        assert epochs > 0, "epochs should be larger than 0"
        assert batch_size > 0, "batch_size should be larger than 0"
        assert lr > 0, "lr should be larger than 0"
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        # ARIMA parameters
        assert len(arima_order) == 3, "arima_order should have 3 integers"
        assert arima_trend in ["n", "c", "t", "ct"], "Invalid arima_trend"
        self.arima_order = arima_order
        self.arima_trend = arima_trend

        # combine model parameter
        assert 0 <= factor <= 1, "factor should be in [0, 1]"
        self.factor = factor

    def load_data(self):
        # print("load data")
        self.train_data: Dict[str, TimeSeriesDataset] = {}
        self.test_data: Dict[str, TimeSeriesDataset] = {}

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

    def forecast(self, verbose=False):
        forecasts = {}
        real_vals = {}
        self.load_data()
        self.model = AttentionLSTM(self.time_steps, self.n_features, self.lr)

        for stock_name in self.stock_names:
            if verbose:
                print(
                    f"====================Forecasting for {stock_name}===================="
                )

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

            # visualize the prediction
            if verbose:
                visualize_results(
                    test_data.df.index[test_data.time_steps :],
                    y_true,
                    lstm_trend,
                    "LSTM Stock Price Prediction",
                    f"{stock_name}_lstm",
                )

            # perform multiplicative decomposition
            data = train_data.labels
            trend, seasonal, residual = multiplicative_decompose(
                data, self.window_length
            )

            # get the complete LSTM prediction
            lstm_signal = lstm_trend.reshape(-1) * seasonal[-len(lstm_trend) :] * 1

            if verbose:
                visualize_results(
                    test_data.df.index[test_data.time_steps :],
                    y_true,
                    lstm_signal,
                    "Adding Seasonal Component",
                    f"{stock_name}_lstm_seasonal",
                )

            # ARIMA model
            arima_trend = arima_forecast(
                train_data.labels.values,
                len(y_true),
                config=(self.arima_order, self.arima_trend),
            )

            # arima_trend = train_data.scaler.inverse_transform(arima_trend.reshape(-1, 1))

            if verbose:
                visualize_results(
                    test_data.df.index[test_data.time_steps :],
                    y_true,
                    arima_trend,
                    "ARIMA Stock Price Prediction",
                    f"{stock_name}_arima",
                )

            arima_singal = arima_trend.reshape(-1) * seasonal[-len(lstm_trend) :] * 1

            # combine the LSTM and ARIMA forecast
            weighted_signal = (
                self.factor * lstm_signal + (1 - self.factor) * arima_singal
            )

            if verbose:
                visualize_results(
                    test_data.df.index[test_data.time_steps :],
                    y_true,
                    weighted_signal.reshape((-1, 1)),
                    "Combine Model",
                    f"{stock_name}_combine",
                )

            forecasts[stock_name] = weighted_signal
            real_vals[stock_name] = y_true.reshape(-1)

            if verbose:
                print("=" * 80)

        return forecasts, real_vals
