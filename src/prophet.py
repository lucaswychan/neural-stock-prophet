import warnings
from typing import Dict

from src.arima import arima_forecast
from src.dataset import TimeSeriesDataset
from src.decompose import multiplicative_decompose
from src.model import AttentionLSTM
from src.utils import get_prediction_trend, visualize_prediction, visualize_results

warnings.simplefilter(action="ignore", category=FutureWarning)


class NeuralStockProphet:
    def __init__(
        self,
        stock_names,
        scaler,
        train_start_date,
        train_end_date,
        test_start_date,
        test_end_date,
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
        self.stock_names = stock_names
        self.scaler = scaler
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date
        self.test_start_date = test_start_date
        self.test_end_date = test_end_date
        self.time_steps = time_steps

        # multiplicative decomposition parameter
        self.window_length = window_length

        # LSTM model parameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        # ARIMA parameters
        self.arima_order = arima_order
        self.arima_trend = arima_trend

        # combine model parameter
        self.factor = factor

    def load_data(self):
        print("load data")
        self.train_data: Dict[str, TimeSeriesDataset] = {}
        self.test_data: Dict[str, TimeSeriesDataset] = {}

        for stock_name in self.stock_names:
            self.train_data[stock_name] = TimeSeriesDataset(
                stock_name,
                self.time_steps,
                self.scaler,
                self.train_start_date,
                self.train_end_date,
            )
            self.test_data[stock_name] = TimeSeriesDataset(
                stock_name,
                self.time_steps,
                self.scaler,
                self.test_start_date,
                self.test_end_date,
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
                batch_size=self.batch_size,
                verbose=verbose
            )

            # visualize the prediction
            if verbose:
                visualize_prediction(self.model, test_data, f"{stock_name}_lstm")

            # perform multiplicative decomposition
            data = train_data.labels
            trend, seasonal, residual = multiplicative_decompose(
                data, self.window_length
            )

            # get the complete LSTM prediction
            lstm_trend = get_prediction_trend(self.model, test_data)
            lstm_signal = lstm_trend.reshape(-1) * seasonal[-len(lstm_trend) :] * 1

            y_true = test_data.scaler.inverse_transform(test_data.y.reshape(-1, 1))

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
                len(test_data.y),
                config=(self.arima_order, self.arima_trend),
            )
            arima_singal = arima_trend * seasonal[-len(lstm_trend) :] * 1

            weighted_signal = (
                self.factor * lstm_signal + (1 - self.factor) * arima_singal
            )

            if verbose:
                visualize_results(
                    test_data.df.index[test_data.time_steps :],
                    y_true,
                    weighted_signal,
                    "Combine Model",
                    f"{stock_name}_combine",
                )

            forecasts[stock_name] = weighted_signal
            real_vals[stock_name] = y_true.reshape(-1)

            if verbose:
                print("=" * 80)

        return forecasts, real_vals

    # @property
    # def stock_names(self):
    #     return self.stock_names

    # @property
    # def train_start_date(self):
    #     return self.train_start_date

    # @property
    # def train_end_date(self):
    #     return self.train_end_date

    # @property
    # def test_start_date(self):
    #     return self.test_start_date

    # @property
    # def test_end_date(self):
    #     return self.test_end_date

    # @stock_names.setter
    # def stock_names(self, stock_names):
    #     self.stock_names = stock_names
    #     self.load_data()

    # @train_start_date.setter
    # def train_date(self, train_start_date):
    #     self.train_start_date = train_start_date
    #     self.load_data()

    # @train_end_date.setter
    # def train_date(self, train_end_date):
    #     self.train_end_date = train_end_date
    #     self.load_data()

    # @test_start_date.setter
    # def train_date(self, test_start_date):
    #     self.test_start_date = test_start_date
    #     self.load_data()

    # @test_end_date.setter
    # def train_date(self, test_end_date):
    #     self.test_end_date = test_end_date
    #     self.load_data()
