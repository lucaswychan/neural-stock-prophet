import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Union

from .arima import arima_forecast
from .dataset import TimeSeriesDataset
from .decompose import multiplicative_decompose
from .model import AttentionLSTM
from .utils import get_prediction_trend, visualize_results

warnings.simplefilter(action="ignore", category=FutureWarning)


class NeuralStockProphet:
    """
    NeuralStockProphet is a class that combines LSTM model with attention mechanisms, ARIMA model, and multiplicative decomposition to forecast stock prices. The class is designed to forecast stock prices for multiple stocks at once.
    
    Parameters
    ----------
    stock_names : List[str]
        A list of stock names to forecast
    scaler_func : callable
        A function that returns a scaler object to normalize the data
    train_start_date : Union[str, datetime]
        The start date of the training data
    train_end_date : Union[str, datetime]
        The end date of the training data
    test_start_date : Union[str, datetime]
        The start date of the testing data
    test_end_date : Union[str, datetime]
        The end date of the testing data
    keep_ratio : float, optional
        The ratio of data to keep for training, by default 0.8
    time_steps : int, optional
        The number of time steps to consider for each prediction, by default 60
    window_length : int, optional
        The window length for multiplicative decomposition, by default 48
    factor : float, optional
        The factor to combine the LSTM and ARIMA forecast, by default 0.9
    epochs : int, optional
        The number of epochs to train the LSTM model, by default 50
    batch_size : int, optional
        The batch size for training the LSTM model, by default 32
    lr : float, optional
        The learning rate for training the LSTM model, by default 1e-3
    arima_order : Tuple[int, int, int], optional
        The order of the ARIMA model, by default (1, 0, 6)
    arima_trend : str, optional
        The trend parameter for the ARIMA model, by default "ct"

    Examples
    --------

    >>> # Create a new instance of NeuralStockProphet
    >>> nsp = NeuralStockProphet(
    ...    stock_names=["AAPL", "GOOGL"],
    ...    scaler_func=lambda: MinMaxScaler(feature_range=(0, 1)),
    ...    train_start_date="2010-01-01",
    ...    train_end_date="2019-12-31",
    ...    test_start_date="2020-01-01",
    ...    test_end_date="2020-12-31",
    ... )

    >>> forecasts, real_vals = nsp.forecast(verbose=True)
    """
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

        self.forecasts = None
        self.real_vals = None

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

        self.forecasts = forecasts
        self.real_vals = real_vals

        return forecasts, real_vals
