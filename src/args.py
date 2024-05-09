import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stock_names",
        nargs="+",
        type=str,
        help="List of stock names to forecast and analyze",
        default=["AAPL", "GOOG", "MSFT", "AMZN", "TSLA"],
    )
    parser.add_argument(
        "--train_date",
        nargs=2,
        type=str,
        help="Start and end date for training data (in the form of YYYY-MM-DD)",
        default=["2016-01-01", "2022-12-31"],
    )
    parser.add_argument(
        "--test_date",
        nargs=2,
        type=str,
        help="Start and end date for testing data (in the form of YYYY-MM-DD)",
        default=["2023-01-01", "2023-12-31"],
    )
    parser.add_argument(
        "--time_steps",
        type=int,
        help="Number of time steps to consider for forecasting",
        default=60,
    )
    parser.add_argument(
        "--window_length",
        type=int,
        help="Length of the window for multiplicative decomposition",
        default=48,
    )
    parser.add_argument(
        "--factor",
        type=float,
        help="Factor to combine the LSTM and ARIMA forecast",
        default=0.9,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs for training the LSTM model",
        default=50,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for training the LSTM model",
        default=32,
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate for training the LSTM model",
        default=1e-3,
    )
    parser.add_argument(
        "--arima_order",
        nargs=3,
        type=int,
        help="Order of the ARIMA model (enter 3 integers)",
        default=[1, 0, 6],
    )
    parser.add_argument(
        "--arima_trend", type=str, help="Trend for the ARIMA model", default="ct"
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        help="Determine if saving and printing the result",
        default=False,
    )
    parser.add_argument(
        "--risk_distribution",
        type=str,
        help="Determine the budget allocation of the portfolio",
        default="eq",
    )

    args = parser.parse_args()

    return args
