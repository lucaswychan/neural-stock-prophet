# Neural Stock Prophet

[![PyPI version](https://badge.fury.io/py/neuralstockprophet.svg)](https://badge.fury.io/py/neuralstockprophet)
[![Downloads](https://pepy.tech/badge/neuralstockprophet)](https://pepy.tech/project/neuralstockprophet)

**neuralstockprophet** combines several techniques and algorithms to enhance the robustness, stability, and interoperability of the stock price prediction algorithm. Stock Price Prediction using a machine learning algorithm helps discover the future value of company stock and other financial assets traded on an exchange. Whereas, the existing methods relied highly on model setup and tuning, without considering the variation of data. Also, the machine learning model faces the problems of overfitting and performance limitations.

Combined techniques:

-   LSTM model with attention mechanisms
-   Multiplicative decomposition
-   ARIMA model

## Installation

-   Stable version

```
pip install neuralstockprophet
```

## Getting Started

```python
nsp = NeuralStockProphet(
        stock_names=["AAPL", "GOOGL"],
        scaler_func=lambda: MinMaxScaler(feature_range=(0, 1)),
        train_start_date="2010-01-01",
        train_end_date="2019-12-31",
        test_start_date="2020-01-01",
        test_end_date="2020-12-31",
    )
```

Get the historical data simply by inputting the stock codes.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/lucaswychan/neural-stock-prophet/blob/main/LICENSE) file for details.

## TODO

There are further improvements that can be made. Please have a look at the [TODO](https://github.com/lucaswychan/neural-stock-prophet/blob/main/TODO).
