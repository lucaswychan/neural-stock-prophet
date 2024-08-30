# Neural Stock Prophet

[![PyPI](https://img.shields.io/pypi/v/neuralstockprophet?label=pypi%20package&color)](https://pypi.org/project/neuralstockprophet/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/neuralstockprophet?color)](https://pypistats.org/packages/neuralstockprophet)
[![GitHub license badge](https://img.shields.io/github/license/lucaswychan/neural-stock-prophet?color=blue)](https://opensource.org/licenses/MIT)

**neuralstockprophet** integrates a variety of advanced techniques and algorithms to enhance the robustness, stability, and interoperability of stock price prediction methodologies. By leveraging machine learning, this package aims to accurately forecast the future values of company stocks and other financial assets traded on exchanges. Unlike existing approaches that predominantly focus on model configuration and tuning—often neglecting the inherent variability within the data—NeuralStockProphet addresses these challenges. Furthermore, it effectively mitigates issues related to overfitting and performance constraints that are commonly encountered in machine learning models.

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
import neuralstockprophet as nsp
import pandas as pd

prophet = nsp.NeuralStockProphet(
        stock_names=["AAPL", "GOOGL"],
        scaler_func=lambda: MinMaxScaler(feature_range=(0, 1)),
        train_start_date="2010-01-01",
        train_end_date="2019-12-31",
        test_start_date="2020-01-01",
        test_end_date="2020-12-31",
    )

forecasts, real_vals = prophet.forecast()

# Use the forecasted results to design the portfolio and get the assets allocation
portfolio = nsp.RiskParityPortfolio(prices=forecasts)

# Evaluate the performance of the portfolio with the forecasted results and the true stock prices
forecast_performance = portfolio.evaluate(forecasts)
real_performance = portfolio.evaluate(real_vals)

analyze_result_df = pd.concat([forecast_performance, real_performance], axis=0)
analyze_result_df.index = ["Forecast portfolio", "True portfolio"]
```

Get the historical data simply by inputting the stock codes.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/lucaswychan/neural-stock-prophet/blob/main/LICENSE) file for details.

## TODO

There are further improvements that can be made. Please have a look at the [TODO](https://github.com/lucaswychan/neural-stock-prophet/blob/main/TODO).
