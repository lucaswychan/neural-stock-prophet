from statsmodels.tsa.arima.model import ARIMA


def arima_forecast(data, len_forecast, config=((1, 0, 6), "ct")):
    order, trend = config
    # Define and fit the model
    model = ARIMA(data, order=order, trend=trend).fit()
    # Make a forecast
    yhat = model.forecast(steps=len_forecast)

    return yhat
