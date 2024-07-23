import numpy as np


def multiplicative_decompose(data, window_length):
    # trend (perform exponential moving average)
    trend = data.ewm(window_length).mean()

    # sesonal component
    data_detrended = data / trend
    seasonal = np.zeros(len(data_detrended))
    for i in range(window_length):
        idx = np.arange(i, len(data_detrended), window_length, dtype=int)
        seasonal[idx] = np.mean(data_detrended[idx])

    # residual component
    residual = data / seasonal / trend

    return trend, seasonal, residual
