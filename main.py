import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from args import args_parser
from prophet import RiskParityPortfolio
from prophet import NeuralStockProphet


def main():
    args = args_parser()

    # Initialize the NeuralStockProphet model
    scaler_func = lambda: MinMaxScaler(feature_range=(0, 1))
    prophet = NeuralStockProphet(
        args.stock_names,
        scaler_func,
        args.train_date[0],
        args.train_date[1],
        args.test_date[0],
        args.test_date[1],
        args.keep_ratio,
        args.time_steps,
        args.window_lengths,
        args.factor,
        args.epochs,
        args.batch_size,
        args.lr,
        tuple(args.arima_order),
        args.arima_trend,
    )

    forecasts, real_vals = prophet.forecast(verbose=args.verbose)

    forecasts = np.array([forecasts[stock] for stock in args.stock_names]).T
    real_vals = np.array([real_vals[stock] for stock in args.stock_names]).T

    # Initialize the Portfolio model
    Cmat = None
    cvec = None
    Dmat = None
    dvec = None
    constraints = (Cmat, cvec, Dmat, dvec)
    portfolio = RiskParityPortfolio(
        forecasts, args.risk_distribution, constraints, seed=1016
    )

    analyze_result_df = pd.concat(
        [portfolio.evaluate(forecasts), portfolio.evaluate(real_vals)], axis=0
    )
    analyze_result_df.index = ["Predicted Portfolio", "True Portfolio"]
    print(analyze_result_df)


if __name__ == "__main__":
    main()
