from src.prophet import NeuralStockProphet
from src.args import args_parser
from src.portfolio import RiskParityPortfolio
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    args = args_parser()
    
    # Initialize the NeuralStockProphet model
    scaler = MinMaxScaler(feature_range = (0, 1))
    prophet = NeuralStockProphet(args.stock_names, scaler, args.train_date, args.test_date, args.time_steps, args.window_length, args.factor, args.epochs, args.batch_size, args.lr, args.arima_order, args.arima_trend)
    
    forecasts, real_vals = prophet.forecast(verbose=args.verbose)
    
    forecasts = np.array([forecasts[stock] for stock in args.stock_names]).T
    real_vals = np.array([real_vals[stock] for stock in args.stock_names]).T
    
    # Initialize the Portfolio model
    Cmat = None
    cvec = None
    Dmat = None
    dvec = None
    constraints = (Cmat, cvec, Dmat, dvec)
    portfolio = RiskParityPortfolio(forecasts, args.risk_distribution, constraints, seed=1016)
    
    analyze_result_df = pd.concat([portfolio.evaluate(forecasts), portfolio.evaluate(real_vals)], axis=0)
    analyze_result_df.index = ["Predicted Portfolio", "True Portfolio"]
    print(analyze_result_df)
    

if __name__ == '__main__':
    main()