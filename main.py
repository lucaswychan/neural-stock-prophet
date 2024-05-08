from src.prophet import NeuralStockProphet
from src.args import args_parser
from src.portfolio import RiskParityPortfolio
from sklearn.preprocessing import MinMaxScaler


def main():
    args = args_parser()
    
    # Initialize the NeuralStockProphet model
    scaler = MinMaxScaler(feature_range = (0, 1))
    prophet = NeuralStockProphet(args.stock_names, scaler, args.train_date, args.test_date, args.time_steps, args.window_length, args.factor, args.epochs, args.batch_size, args.lr, args.arima_order, args.arima_trend)
    
    # Initialize the Portfolio model
    Cmat = None
    cvec = None
    Dmat = None
    dvec = None
    constraints = (Cmat, cvec, Dmat, dvec)
    portfolio = RiskParityPortfolio(prophet, args.risk_distribution, constraints, seed=1016)
    
    

if __name__ == '__main__':
    main()