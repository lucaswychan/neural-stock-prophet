import riskparityportfolio as rpp # requires manually install jax, jaxlib, tqdm, quadprog
import numpy as np
import pandas as pd
import empyrical as emp

class Portfolio:
    def __init__(self, prices, risk_distribution="eq", constraints=(None, None, None, None), seed=42):
        self.prices = prices
        self.construct_portfolio(prices, risk_distribution, constraints, seed)

    def construct_portfolio(self, risk_distribution="eq", constraints=(None, None, None, None), seed=42):
        """
        minimize R(w) - alpha * mu.T * w + lambda * w.T Sigma w
        subject to Cw = c, Dw <= d
        
        please visit https://github.com/convexfi/riskparity.py for more portfolio construction information
        """
        Cmat, cvec, Dmat, dvec = constraints
        if (Cmat is None) != (cvec is None) or (Dmat is None) != (dvec is None):
            raise ValueError("Invalid constraints")
        
        if seed:
            np.random.seed(seed)
        
        # Calculate the returns based on the predicted and true prices
        prices_df = pd.DataFrame(self.prices)
        
        # log returns for covariance matrix calculation
        log_returns = np.diff(np.log(prices_df), axis=0)
        
        Sigma = np.cov(log_returns.T)
        
        b = None
        if risk_distribution == "eq":
            b = np.ones(len(Sigma)) / len(Sigma)
        elif risk_distribution == "mv":
            b = np.linalg.inv(Sigma).dot(np.ones(len(Sigma)))
        else:
            raise ValueError("Invalid risk distribution : {}".format(risk_distribution))
        
        # Construct a risk parity portfolio
        self.portfolio = rpp.RiskParityPortfolio(covariance=Sigma, budget=b)
        self.portfolio.design(Cmat=Cmat, cvec=cvec, Dmat=Dmat, dvec=dvec)
    
    def change_prices(self, prices):
        self.prices = prices
        self.construct_portfolio(prices)
    
    @property
    def risk(self):
        return self.portfolio.risk()

    def get_portfolio_pnl(portfolio, prices):
        prices_df = pd.DataFrame(prices)
        prices_mat = np.asarray(prices_df)
        lin_prices = np.diff(prices_mat, axis=0) / prices_mat[:-1]
        
        return emp.cum_returns(lin_prices @ portfolio.weights)


    @property
    def log_returns(self):
        prices_df = pd.DataFrame(self.prices)
        log_returns = np.diff(np.log(prices_df), axis=0)
        return log_returns

    @property
    def lin_returns(self):
        prices_df = pd.DataFrame(self.prices)
        prices_mat = np.asarray(prices_df)
        lin_returns = np.diff(prices_mat, axis=0) / prices_mat[:-1]
        return lin_returns

    @property
    def weights(self):
        return self.portfolio.weights
