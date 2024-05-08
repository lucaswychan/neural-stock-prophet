import riskparityportfolio as rpp # requires manually install jax, jaxlib, tqdm, quadprog
import numpy as np
import pandas as pd
import empyrical as emp

class RiskParityPortfolio(rpp.RiskParityPortfolio):
    def __init__(self, prices, risk_distribution="eq", constraints=(None, None, None, None), seed=42):
        self.prices = prices
        
        self.construct(prices, risk_distribution, constraints, seed)

    def construct(self, risk_distribution="eq", constraints=(None, None, None, None), seed=42):
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
        
        Sigma = np.cov(self.log_returns.T)
        
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
        
    def evaluate(self, prices):
        lin_returns = np.diff(prices, axis=0) / prices[:-1]
        ret = lin_returns @ self.weights
        
        ret_df = pd.DataFrame({"Portfolio": ret})
        
        analyze_result_df = pd.DataFrame({"Sharpe ratio": ret_df.apply(emp.sharpe_ratio), "Max Drawdown": ret_df.apply(emp.max_drawdown).apply(lambda x: f"{x:.2%}"), "Annual return": ret_df.apply(emp.annual_return).apply(lambda x: f"{x:.2%}"), "Annual volatility": ret_df.apply(emp.annual_volatility).apply(lambda x: f"{x:.2%}")})
        
        return analyze_result_df
        
    @property
    def weights(self):
        return self.portfolio.weights
    
    @property
    def risk(self):
        return self.portfolio.risk()

    @property
    def log_returns(self):
        prices_df = pd.DataFrame(self.prices)
        log_returns = np.diff(np.log(prices_df), axis=0)
        return log_returns

    @property
    def lin_returns(self):
        lin_returns = np.diff(self.prices, axis=0) / self.prices[:-1]
        return lin_returns
    
    @property
    def risk(self):
        return self.portfolio.risk_contributions
    
    @property
    def volatility(self):
        return self.portfolio.volatility
    
    @property
    def number_of_assests(self):
        return self.portfolio.number_of_assets
    
    @property
    def budget(self):
        return self.portfolio.budget
    
    @property
    def Sigma(self):
        return self.portfolio.covariance
