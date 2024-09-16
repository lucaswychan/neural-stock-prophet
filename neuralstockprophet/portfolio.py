from abc import ABC, abstractmethod
from typing import Optional, Tuple

import empyrical as emp
import numpy as np
import pandas as pd
import riskparityportfolio as rpp  # requires manually install jax, jaxlib, tqdm, quadprog

from .risk_strategy import *

__all__ = ["BasePortfolio", "RiskParityPortfolio"]


class BasePortfolio(ABC):
    """
    BasePortfolio is an abstract class that defines the basic structure of a portfolio

    Override the `construct` and `evaluate` methods to implement a custom portfolio
    """

    @abstractmethod
    def construct(self, **kwargs):
        """
        The method to construct the portfolio.
        """
        pass

    @abstractmethod
    def evaluate(self, prices, **kwargs):
        """
        The method to evaluate the portfolio performance.
        """
        pass


class RiskParityPortfolio(rpp.RiskParityPortfolio, BasePortfolio):
    def __init__(
        self,
        prices: np.ndarray,
        strategy: RiskDistributionStrategy = EqualBudgetsStrategy(),
        constraints: tuple = (None, None, None, None),
        weights=None,
        risk_concentration=None,
        seed: Optional[int] = 1016,
    ):
        self.prices = prices
        self.Sigma = np.cov(self.log_returns.T)

        budget = strategy.calculate(self.Sigma)

        rpp.RiskParityPortfolio.__init__(
            self,
            covariance=self.Sigma,
            budget=budget,
            weights=weights,
            risk_concentration=risk_concentration,
        )

        self.seed = seed
        self.constraints = constraints

        self.construct()

    def construct(self) -> None:
        """
        minimize R(w) - alpha * mu.T * w + lambda * w.T Sigma w
        subject to Cw = c, Dw <= d

        please visit https://github.com/convexfi/riskparity.py for more portfolio construction information
        """
        Cmat, cvec, Dmat, dvec = self.constraints
        if (Cmat is None) != (cvec is None) or (Dmat is None) != (dvec is None):
            raise ValueError("Invalid constraints")

        if self.seed:
            np.random.seed(self.seed)

        # Construct a risk parity portfolio
        self.design(Cmat=Cmat, cvec=cvec, Dmat=Dmat, dvec=dvec)

    def evaluate(self, prices, **kwargs) -> pd.DataFrame:
        """
        Evaluate the portfolio performance with the true stock prices

        Parameters
        ----------
        prices : np.ndarray
            The true stock prices

        Returns
        -------
        analyze_result_df : pd.DataFrame
            The performance metrics of the portfolio
        """
        lin_returns = np.diff(prices, axis=0) / prices[:-1]
        ret = lin_returns @ self.weights

        ret_df = pd.DataFrame({"Portfolio": ret})

        analyze_result_df = pd.DataFrame(
            {
                "Sharpe ratio": ret_df.apply(emp.sharpe_ratio),
                "Max Drawdown": ret_df.apply(emp.max_drawdown).apply(lambda x: f"{x:.2%}"),
                "Annual return": ret_df.apply(emp.annual_return).apply(lambda x: f"{x:.2%}"),
                "Annual volatility": ret_df.apply(emp.annual_volatility).apply(lambda x: f"{x:.2%}"),
            }
        )

        return analyze_result_df

    @property
    def log_returns(self):
        if self.prices is None:
            raise ValueError("Prices are not set")
        log_returns = np.diff(np.log(self.prices), axis=0)
        return log_returns

    @property
    def lin_returns(self):
        if self.prices is None:
            raise ValueError("Prices are not set")
        lin_returns = np.diff(self.prices, axis=0) / self.prices[:-1]
        return lin_returns
