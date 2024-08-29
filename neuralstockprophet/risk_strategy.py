from abc import ABC, abstractmethod

import numpy as np

__all__ = ["RiskDistributionStrategy", "EqualBudgetsStrategy", "MeanVarianceBudgetsStrategy"]


class RiskDistributionStrategy(ABC):
    @abstractmethod
    def calculate(self, Sigma):
        pass


class EqualBudgetsStrategy(RiskDistributionStrategy):
    def calculate(self, Sigma):
        n = Sigma.shape[0]
        return np.ones(n) / n


class MeanVarianceBudgetsStrategy(RiskDistributionStrategy):
    def calculate(self, Sigma):
        return np.linalg.inv(Sigma).dot(np.ones(Sigma.shape))
