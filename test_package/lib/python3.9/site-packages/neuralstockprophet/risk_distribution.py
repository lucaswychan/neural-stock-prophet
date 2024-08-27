import numpy as np


class RiskDistribution:
    def __init__(self, Sigma):
        self.Sigma = Sigma

    # feel free to add your own weights calculation method
    def calculate_budgets(self, risk_distribution):
        if risk_distribution == "eq":
            return self.equal_budgets()
        elif risk_distribution == "mv":
            return self.mean_variance_budgets()
        else:
            raise ValueError("Invalid risk distribution: {}".format(risk_distribution))

    def equal_budgets(self):
        return np.ones(len(self.Sigma)) / len(self.Sigma)

    def mean_variance_budgets(self):
        return np.linalg.inv(self.Sigma).dot(np.ones(len(self.Sigma)))
