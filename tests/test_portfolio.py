import unittest
from neuralstockprophet import BasePortfolio, RiskParityPortfolio

class TestBasePortfolio(unittest.TestCase):
    def setUp(self):
        self.bp = BasePortfolio()


class TestRiskParityPortfolio(unittest.TestCase):
    def setUp(self):
        self.rpp = RiskParityPortfolio()


if __name__ == '__main__':
    unittest.main()