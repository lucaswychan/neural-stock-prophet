import unittest
from sklearn.preprocessing import MinMaxScaler

from neuralstockprophet import NeuralStockProphet


class TestNeuralStockProphet(unittest.TestCase):
    def setUp(self):
        self.stocks = ["AAPL", "GOOGL"]
        self.train_start_date = "2010-01-01"
        self.train_end_date = "2019-12-31"
        self.test_start_date = "2020-01-01"
        self.test_end_date = "2020-12-31"
        self.nsp = NeuralStockProphet(
            stock_names=self.stocks,
            scaler_func=lambda: MinMaxScaler(feature_range=(0, 1)),
            train_start_date=self.train_start_date,
            train_end_date=self.train_end_date,
            test_start_date=self.test_start_date,
            test_end_date=self.test_end_date,
        )

    def test_train_model(self):
        # Test if the model is trained successfully
        self.nsp.train_model()
        self.assertIsNotNone(self.nsp.model)

    def test_predict(self):
        # Test if the model can make predictions
        self.nsp.train_model()
        prediction = self.nsp.predict()
        self.assertIsNotNone(prediction)

    def test_evaluate_model(self):
        # Test if the model evaluation is accurate
        self.nsp.train_model()
        evaluation = self.nsp.evaluate_model()
        self.assertGreaterEqual(evaluation, 0.0)
        self.assertLessEqual(evaluation, 1.0)


if __name__ == "__main__":
    unittest.main()
