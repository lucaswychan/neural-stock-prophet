import unittest
from neuralstockprophet import NeuralStockProphet

class TestNeuralStockProphet(unittest.TestCase):
    def setUp(self):
        self.nsp = NeuralStockProphet()

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

if __name__ == '__main__':
    unittest.main()