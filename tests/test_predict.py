import unittest
from src.predict import predict_message

class FakeModel:
    classes_ = ["ham", "spam"]
    def predict(self, messages):
        return ["spam"]
    def predict_proba(self, messages):
        return [[0.2, 0.8]]

class PredictionTests(unittest.TestCase):
    def test_predict_returns_label_and_score(self):
        result = predict_message("Claim your prize now", model=FakeModel())
        self.assertEqual(result["label"], "spam")
        self.assertAlmostEqual(result["spam_probability"], 0.8)

    def test_empty_message_rejected(self):
        with self.assertRaises(ValueError):
            predict_message("  ", model=FakeModel())

if __name__ == "__main__":
    unittest.main()
