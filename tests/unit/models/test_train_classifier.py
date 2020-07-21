import unittest

from models.train_classifier import tokenize


class TestTokenizer(unittest.TestCase):
    def setUp(self):
        self.message = "We need food"

    def test_tokenize_splits_simple_sentence(self):
        # arrange
        message = self.message

        # act
        toeknized_message = tokenize(message)

        # assert
        self.assertEqual(len(toeknized_message), 3)
        self.assertListEqual(toeknized_message, ["we", "need", "food"])
