import unittest

import pandas as pd

from data.process_data import clean_data


class TestCleanData(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({"categories": ["related-1;request-0;offer-0"]})
        self.cleaned_df = pd.DataFrame({"related": [1], "request": [0], "offer": [0]})

    def test_clean_data(self):
        # arrange
        df = self.df

        # act
        cleaned_df = clean_data(df)

        # assert
        pd.testing.assert_frame_equal(cleaned_df, self.cleaned_df)
