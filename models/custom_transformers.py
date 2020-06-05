import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
import nltk

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

from models.train_classifier import tokenize


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text: str):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
