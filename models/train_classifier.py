import sys
from typing import Tuple

from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import pickle as pkl

import nltk

nltk.download(['punkt', 'wordnet'])

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

# from models.custom_transformers import StartingVerbExtractor
from sklearn.base import BaseEstimator, TransformerMixin


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


def load_data(database_filepath: str) -> Tuple[np.array, np.array, np.array]:
    """
    Load the training data from the SQLite database.

    :param database_filepath: Database name to load cleaned data from.
    :return:
    """
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql_table("RawData", engine)
    X = df["message"].values
    y = df.drop(columns=["id", "message", "original", "genre"]).values
    category_names = df.drop(columns=["id", "message", "original", "genre"]).columns.values

    return X, y, category_names


def tokenize(text: str):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    # pipeline = Pipeline([
    #     ('features', FeatureUnion([
    #
    #         ('text_pipeline', Pipeline([
    #             ('vect', CountVectorizer(tokenizer=tokenize)),
    #             ('tfidf', TfidfTransformer())
    #         ])),
    #
    #         ('starting_verb', StartingVerbExtractor())
    #     ])),
    #
    #     ('clf', MultiOutputClassifier(RandomForestClassifier()))
    # ])

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # specify parameters for grid search
    parameters = {
        'tfidf__smooth_idf': (True, False),
        'clf__estimator__warm_start': (True, False),
        'clf__estimator__min_samples_leaf': [2, 3, 4],
    }

    # create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, y_test, category_names):
    y_pred = model.predict(X_test)
    for i in range(36):
        print(i)
        print(classification_report(y_test[:, i], y_pred[:, i]))

    print("\nBest Parameters:", model.best_params_)


def save_model(model, model_filepath):
    model_file = open(model_filepath, 'w')
    pkl.dump(model, model_file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')
        # database_filepath 'sqlite:///DisasterResponse.db'
        # model classifier.pkl


if __name__ == '__main__':
    main()
