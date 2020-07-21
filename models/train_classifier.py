import sys
from typing import Tuple, List

from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import joblib
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
import nltk

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


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
    :return: Training data, target and category nanes for target values.
    """
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql_table("RawData", engine)
    X = df["message"].values
    y = df.drop(columns=["id", "message", "original", "genre"]).values
    category_names = df.drop(columns=["id", "message", "original", "genre"]).columns.values

    return X, y, category_names


def tokenize(text: str) -> List[str]:
    """
    Lowers and splits a string up into separate tokens.

    :param text: Raw string to tokenize.
    :return: List of separated tokens.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model() -> GridSearchCV:
    """
    Build the sklearn model pipeline with paramters to grid search over.

    :return: Sklearn grid searchable pipeline.
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # pipeline = Pipeline([
    #     ('vect', CountVectorizer(tokenizer=tokenize)),
    #     ('tfidf', TfidfTransformer()),
    #     ('clf', MultiOutputClassifier(RandomForestClassifier()))
    # ])

    # specify parameters for grid search
    parameters = {
        'features__text_pipeline__tfidf__smooth_idf': (True, False),
        'clf__estimator__warm_start': (True, False),
        'clf__estimator__min_samples_leaf': [2, 3, 4],
    }

    # create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, y_test, category_names: np.array):
    """
    Returns the best parameters from a grid search.

    :param model: Trained model.
    :param X_test: Data for features.
    :param y_test: Actual labels to check against.
    :param category_names: Named of predicted labels.
    :return: Return the parameters of the best model from grid search.
    """
    y_pred = model.predict(X_test)
    for i in range(36):
        print(i)
        print(classification_report(y_test[:, i], y_pred[:, i]))

    print("\nBest Parameters:", model.best_params_)


def save_model(model: GridSearchCV, model_filepath: str) -> None:
    """
    Save the serialised model object.

    :param model: Model object.
    :param model_filepath: Model filename.
    """
    joblib.dump(model, model_filepath)


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
