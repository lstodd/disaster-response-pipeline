import sys

import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath: str, categories_filepath: str) -> pd.DataFrame:
    """
    Load messages, categories and return a single DataFrame with duplicates removed.

    :param messages_filepath: File containing
    :param categories_filepath:
    :return: DataFrame containing
    """
    messages = pd.read_csv(messages_filepath)
    messages.drop_duplicates(inplace=True)

    categories = pd.read_csv(categories_filepath)
    categories.drop_duplicates(inplace=True)

    df = pd.merge(messages, categories, on="id", how="left")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess categories into separate columns.

    :param df: Raw DataFrame containing single column for all labels.
    :return: Cleaned DataFrame.
    """
    categories = df.categories.str.split(";", expand=True)

    row = categories.head(1)
    category_colnames = row.apply(lambda x: str(x[0][:-2])).values
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype("str").str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    df.drop(columns="categories", inplace=True)
    df = pd.concat([df, categories], axis=1)

    return df


def save_data(df, database_filename) -> None:
    """
    Save DataFrame to SQL engine.

    :param df: Cleaned DataFrame.
    :param database_filename: Filename to store SQL database.
    """
    engine = create_engine(database_filename)
    df.to_sql('RawData', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
