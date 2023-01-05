import pandas as pd
import nltk
from textblob import TextBlob

EXIST_TRAIN_DATASET_PATH = "../data/EXIST2021_train.tsv"
EXIST_TEST_DATASET_PATH = "../data/EXIST2021_test.tsv"

def read_train_dataset():
    """
    Function that reads the EXIST train file to load the 
    train dataset as dataframe.

    Returns
    -------
    A Pandas dataframe
    """
    return pd.read_table(EXIST_TRAIN_DATASET_PATH)

def read_test_dataset():
    """
    Function that reads the EXIST test file to load the 
    test dataset as dataframe.

    Returns
    -------
    A Pandas dataframe
    """
    return pd.read_table(EXIST_TEST_DATASET_PATH)

def get_top_ngrams(dataset: pd.DataFrame, column: str, n_words: int):
    """
    Function that splits a set of texts into sentences 
    of N words (N-grams) to then calculate the frequency 
    of the phrases in the entire set.

    Parameters
    ----------
    dataset : Pandas dataframe
        The data which contains a set of texts to create and
        compute N-grams.
    column : str
        The column name in which the set of texts is stored.
    n_words : int
        The number of words to consider when creating sentences.

    Returns
    -------
    A Pandas dataframe
    """
    # Split texts in sentences of N words
    dataset["ngrams"] = dataset[column].str \
        .split() \
        .apply(lambda x: list(map(" ".join, nltk.ngrams(x, n=n_words))))

    # Compute the frequency per sentece
    return (dataset.\
        assign(count=dataset["ngrams"].str.len()) \
        .explode("ngrams")) \
        .sort_values("count", ascending=False)

def get_sentiments(dataset: pd.DataFrame, class_column: str, text_column: str):
    """
    Function that gets the sentiments of a set of text 
    splitted by a class column. The main purpose of this
    function is to know the number of positive, neutral 
    and negative texts per class.

    Parameters
    ----------
    dataset : Pandas dataframe
        The data which contains a set of texts to get the
        sentiments.
    class_column : str
        The column name which stores the class labels.
    text_column : str
        The column name which stores the text to process.

    Returns
    -------
    None if there aren't any class labels
    A dictionary with the sentiment frequencies per class
    """
    # Dictionary to save the sentiment frequency per class
    sentiment_freq_per_class = {}

    # Get a list of unique class labels
    unique_classes = list(set(list(dataset[class_column].values)))

    # Iterate over the class labels
    for class_ in unique_classes:

        # Get texts that belong to the current class
        one_class_texts = list(dataset[dataset[class_column] == class_][text_column].values)

        # Get the sentiment of each text
        # - Positive if polarity > 0
        # - Negative if polarity < 0
        # - Neutral if polatiry = 0
        one_class_sentiments = [
            "pos" if TextBlob(text).polarity > 0 else (
            "neg" if TextBlob(text).polarity < 0 
            else "neu") for text in one_class_texts]

        # Compute the frequency of each sentiment
        one_class_sentiment_frequencies = {sentiment:one_class_sentiments.count(sentiment) 
            for sentiment in ["pos", "neg", "neu"]}

        # Sort the quantities in descending order
        sentiment_freq_per_class[class_] = dict(sorted(one_class_sentiment_frequencies.items(), 
                    key=lambda item: item[1], reverse=True))

    return sentiment_freq_per_class