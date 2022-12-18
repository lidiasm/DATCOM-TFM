import re
import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Descarga una lista de stopwords
nltk.download('stopwords')

# Lista de stopwords en inglés y español
STOPWORDS = nltk.corpus.stopwords.words("spanish") + nltk.corpus.stopwords.words("english")

def delete_urls(dataset: pd.DataFrame, column: str):
    """
    Removes URLs and web links from a set of text.

    Parameters
    ----------
    dataset : Pandas dataframe
        The data which contains a set of texts.
    column : str
        The column name in which the set of texts is stored.

    Returns
    -------
    A Series column
    """
    return dataset[column].apply(lambda x: re.split("https:\/\/.*", str(x))[0])

def delete_mentioned_users(dataset: pd.DataFrame, column: str):
    """
    Removes mentioned users from a set of text.

    Parameters
    ----------
    dataset : Pandas dataframe
        The data which contains a set of texts.
    column : str
        The column name in which the set of texts is stored.

    Returns
    -------
    A Series column
    """
    return dataset[column].str \
        .replace("@([a-zA-Z0-9_]{1,50})", "", regex=True) 

def delete_hashtags(dataset: pd.DataFrame, column: str):
    """
    Removes hashtags and their content from a set of text.

    Parameters
    ----------
    dataset : Pandas dataframe
        The data which contains a set of texts.
    column : str
        The column name in which the set of texts is stored.

    Returns
    -------
    A Series column
    """
    return dataset[column].str \
        .replace("#([a-zA-Z0-9_]{1,50})", "", regex=True)

def delete_non_alphabetic_chars(dataset: pd.DataFrame, column: str):
    """
    Removes non-alphabetic characters (special characters,
    digits, ...) from a set of text.

    Parameters
    ----------
    dataset : Pandas dataframe
        The data which contains a set of texts.
    column : str
        The column name in which the set of texts is stored.

    Returns
    -------
    A Series column
    """
    return dataset[column].str \
        .replace("[^a-zA-Z ]", "")

def delete_stopwords(dataset: pd.DataFrame, column: str):
    """
    Removes stopwords from a set of english and
    spanish texts using the NLTK stopword list.

    Parameters
    ----------
    dataset : Pandas dataframe
        The data which contains a set of texts.
    column : str
        The column name in which the set of texts is stored.

    Returns
    -------
    A Series column
    """
    return dataset[column].apply(lambda x: " ".join(
        [word for word in x.split() if word not in STOPWORDS and len(word) > 1]))

def delete_words_one_char(dataset: pd.DataFrame, column: str):
    """
    Removes words composed of one letter within
    a set of texts.

    Parameters
    ----------
    dataset : Pandas dataframe
        The data which contains a set of texts.
    column : str
        The column name in which the set of texts is stored.

    Returns
    -------
    A Series column
    """
    return dataset[column].apply(lambda x: " ".join(
        [word for word in x.split() if len(word) > 1]))

def to_lowercase(dataset: pd.DataFrame, column: str):
    """
    Converts all characters to lowercase within
    a set of text.

    Parameters
    ----------
    dataset : Pandas dataframe
        The data which contains a set of texts.
    column : str
        The column name in which the set of texts is stored.

    Returns
    -------
    A Series column
    """
    return dataset[column].str.lower()

def text_processing_pipeline(dataset: pd.DataFrame, column: str):
    """
    Applies the next text processing techniques to a text
    column in a dataset. 
        - Removes links from texts.
        - Removes mentioned users from texts.
        - Removes hashtags from texts.
        - Removes special characters, digits, ...
          leaving only alphabetical characters.
        - Removes stopwords from english and spanish texts.
        - Removes one-size words from texts.
        - Converts characters to lowercase.

    Parameters
    ----------
    dataset : Pandas dataframe
        The data which contains a set of texts.
    column : str
        The column name in which the set of texts is stored.

    Returns
    -------
    A Pandas dataframe with a new column to store the cleaned text.
    """
    # Column name to store the cleaned text
    cleaned_col = "cleaned_text"

    # Delete URLs
    dataset[cleaned_col] = delete_urls(dataset, column)

    # Delete mentioned users in tweets
    dataset[cleaned_col] = delete_mentioned_users(dataset, cleaned_col)

    # Delete hashtags in tweets
    dataset[cleaned_col] = delete_hashtags(dataset, cleaned_col)
        
    # Delete non-alphabetic characters
    dataset[cleaned_col] = delete_non_alphabetic_chars(dataset, cleaned_col)

    # Delete stopwords for spanish and english texts
    dataset[cleaned_col] = delete_stopwords(dataset, cleaned_col)

    # Delete one-char words
    dataset[cleaned_col] = delete_words_one_char(dataset, cleaned_col)
    
    # Convert each character to lowercase
    dataset[cleaned_col] = to_lowercase(dataset, cleaned_col)

    return dataset

def to_bag_of_words(docs: list):
    """
    Creates a bag of words converting a list of text documents
    into numeric vectors by computing the word frequencies
    per text document.

    Parameters
    ----------
    doc: list
        A list of text documents.

    Returns
    -------
    A numpy 2-array.
    """
    vectorizer = CountVectorizer()
    return vectorizer.fit_transform(docs).toarray()
