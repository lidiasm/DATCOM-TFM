import re
import nltk

# Descarga una lista de stopwords
nltk.download('stopwords')

# Lista de stopwords en inglés y español
STOPWORDS = nltk.corpus.stopwords.words("spanish") + nltk.corpus.stopwords.words("english")

def delete_urls(dataset, column):
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

def delete_mentioned_users(dataset, column):
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

def delete_hashtags(dataset, column):
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

def delete_non_alphabetic_chars(dataset, column):
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

def delete_stopwords(dataset, column):
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

def delete_words_one_char(dataset, column):
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

def to_lowercase(dataset, column):
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
