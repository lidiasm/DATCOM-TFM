import re
import nltk
import spacy
import warnings
import pandas as pd
from autocorrect import Speller

# Ignore regex warnins
warnings.filterwarnings('ignore')

# Global variable to store the name of the column
# that will save the processed texts
CLEAN_TEXTS_COLNAME = 'clean_text'

# Download a list of stopwords
nltk.download('stopwords')

# Global variable to store the list of stopwords
# for English and Spanish texts
STOPWORDS = nltk.corpus.stopwords.words('spanish') + \
            nltk.corpus.stopwords.words('english')


def delete_urls(dataset: pd.DataFrame, text_col: str):
    '''
    Removes URLs and web links from a set of text.

    Parameters
    ----------
    dataset : Pandas dataframe
        The data which contains a set of texts.
    text_col : str
        The column name in which the set of texts is stored.

    Returns
    -------
    A Pandas Series column with the processed texts.
    '''
    return dataset[text_col].apply(lambda x: re.split(
        pattern='https:\/\/.*',
        string=str(x))[0])


def delete_mentioned_users(dataset: pd.DataFrame, text_col: str):
    '''
    Removes mentioned users from a set of text.

    Parameters
    ----------
    dataset : Pandas dataframe
        The data which contains a set of texts.
    text_col : str
        The column name in which the set of texts is stored.

    Returns
    -------
    A Pandas Series column with the processed texts.
    '''
    return dataset[text_col].str.replace(
        pat='@([a-zA-Z0-9_]{1,50})',
        repl='')


def delete_non_alphabetic_chars(dataset: pd.DataFrame, text_col: str):
    '''
    Removes non-alphabetic characters (special characters,
    digits, ...) from a set of text.

    Parameters
    ----------
    dataset : Pandas dataframe
        The data which contains a set of texts.
    text_col : str
        The column name in which the set of texts is stored.

    Returns
    -------
    A Pandas Series column with the processed texts.
    '''
    return dataset[text_col].str.replace(
        pat='[^a-zA-Z ]', 
        repl='')


def delete_words_without_vowels(dataset: pd.DataFrame, text_col: str):
    '''
    Removes words that don't contain any vowels from a
    set of texts because they are misspelled and useless.

    Parameters
    ----------
    dataset : Pandas dataframe
        The data which contains a set of texts.
    text_col : str
        The column name in which the set of texts is stored.

    Returns
    -------
    A Pandas Series column with the processed texts.
    '''
    vowset = set(['a', 'e', 'i', 'o', 'u'])
    return dataset[text_col].apply(lambda x: ' '.join([
        word for word in x.split() if len(vowset.intersection(word)) > 0
    ]))


def delete_stopwords(dataset: pd.DataFrame, text_col: str, add_stopwords: list = []):
    '''
    Removes the stopwords found in a set of English and Spanish
    texts using the list of stopwords from NLTK library as
    well as the list of stopwords provided, if any.

    Parameters
    ----------
    dataset : Pandas dataframe
        The data which contains a set of texts.
    text_col : str
        The column name in which the set of texts is stored.
    add_stopwords : list (optional, default empty list)
        A list of additional stopwords to exclude.

    Returns
    -------
    A Pandas Series column with the processed texts.
    '''
    STOPWORDS.extend(add_stopwords)

    return dataset[text_col].apply(lambda x: ' '.join([
        word for word in x.split() if word not in STOPWORDS and len(word) > 1
    ]))


def to_lowercase(dataset: pd.DataFrame, text_col: str):
    '''
    Converts every character of each text to lowercase.

    Parameters
    ----------
    dataset : Pandas dataframe
        The data which contains a set of texts.
    text_col : str
        The column name in which the set of texts is stored.

    Returns
    -------
    A Pandas Series column with the processed texts.
    '''
    return dataset[text_col].str.lower()


def correct_misspelled_words(dataset: pd.DataFrame, text_col: str):
    '''
    Detects possible misspelled words in a set of English
    and Spanish texts to then replaces them with the right 
    words.

    Parameters
    ----------
    dataset : Pandas dataframe
        The data which contains a set of texts.
    text_col : str
        The column name in which the set of texts is stored.

    Returns
    -------
    A Pandas Series column with the processed texts.
    '''
    en_speller = Speller()
    es_speller = Speller('es')

    en_dataset = dataset[dataset['language'] == 'en']
    es_dataset = dataset[dataset['language'] == 'es']

    en_dataset[text_col] = en_dataset[text_col].apply(lambda x: en_speller(x))
    es_dataset[text_col] = es_dataset[text_col].apply(lambda x: es_speller(x))

    return pd.concat([en_dataset, es_dataset])[text_col]


def lemmatize_texts(dataset: pd.DataFrame, text_col: str):
    '''
    Iterates over a set of documents in order to replace 
    each verb with its infinitive form and each word with 
    its single form. The lemmatizer to apply depends on the 
    text language, considering the following languages: 
       - English 
       - Spanish.

    Parameters
    ----------
    dataset : Pandas dataframe
        The data which contains a set of texts to lemmatize.
    text_col : str
        The column name in which the set of texts is stored.

    Returns
    -------
    A list of strings with the lemmatized texts.
    '''
    lemm_texts = []

    # Create two lemmatizer for english and spanish texts
    en_lemmatizer = spacy.load('en_core_web_lg')
    es_lemmatizer = spacy.load('es_core_news_lg')

    for record in dataset.to_records('dict'):

        # Select the lemmatizer depending on the text language
        lemmatizer = en_lemmatizer if record['language'] == 'en' else es_lemmatizer

        current_lemm_text = ''
        # Iterate over the words to be lemmatized
        for word in lemmatizer(record[text_col]):
            current_lemm_text += word.lemma_ + ' '

        lemm_texts.append(current_lemm_text)

    return lemm_texts


def process_texts_encode_labels(
        dataset: pd.DataFrame, text_col: str, class_col: str,
        lemmatize: bool, correct_words: bool):
    '''
    Processes the texts stored in the provided dataset
    as well as the class labels applying the following 
    techniques:

        - Delete URLs.
        - Delete mentioned users.
        - Delete non-alphabetic characters.
        - Delete words without any vowels.
        - Convert each character to lowercase.
        - Map categorical class labels to numeric class labels.
        - (optional) Lemmatize texts.
        - (optional) Detect and correct misspelled words.

    Parameters
    ----------
    dataset : Pandas dataframe
        A dataset with the list of texts to process and the
        list of class labels to encode.
    text_col : str
        The column name which contains the texts to process.
    class_col : str
        The column name which contains the class labels to
        encode to numeric class labels.
    lemmatize : bool
        True to lemmatize the texts stored in the dataset.
    correct_words : bool
        True to detect and correct misspelled words from
        the texts stored in the dataset.

    Returns
    -------
    A Pandas dataframe with the processed dataset.
    '''
    dataset[CLEAN_TEXTS_COLNAME] = delete_urls(
        dataset=dataset, 
        text_col=text_col)

    dataset[CLEAN_TEXTS_COLNAME] = delete_mentioned_users(
        dataset=dataset, 
        text_col=CLEAN_TEXTS_COLNAME)

    dataset[CLEAN_TEXTS_COLNAME] = delete_non_alphabetic_chars(
        dataset=dataset, 
        text_col=CLEAN_TEXTS_COLNAME)
    
    dataset[CLEAN_TEXTS_COLNAME] = delete_words_without_vowels(
        dataset=dataset, 
        text_col=CLEAN_TEXTS_COLNAME)
    
    dataset[CLEAN_TEXTS_COLNAME] = to_lowercase(
        dataset=dataset, 
        text_col=CLEAN_TEXTS_COLNAME)
    
    # Encode the class labels to numeric class labels
    dataset[class_col].replace(
        ['non-sexist', 'sexist'],
        [0, 1],
        inplace=True
    )

    if (correct_words):
        dataset[CLEAN_TEXTS_COLNAME] = correct_misspelled_words(
            dataset=dataset, 
            text_col=CLEAN_TEXTS_COLNAME)

    if (lemmatize):
        dataset[CLEAN_TEXTS_COLNAME] = lemmatize_texts(
            dataset=dataset, 
            text_col=CLEAN_TEXTS_COLNAME)

    return dataset