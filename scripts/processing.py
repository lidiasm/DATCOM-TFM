import os
import re
import nltk
import spacy
import warnings
import pandas as pd
from sklearn import preprocessing
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
from autocorrect import Speller

# Ignore regex warnins
warnings.filterwarnings('ignore')

# Download a list of stopwords
nltk.download('stopwords')

# Get the spanish and engish stopwords
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
    A Series column
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
    A Series column
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
    A Series column
    '''
    return dataset[text_col].str.replace(
        pat='[^a-zA-Z ]', 
        repl='')


def delete_stopwords(dataset: pd.DataFrame, text_col: str, add_stopwords: list = []):
    '''
    Removes stopwords from a set of english and
    spanish texts using the NLTK stopword list.

    Parameters
    ----------
    dataset : Pandas dataframe
        The data which contains a set of texts.
    text_col : str
        The column name in which the set of texts is stored.
    add_stopwords : list
        A list of additional stopwords to exclude.

    Returns
    -------
    A Series column
    '''
    STOPWORDS.extend(add_stopwords)

    return dataset[text_col].apply(lambda x: ' '.join([
        word for word in x.split() if word not in STOPWORDS and len(word) > 1
    ]))


def delete_wrong_words(dataset: pd.DataFrame, text_col: str):
    '''
    Removes words composed of one letter or of only consonants
    without vowels within a set of texts.

    Parameters
    ----------
    dataset : Pandas dataframe
        The data which contains a set of texts.
    text_col : str
        The column name in which the set of texts is stored.

    Returns
    -------
    A Series column with the processed text
    '''
    vowset = set(['a', 'e', 'i', 'o', 'u'])
    return dataset[text_col].apply(lambda x: ' '.join([
        word for word in x.split() if len(word) > 1 and len(vowset.intersection(word)) > 0
    ]))


def to_lowercase(dataset: pd.DataFrame, text_col: str):
    '''
    Converts all characters to lowercase within
    a set of text.

    Parameters
    ----------
    dataset : Pandas dataframe
        The data which contains a set of texts.
    text_col : str
        The column name in which the set of texts is stored.

    Returns
    -------
    A Series column
    '''
    return dataset[text_col].str.lower()


def correct_misspelled_words(dataset: pd.DataFrame, text_col: str):
    '''
    Detects possible misspelled words within a set of texts, only in English
    and Spanish, to then replaces them with the right words. Texts should be 
    stored in a column of a provided dataframe.

    Parameters
    ----------
    dataset : Pandas dataframe
        The data which contains a set of texts.
    text_col : str
        The column name in which the set of texts is stored.

    Returns
    -------
    A Series column
    '''
    en_speller = Speller()
    es_speller = Speller('es')

    en_dataset = dataset[dataset['language'] == 'en']
    es_dataset = dataset[dataset['language'] == 'es']

    en_dataset[text_col] = en_dataset[text_col].apply(lambda x: en_speller(x))
    es_dataset[text_col] = es_dataset[text_col].apply(lambda x: es_speller(x))

    return pd.concat([en_dataset, es_dataset])[text_col]


def to_lemmatized_texts(dataset: pd.DataFrame, text_col: str):
    '''
    Iterates over a set of documents in order to replace 
    each verb with its infinitive form and each word with 
    its single form. The lemmatizer to apply depends on the 
    text language, considering two languages: english and spanish.

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
    # List to save the lemmatized texts
    lemm_texts = []

    # Create two lemmatizer for english and spanish texts
    en_lemmatizer = spacy.load('en_core_web_lg')
    es_lemmatizer = spacy.load('es_core_news_lg')

    # Iterate over the set of texts
    for record in dataset.to_records('dict'):

        # Select the lemmatizer depending on the text language
        lemmatizer = en_lemmatizer if record['language'] == 'en' else es_lemmatizer

        # String to save the current text when it's lemmatized
        current_lemm_text = ''

        # Iterate over the words to be lemmatized
        for word in lemmatizer(record[text_col]):
            current_lemm_text += word.lemma_ + ' '

        # Add the lemmatized text to the dataset
        lemm_texts.append(current_lemm_text)

    return lemm_texts


def to_stemmed_texts(dataset: pd.DataFrame, text_col: str):
    '''
    Iterates over a set of documents in order to replace each
    word with its root form. The stemmer to apply depends on the 
    text language, considering two languages: english and spanish.

    Parameters
    ----------
    dataset : Pandas dataframe
        The data which contains a set of texts.
    text_col : str
        The column name in which the set of texts is stored.

    Returns
    -------
    A list of strings with the stemmed texts.
    '''
    # List to save the stemmed texts
    stemm_texts = []

    # Create two stemmer for english and spanish texts
    en_stemmer = SnowballStemmer(language='english')
    es_stemmer = SnowballStemmer(language='spanish')

    # Iterate over the set of texts
    for record in dataset.to_records('dict'):

        # Select the stemmer depending on the text language
        stemmer = en_stemmer if record['language'] == 'en' else es_stemmer

        # String to save the current text when it's stemmed
        current_stemm_text = ''

        # Iterate over the words to be stemmed
        for word in record[text_col].split(' '):
            current_stemm_text += stemmer.stem(word) + ' '

        # Add the stemmed text to the dataset
        stemm_texts.append(current_stemm_text)

    return stemm_texts


def text_processing_pipeline(dataset: pd.DataFrame, text_col: str,
    add_stopwords: list = [], lemm: bool = False, 
    stemm: bool = False, correct_words=False):
    '''
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
        - Replaces misspelled words with the right ones. Only English and Spanish.

    Parameters
    ----------
    dataset : Pandas dataframe
        The data which contains a set of texts.
    text_col : str
        The column name in which the set of texts is stored.
    add_stopwords : list
        A list of additional stopwords to exclude.
    lemm : bool
        True to lemmatize the texts, False to not apply it.
    stemm : bool
        True to apply stemming to the texts, False to not apply it.
    correct_words : bool
        True to detect and correct misspelled words, False to not do it.

    Returns
    -------
    A Pandas dataframe with a new column to store the cleaned text.
    '''
    # Column name to store the cleaned text
    cleaned_col = 'cleaned_text'

    # Delete URLs
    dataset[cleaned_col] = delete_urls(
        dataset=dataset, 
        column=text_col)

    # Delete mentioned users in tweets
    dataset[cleaned_col] = delete_mentioned_users(
        dataset=dataset, 
        column=cleaned_col)

    # Delete non-alphabetic characters
    dataset[cleaned_col] = delete_non_alphabetic_chars(
        dataset=dataset, 
        column=cleaned_col)

    # Delete stopwords for spanish and english texts
    dataset[cleaned_col] = delete_stopwords(
        dataset=dataset, 
        column=cleaned_col,
        add_stopwords=add_stopwords)

    # Delete one-char words
    dataset[cleaned_col] = delete_wrong_words(
        dataset=dataset, 
        column=cleaned_col)

    # Convert each character to lowercase
    dataset[cleaned_col] = to_lowercase(
        dataset=dataset, 
        column=cleaned_col)
  
    # Correct misspelled words
    if (correct_words and type('language') == str and 'language' != ''):
      dataset[cleaned_col] = correct_misspelled_words(
        dataset=dataset, 
        text_col=cleaned_col)

    # Lemmatize the texts
    if (lemm and type('language') == str and 'language' != ''):
        dataset[cleaned_col] = to_lemmatized_texts(
            dataset=dataset, 
            text_col=cleaned_col)

    # Text stemming
    if (stemm and type('language') == str and 'language' != ''):
        dataset[cleaned_col] = to_stemmed_texts(
            dataset=dataset, 
            text_col=cleaned_col)

    return dataset


def to_numeric_labels(dataset: pd.DataFrame, class_col: str, encoding: dict = {}):
    '''
    Replaces categorical labels stored in a specific column within
    a dataset with the provided list of numeric values or creating
    it using a LabelEncoder object.

    Parameters
    ----------
    dataset : Pandas dataframe
        A dataset which contains the class labels to encode.
    class_col : str
        The column name in which the class labels are stored.
    encoding : dict (optional)
        A dictionary which contains the categorical labels and
        their numeric labels associated.

    Returns
    -------
    A list with the encoded labels if an encoding is provided.
    Otherwise a dictionary whose keys are:
        - 'values': contains a list with the encoded values
        - 'classes': stores a dictionary with the links between 
                    the original and the encoded labels.
    '''
    if len(encoding) > 0:
        # Replace the categorical labels with the numeric labels
        return [encoding[val] for val in list(dataset[class_col].values)]

    else:
        # Create a new encoder
        encoder = preprocessing.LabelEncoder()

        # Encode the categorical values to numeric
        encoded_values = encoder.fit(y=list(dataset[class_col].values))

        return {
            'values': encoded_values,
            'classes': encoder.classes_
        }


def process_encode_datasets(
    train_df: pd.DataFrame, test_df: pd.DataFrame,
    lemm: bool, stemm: bool, correct_words: bool):
    '''
    Processes the sets of train and test documents
    and encodes the class labels of both datasets to convert
    them into numeric labels.

    Parameters
    ----------
    train_df : Pandas dataframe
        A dataset which contains the train samples.
    test_df : Pandas dataframe
        A dataset which contains the test samples.
    lemm : bool
        True to lemmatize the texts, False to not apply it.
    stemm : bool
        True to apply stemming to the texts, False to not apply it.
    correct_words : bool
        True to detect and correct misspelled words, False to not apply it.

    Returns
    -------
    A dictionary with the processed data and the encoded labels.
        - 'train_df': a Pandas dataframe with the processed train texts.
        - 'test_df': a Pandas dataframe with the processed test texts.
        - 'encoded_train_labels': a list of numeric values with the encoded train labels.
        - 'encoded_test_labels': a list of numeric values with the encoded test labels.
    '''
    # Specific encoding for class labels
    encoding = {
        'non-sexist': 0, 
        'sexist': 1
    }

    return {
        'train_df': text_processing_pipeline(
            dataset=train_df, 
            text_col='text', 
            lemm=lemm,
            stemm=stemm, 
            correct_words=correct_words),
        'test_df':text_processing_pipeline(
            dataset=test_df, 
            text_col='text', 
            lemm=lemm,
            stemm=stemm, 
            correct_words=correct_words),
        'encoded_train_labels': to_numeric_labels(
            dataset=train_df, 
            column='task1', 
            encoding=encoding),
        'encoded_test_labels': to_numeric_labels(
            dataset=test_df, 
            column='task1', 
            encoding=encoding)
    }
