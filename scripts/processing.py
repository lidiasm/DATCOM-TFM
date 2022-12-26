import re
import nltk
import spacy
import warnings
import pandas as pd
from sklearn import preprocessing
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from gensim.models import Word2Vec

# Ignore regex warnins
warnings.filterwarnings("ignore")

# Download a list of stopwords
nltk.download('stopwords')

# Get the spanish and engish stopwords
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
        .replace("@([a-zA-Z0-9_]{1,50})", "") 

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
        .replace("#([a-zA-Z0-9_]{1,50})", "")

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

def to_lemmatized_texts(dataset: pd.DataFrame, text_col: str, language_col: str):
    """
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
    language_col : str
        The column name which stores the language of each text.

    Returns
    -------
    A list of strings with the lemmatized texts.
    """
    # List to save the lemmatized texts
    lemm_texts = []

    # Create two lemmatizer for english and spanish texts
    en_lemmatizer = spacy.load("en_core_web_lg")
    es_lemmatizer = spacy.load("es_core_news_lg")

    # Iterate over the set of texts
    for record in dataset.to_records("dict"):

        # Select the lemmatizer depending on the text language
        lemmatizer = en_lemmatizer if record[language_col] == "en" else es_lemmatizer

        # String to save the current text when it's lemmatized
        current_lemm_text = ""

        # Iterate over the words to be lemmatized
        for word in lemmatizer(record[text_col]):
            current_lemm_text += word.lemma_ + " "
        
        # Add the lemmatized text to the dataset
        lemm_texts.append(current_lemm_text)
    
    return lemm_texts

def to_stemmed_texts(dataset: pd.DataFrame, text_col: str, language_col: str):
    """
    Iterates over a set of documents in order to replace each
    word with its root form. The stemmer to apply depends on the 
    text language, considering two languages: english and spanish.

    Parameters
    ----------
    dataset : Pandas dataframe
        The data which contains a set of texts.
    text_col : str
        The column name in which the set of texts is stored.
    language_col : str
        The column name which stores the language of each text.

    Returns
    -------
    A list of strings with the stemmed texts.
    """
    # List to save the stemmed texts
    stemm_texts = []

    # Create two stemmer for english and spanish texts
    en_stemmer = SnowballStemmer(language="english")
    es_stemmer = SnowballStemmer(language="spanish")

    # Iterate over the set of texts
    for record in dataset.to_records("dict"):

        # Select the stemmer depending on the text language
        stemmer = en_stemmer if record[language_col] == "en" else es_stemmer

        # String to save the current text when it's stemmed
        current_stemm_text = ""

        # Iterate over the words to be stemmed
        for word in record[text_col].split(" "):
            current_stemm_text += stemmer.stem(word) + " "
        
        # Add the stemmed text to the dataset
        stemm_texts.append(current_stemm_text)
    
    return stemm_texts

def text_processing_pipeline(dataset: pd.DataFrame, text_col: str, 
    lemm: bool = False, stemm: bool = False, language_col: str = ""):
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
    text_col : str
        The column name in which the set of texts is stored.
    language_col : str
        The column name in which the text languages are stored.

    Returns
    -------
    A Pandas dataframe with a new column to store the cleaned text.
    """
    # Column name to store the cleaned text
    cleaned_col = "cleaned_text"

    # Delete URLs
    dataset[cleaned_col] = delete_urls(dataset, text_col)

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

    # Lemmatize the texts
    if (lemm and type(language_col) == str and language_col != ""):
        dataset[cleaned_col] = to_lemmatized_texts(dataset, cleaned_col, language_col)

    # Text stemming
    if (stemm and type(language_col) == str and language_col != ""):
        dataset[cleaned_col] = to_stemmed_texts(dataset, cleaned_col, language_col)

    return dataset

def encode_to_numeric_labels(dataset: pd.DataFrame, column: str, encodings: dict = {}):
    """
    Replaces categorical labels stored in a specific column within
    a dataset with the provided list of numeric values or creating
    it using a LabelEncoder object.

    Parameters
    ----------
    dataset : Pandas dataframe
        A dataset which contains the class labels to encode.
    column : str
        The column name in which the class labels are stored.
    encodings : dict (optional)
        A dictionary which contains the categorical labels and
        their numeric labels associated.

    Returns
    -------
    A list with the encoded labels if a list of encodings is provided.
    Otherwise a dictionary whose keys are:
        - 'values': contains a list with the encoded values
        - 'classes': stores a dictionary with the links between 
                    the original and the encoded labels.
    """
    if len(encodings) > 0:
        # Replace the categorical labels with the numeric labels
        return [encodings[val] for val in list(dataset[column].values)]
    
    else:
        # Create a new encoder
        encoder = preprocessing.LabelEncoder()

        # Encode the categorical values to numeric
        encoded_values = encoder.fit(list(dataset[column].values))

        # Get the 
        encoded_classes = encoder.classes_
    
        return {
            "values": encoded_values,
            "classes": encoded_classes
        }

def to_bag_of_words(training_docs: list, testing_docs: list):
    """
    Creates a training and testing bag of words converting 
    them into numeric vectors by computing the word frequencies 
    per document. Both datasets are needed so both bags have 
    the same number of features.

    Parameters
    ----------
    training_docs : list
        A list of training documents.
    testing_docs : list
        A list of testing documents.

    Returns
    -------
    A dictionary whose keys are:
        - 'training': contains the bag of training words.
        - 'testing': contains the bag of testing
    """
    # Create a CountVectorizer object
    bg_vectorizer = CountVectorizer()

    # Train the object with the training dataset in order to then
    # encode both datasets
    return {
        "training": bg_vectorizer.fit_transform(training_docs),
        "testing": bg_vectorizer.transform(testing_docs).toarray()
    }

def to_tf_idf(training_docs: list, testing_docs: list):
    """
    Creates two lists of training and testing documents
    encoded after applying TF-IDF. This technique computes
    the absolute and relative frequencies to then select the
    most relevant terms for each document and the entire 
    population. Both datasets are needed so both lists have 
    the same number of features.

    Parameters
    ----------
    training_docs : list
        A list of training documents.
    testing_docs : list
        A list of testing documents.

    Returns
    -------
    A dictionary whose keys are:
        - 'training': contains a list with the training TF-IDF numbers.
        - 'testing': contains a list with the testing TF-IDF numbers.
    """
    # Create a TF-IDF vectorizer object
    tfidf_vectorizer = TfidfVectorizer()

    # Train the object with the training dataset in order to then
    # encode both datasets
    return {
        "training": tfidf_vectorizer.fit_transform(training_docs),
        "testing": tfidf_vectorizer.transform(testing_docs).toarray()
    }

def to_word_2_vec(docs: list, vector_size: int = 100, 
    window:int = 5, min_count: int = 5, epochs: int = 5, algorithm: int = 0):
    """
    Creates a Word2Vec model to train it using a provided list
    of documents with the goal of converting them into word
    embeddings. To do that each document is splitted into multiple
    words to then encode them as numeric vectors.

    Parameters
    ----------
    docs : list
        The list of text documents to encode.
    vector_size : int, optional (default 100)
        Size of the word embedding vectors.
    window : int, optional (default 5)
        Max distance between the current and predicted word 
        within a sentece.
    min_count : int, optional (default 5)
        Ignores words with a frequency lower than this value.
    epochs : int, optional (default 5)
        Number of iterations over the set of texts.
    algorithm : int, optional (default 0)
        Training algorithm: 0 for CBOW, 1 for Skip-Gram.

    Returns
    -------
    A list of Numpy ndarray with the word embeddings per document.
    """
    # Split each document into tokens (words) deleting the last whitespace
    tokens = [doc.split(" ") for doc in docs]

    # Create a Word2Vec model with a particular vector size
    w2v_model = Word2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=algorithm)

    # Build a set of terms to then train the model
    w2v_model.build_vocab(tokens)
    w2v_model.train(
        tokens, 
        total_examples=w2v_model.corpus_count,
        epochs=epochs)
    w2v_vocab = set(w2v_model.wv.index_to_key)

    # Create aggregated sentence vectors based on the tokens and Word2Vec vocabulary
    agg_sentences = np.array([np.array(
        [w2v_model.wv[word] for word in doc if word in w2v_vocab]) 
        for doc in tokens])

    # Normalize sentence vectors using the averaging of the word vectors 
    # for each sentence in order to then be used in ML models
    return [sent.mean(axis=0) if sent.size else np.zeros(vector_size, dtype=float) 
            for sent in agg_sentences] 

def word2vec_pipeline(
    training_df: pd.DataFrame, training_text_col: str,
    testing_df: pd.DataFrame, testing_text_col: str, 
    vector_size: int = 100, window:int = 5, min_count: int = 5, epochs: int = 5, algorithm: int = 0):
    """
    Creates a training and a testing datasets based on the
    Word2Vec technique to encode a set of texts as word embeddings.
    These datasets are aimed to be used directly in the building 
    Machine Learning models.

    Parameters
    ----------
    training_df : Pandas dataframe
        A training dataset to encode.
    training_text_col : str
        A column name in which there are the set of training texts to encode.
    testing_df : Pandas dataframe
        A testing dataset to encode.
    testing_text_col : str
        A column name in which there are the set of testing texts to encode.
    vector_size : int, optional (default 100)
        Size of the word embedding vectors.
    window : int, optional (default 5)
        Max distance between the current and predicted word 
        within a sentece.
    min_count : int, optional (default 5)
        Ignores words with a frequency lower than this value.
    epochs : int, optional (default 5)
        Number of iterations over the set of texts.
    algorithm : int, optional (default 0)
        Training algorithm: 0 for CBOW, 1 for Skip-Gram.

    Returns
    -------
    A dictionary whose keys are:
        - 'training': contains a training dataset encoded through Word2Vec.
        - 'testing': contains a testing dataset encoded through Word2Vec.
    """
    # Create the training and testing Word2Vec embeddings
    training_w2v_embeddings = to_word_2_vec(
        list(training_df[training_text_col].values),
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        epochs=epochs,
        algorithm=algorithm
    )
    testing_w2v_embeddings = to_word_2_vec(
        list(testing_df[testing_text_col].values),
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        epochs=epochs,
        algorithm=algorithm
    )

    # Create a training and a testing datasets adding names to
    # the new created columns
    training_w2v_df = pd.DataFrame(training_w2v_embeddings)
    training_w2v_df.columns = [f"Feature {index+1}" for index in range(0, training_w2v_df.shape[1])]

    testing_w2v_df = pd.DataFrame(testing_w2v_embeddings)
    testing_w2v_df.columns = [f"Feature {index+1}" for index in range(0, testing_w2v_df.shape[1])]

    return {
        "training": training_w2v_df,
        "testing": testing_w2v_df
    }