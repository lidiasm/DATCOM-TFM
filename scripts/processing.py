import os
import re
import nltk
import spacy
import warnings
import pandas as pd
from sklearn import preprocessing
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from gensim.models import Word2Vec, FastText
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
import gensim.downloader as api
from sklearn.model_selection import train_test_split

# Ignore regex warnins
warnings.filterwarnings("ignore")

# Download a list of stopwords
nltk.download('stopwords')

# Get the spanish and engish stopwords
STOPWORDS = nltk.corpus.stopwords.words("spanish") + \
            nltk.corpus.stopwords.words("english")

# Rate to split the train dataset into a train and a validation dataset
TRAIN_RATE = 0.8

# Paths to the processed train, validation and test datasets
TRAIN_DF_PATH = "../data/processed_EXIST2021_train.csv"
VALID_DF_PATH = "../data/processed_EXIST2021_valid.csv"
TEST_DF_PATH = "../data/processed_EXIST2021_test.csv"

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
    return dataset[column].apply(lambda x: re.split(
        pattern="https:\/\/.*",
        string=str(x))[0])


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
    return dataset[column].str.replace(
        pat="@([a-zA-Z0-9_]{1,50})",
        repl="")


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
    return dataset[column].str.replace(
        pat="#([a-zA-Z0-9_]{1,50})", 
        repl="")


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
    return dataset[column].str.replace(
        pat="[^a-zA-Z ]", 
        repl="")


def delete_stopwords(dataset: pd.DataFrame, column: str, add_stopwords: list = []):
    """
    Removes stopwords from a set of english and
    spanish texts using the NLTK stopword list.

    Parameters
    ----------
    dataset : Pandas dataframe
        The data which contains a set of texts.
    column : str
        The column name in which the set of texts is stored.
    add_stopwords : list
        A list of additional stopwords to exclude.

    Returns
    -------
    A Series column
    """
    STOPWORDS.extend(add_stopwords)

    return dataset[column].apply(lambda x: " ".join([
        word for word in x.split() if word not in STOPWORDS and len(word) > 1
    ]))


def delete_wrong_words(dataset: pd.DataFrame, column: str):
    """
    Removes words composed of one letter or of only consonants
    without vowels within a set of texts.

    Parameters
    ----------
    dataset : Pandas dataframe
        The data which contains a set of texts.
    column : str
        The column name in which the set of texts is stored.

    Returns
    -------
    A Series column with the processed text
    """
    vowset = set(["a", "e", "i", "o", "u"])
    return dataset[column].apply(lambda x: " ".join([
        word for word in x.split() if len(word) > 1 and len(vowset.intersection(word)) > 0
    ]))

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
                            add_stopwords: list = [],
                            lemm: bool = False, stemm: bool = False, 
                            language_col: str = ""):
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
    add_stopwords : list
        A list of additional stopwords to exclude.
    lemm : bool
        True to lemmatize the texts, False to not apply it.
    stemm : bool
        True to apply stemming to the texts, False to not apply it.
    language_col : str
        The column name in which the text languages are stored.

    Returns
    -------
    A Pandas dataframe with a new column to store the cleaned text.
    """
    # Column name to store the cleaned text
    cleaned_col = "cleaned_text"

    # Delete URLs
    dataset[cleaned_col] = delete_urls(
        dataset=dataset, 
        column=text_col)

    # Delete mentioned users in tweets
    dataset[cleaned_col] = delete_mentioned_users(
        dataset=dataset, 
        column=cleaned_col)

    # Delete hashtags in tweets
    dataset[cleaned_col] = delete_hashtags(
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

    # Lemmatize the texts
    if (lemm and type(language_col) == str and language_col != ""):
        dataset[cleaned_col] = to_lemmatized_texts(
            dataset=dataset, 
            text_col=cleaned_col, 
            language_col=language_col)

    # Text stemming
    if (stemm and type(language_col) == str and language_col != ""):
        dataset[cleaned_col] = to_stemmed_texts(
            dataset=dataset, 
            text_col=cleaned_col, 
            language_col=language_col)

    return dataset


def to_numeric_labels(dataset: pd.DataFrame, column: str, encoding: dict = {}):
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
    """
    if len(encoding) > 0:
        # Replace the categorical labels with the numeric labels
        return [encoding[val] for val in list(dataset[column].values)]

    else:
        # Create a new encoder
        encoder = preprocessing.LabelEncoder()

        # Encode the categorical values to numeric
        encoded_values = encoder.fit(y=list(dataset[column].values))

        return {
            "values": encoded_values,
            "classes": encoder.classes_
        }

def process_encode_datasets(
    train_df: pd.DataFrame, test_df: pd.DataFrame,
    lemm: bool, stemm: bool):
    """
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

    Returns
    -------
    A dictionary with the processed data and the encoded labels.
        - 'train_df': a Pandas dataframe with the processed train texts.
        - 'test_df': a Pandas dataframe with the processed test texts.
        - 'encoded_train_labels': a list of numeric values with the encoded train labels.
        - 'encoded_test_labels': a list of numeric values with the encoded test labels.
    """
    # Specific encoding for class labels
    encoding = {
        "non-sexist": 0, 
        "sexist": 1
    }

    return {
        "train_df": text_processing_pipeline(
            dataset=train_df, 
            text_col="text", 
            lemm=lemm,
            stemm=stemm, 
            language_col="language"),
        "test_df":text_processing_pipeline(
            dataset=test_df, 
            text_col="text", 
            lemm=lemm,
            stemm=stemm, 
            language_col="language"),
        "encoded_train_labels": to_numeric_labels(
            dataset=train_df, 
            column="task1", 
            encoding=encoding),
        "encoded_test_labels": to_numeric_labels(
            dataset=test_df, 
            column="task1", 
            encoding=encoding)
    }

def get_train_valid_test_df(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Process the train and test datasets and encode the class
    labels as numeric values returning a train dataset splitted
    into 80% for training and 20% for validation, as well as the
    entire test dataset. 
    This function will be used to build LSTM models using Pytorch library.

    Parameters
    ----------
    train_df : Pandas dataframe
        A dataset which contains the train samples.
    test_df : Pandas dataframe
        A dataset which contains the test samples.

    Returns
    -------
    A dictionary with the processed data and the encoded labels.
        - 'train_df': a Pandas dataframe with the 80% of processed train texts.
        - 'valid_df': a Pandas dataframe with the 20% of processed train texts.
        - 'test_df': a Pandas dataframe with the 100% of processed test texts.
    """
    # Read the train, validation and test datasets if they
    # are contained in files
    if (os.path.exists(TRAIN_DF_PATH) and 
        os.path.exists(VALID_DF_PATH) and 
        os.path.exists(TEST_DF_PATH)):
        return {
            "train_df": pd.read_csv(TRAIN_DF_PATH),
            "valid_df": pd.read_csv(VALID_DF_PATH),
            "test_df": pd.read_csv(TEST_DF_PATH)
        }

    # Create the train, validation and test datasets to then
    # save them in files
    else:
        # Process documents and encode class labels as numeric labels
        processed_data = process_encode_datasets(
            train_df=train_df,
            test_df=test_df,
            lemm=False, 
            stemm=True)

        # Split the train dataset into train and validation datasets
        processed_data["train_df"]["num_task1"] = processed_data["encoded_train_labels"]
        train_valid_dfs = train_test_split(
            processed_data["train_df"], 
            train_size=TRAIN_RATE, 
            random_state=1) # Pass an integer to get duplicable results

        # Save the procesed train dataset in a file
        processed_train_df = pd.DataFrame(train_valid_dfs[0])
        processed_train_df.to_csv(TRAIN_DF_PATH)

        # Save the procesed validation dataset in a file
        processed_valid_df = pd.DataFrame(train_valid_dfs[1])
        processed_valid_df.to_csv(VALID_DF_PATH)

        # Save the procesed test dataset in a file
        processed_data["test_df"]["num_task1"] = processed_data["encoded_test_labels"]
        processed_test_df = processed_data["test_df"]
        processed_test_df.to_csv(TEST_DF_PATH)
        
        return {
            "train_df": processed_train_df,
            "valid_df": processed_valid_df,
            "test_df": processed_test_df
        }

def to_bag_of_words(train_docs: list, test_docs: list):
    """
    Creates a train and test bag of words converting 
    them into numeric vectors by computing the word frequencies 
    per document. Both datasets are needed so both bags have 
    the same number of features.

    Parameters
    ----------
    train_docs : list
        A list of train documents.
    test_docs : list
        A list of test documents.

    Returns
    -------
    A dictionary whose keys are:
        - 'train': contains the bag of train words.
        - 'test': contains the bag of test
    """
    # Create a CountVectorizer object
    bg_vectorizer = CountVectorizer()

    # Train the object with the train dataset in order to then
    # encode both datasets
    return {
        "train": bg_vectorizer.fit_transform(raw_documents=train_docs),
        "test": bg_vectorizer.transform(raw_documents=test_docs).toarray()
    }


def to_tf_idf(train_docs: list, test_docs: list):
    """
    Creates two lists of train and test documents
    encoded after applying TF-IDF. This technique computes
    the absolute and relative frequencies to then select the
    most relevant terms for each document and the entire 
    population. Both datasets are needed so both lists have 
    the same number of features.

    Parameters
    ----------
    train_docs : list
        A list of train documents.
    test_docs : list
        A list of test documents.

    Returns
    -------
    A dictionary whose keys are:
        - 'train': contains a list with the train TF-IDF numbers.
        - 'test': contains a list with the test TF-IDF numbers.
    """
    # Create a TF-IDF vectorizer object
    tfidf_vectorizer = TfidfVectorizer()

    # Train the object with the train dataset in order to then
    # encode both datasets
    return {
        "train": tfidf_vectorizer.fit_transform(raw_documents=train_docs),
        "test": tfidf_vectorizer.transform(raw_documents=test_docs).toarray()
    }


def to_word_2_vec(docs: list, vector_size: int = 100,
                  window: int = 5, min_count: int = 5, 
                  epochs: int = 5, algorithm: int = 0):
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
        train algorithm: 0 for CBOW, 1 for Skip-Gram.

    Returns
    -------
    A list of Numpy ndarray with the word embeddings per document.
    """
    # Split each document into tokens (words) deleting the last whitespace
    tokens = [doc.split(" ") for doc in docs]

    # Create a Word2Vec model with a particular vector size
    w2v_model = Word2Vec(
        sentences=tokens,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=algorithm,
        epochs=epochs)

    # Build a set of terms to then train the model
    w2v_vocab = set(w2v_model.wv.index_to_key)

    # Create aggregated sentence vectors based on the tokens and Word2Vec vocabulary
    agg_sentences = np.array([np.array(
        [w2v_model.wv[word] for word in doc if word in w2v_vocab])
        for doc in tokens])

    # Normalize sentence vectors using the averaging of the word vectors
    # for each sentence in order to then be used in ML models
    return [
        sent.mean(axis=0) if sent.size else np.zeros(vector_size, dtype=float)
        for sent in agg_sentences
    ]


def word2vec_pipeline(
        train_df: pd.DataFrame, train_text_col: str,
        test_df: pd.DataFrame, test_text_col: str,
        vector_size: int = 100, window: int = 5, min_count: int = 5, 
        epochs: int = 5, alg: int = 0):
    """
    Creates a train and a test datasets based on the
    Word2Vec technique to encode a set of texts as word embeddings.
    These datasets are aimed to be used directly in the building of
    Machine Learning models.

    Parameters
    ----------
    train_df : Pandas dataframe
        A train dataset to encode.
    train_text_col : str
        A column name in which there are the set of train texts to encode.
    test_df : Pandas dataframe
        A test dataset to encode.
    test_text_col : str
        A column name in which there are the set of test texts to encode.
    vector_size : int, optional (default 100)
        Size of the word embedding vectors.
    window : int, optional (default 5)
        Max distance between the current and predicted word 
        within a sentece.
    min_count : int, optional (default 5)
        Ignores words with a frequency lower than this value.
    epochs : int, optional (default 5)
        Number of iterations over the set of texts.
    alg : int, optional (default 0)
        train algorithm: 0 for CBOW, 1 for Skip-Gram.

    Returns
    -------
    A dictionary whose keys are:
        - 'train': contains a train dataset encoded through Word2Vec.
        - 'test': contains a test dataset encoded through Word2Vec.
    """
    # Create the train and test Word2Vec embeddings
    train_w2v_embeddings = to_word_2_vec(
        docs=list(train_df[train_text_col].values),
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        epochs=epochs,
        algorithm=alg
    )
    test_w2v_embeddings = to_word_2_vec(
        docs=list(test_df[test_text_col].values),
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        epochs=epochs,
        algorithm=alg
    )

    # Create a train and a test datasets adding names to
    # the new created columns
    train_w2v_df = pd.DataFrame(data=train_w2v_embeddings)
    train_w2v_df.columns = [
        f"Feature {index+1}" for index in range(0, train_w2v_df.shape[1])
    ]

    test_w2v_df = pd.DataFrame(data=test_w2v_embeddings)
    test_w2v_df.columns = [
        f"Feature {index+1}" for index in range(0, test_w2v_df.shape[1])
    ]

    return {
        "train": train_w2v_df,
        "test": test_w2v_df
    }


def to_doc_2_vec(docs: list, vector_size: int = 100,
                window: int = 5, min_count: int = 5, 
                epochs: int = 5, algorithm: int = 0):
    """
    Creates a Doc2Vec model to train it using a provided list
    of documents with the goal of converting each word to embeddings along
    with the documents themselves. To do that each document is splitted 
    into multiple words and documents to then encode them as numeric vectors.

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
        train algorithm: 0 for PV-DBOW, 1 for PV-DM.

    Returns
    -------
    A list of Numpy ndarray with the word and document embeddings.
    """
    # Split each document into tokens (words) deleting the last whitespace
    tokens = [doc.split(" ") for doc in docs]

    # Create another vector for the documents
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(tokens)]

    # Create and train a Word2Vec model with a particular vector size
    d2v_model = Doc2Vec(
        documents=documents,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        dm=algorithm,
        epochs=epochs)

    # Build a set of terms to then train the model
    d2v_vocab = set(d2v_model.wv.index_to_key)

    # Create aggregated sentence vectors based on the tokens and Word2Vec vocabulary
    agg_sentences = np.array([
            np.array([
                d2v_model.wv[word] for word in doc if word in d2v_vocab
            ])
        for doc in tokens
    ])

    # Normalize sentence vectors using the averaging of the word vectors
    # for each sentence in order to then be used in ML models
    return [
        sent.mean(axis=0) if sent.size else np.zeros(vector_size, dtype=float)
        for sent in agg_sentences
    ]


def doc2vec_pipeline(
        train_df: pd.DataFrame, train_text_col: str,
        test_df: pd.DataFrame, test_text_col: str,
        vector_size: int = 100, window: int = 5, 
        min_count: int = 5, epochs: int = 5, alg: int = 0):
    """
    Creates a train and a test datasets based on the
    Doc2Vec technique to encode a set of texts as word and doc embeddings.
    These datasets are aimed to be used directly in the building of
    Machine Learning models.

    Parameters
    ----------
    train_df : Pandas dataframe
        A train dataset to encode.
    train_text_col : str
        A column name in which there are the set of train texts to encode.
    test_df : Pandas dataframe
        A test dataset to encode.
    test_text_col : str
        A column name in which there are the set of test texts to encode.
    vector_size : int, optional (default 100)
        Size of the word embedding vectors.
    window : int, optional (default 5)
        Max distance between the current and predicted word 
        within a sentece.
    min_count : int, optional (default 5)
        Ignores words with a frequency lower than this value.
    epochs : int, optional (default 5)
        Number of iterations over the set of texts.
    alg : int, optional (default 0)
        train algorithm: 0 for CBOW, 1 for Skip-Gram.

    Returns
    -------
    A dictionary whose keys are:
        - 'train': contains a train dataset encoded through Doc2Vec.
        - 'test': contains a test dataset encoded through Doc2Vec.
    """
    # Create the train and test Doc2Vec embeddings
    train_d2v_embeddings = to_doc_2_vec(
        docs=list(train_df[train_text_col].values),
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        epochs=epochs,
        algorithm=alg
    )
    test_d2v_embeddings = to_doc_2_vec(
        docs=list(test_df[test_text_col].values),
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        epochs=epochs,
        algorithm=alg
    )

    # Create a train and a test datasets adding names to
    # the new created columns
    train_d2v_df = pd.DataFrame(data=train_d2v_embeddings)
    train_d2v_df.columns = [
        f"Feature {index+1}" for index in range(0, train_d2v_df.shape[1])
    ]

    test_d2v_df = pd.DataFrame(data=test_d2v_embeddings)
    test_d2v_df.columns = [
        f"Feature {index+1}" for index in range(0, test_d2v_df.shape[1])
    ]

    return {
        "train": train_d2v_df,
        "test": test_d2v_df
    }


def to_fast_text(docs: list, vector_size: int = 100,
                window: int = 5, min_count: int = 5, 
                epochs: int = 5, algorithm: int = 0):
    """
    Creates a FastText model to train it using a provided list
    of documents with the goal of converting them into word
    embeddings. ---------------------------------

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
        train algorithm: 0 for CBOW, 1 for Skip-Gram.

    Returns
    -------
    A list of Numpy ndarray with the word embeddings per document.
    """
    # Split each document into tokens (words) deleting the last whitespace
    tokens = [doc.split(" ") for doc in docs]

    # Create a FastText model with a particular vector size
    ft_model = FastText(
        sentences=tokens,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=algorithm,
        epochs=epochs)

    # Build a set of terms to then train the model
    ft_vocab = set(ft_model.wv.index_to_key)

    # Create aggregated sentence vectors based on the tokens and FastText vocabulary
    agg_sentences = np.array([np.array(
        [
            ft_model.wv[word] for word in doc if word in ft_vocab
        ])
        for doc in tokens
    ])

    # Normalize sentence vectors using the averaging of the word vectors
    # for each sentence in order to then be used in ML models
    return [
        sent.mean(axis=0) if sent.size else np.zeros(vector_size, dtype=float)
        for sent in agg_sentences
    ]


def fasttext_pipeline(
        train_df: pd.DataFrame, train_text_col: str,
        test_df: pd.DataFrame, test_text_col: str,
        vector_size: int = 100, window: int = 5, 
        min_count: int = 5, epochs: int = 5, alg: int = 0):
    """
    Creates a train and a test datasets based on the
    FastText technique to encode a set of texts as word embeddings.
    These datasets are aimed to be used directly in the building of
    Machine Learning models.

    Parameters
    ----------
    train_df : Pandas dataframe
        A train dataset to encode.
    train_text_col : str
        A column name in which there are the set of train texts to encode.
    test_df : Pandas dataframe
        A test dataset to encode.
    test_text_col : str
        A column name in which there are the set of test texts to encode.
    vector_size : int, optional (default 100)
        Size of the word embedding vectors.
    window : int, optional (default 5)
        Max distance between the current and predicted word 
        within a sentece.
    min_count : int, optional (default 5)
        Ignores words with a frequency lower than this value.
    epochs : int, optional (default 5)
        Number of iterations over the set of texts.
    alg : int, optional (default 0)
        train algorithm: 0 for CBOW, 1 for Skip-Gram.

    Returns
    -------
    A dictionary whose keys are:
        - 'train': contains a train dataset encoded through FastText.
        - 'test': contains a test dataset encoded through FastText.
    """
    # Create the train and test FastText embeddings
    train_ft_embeddings = to_fast_text(
        docs=list(train_df[train_text_col].values),
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        epochs=epochs,
        algorithm=alg
    )
    test_ft_embeddings = to_fast_text(
        docs=list(test_df[test_text_col].values),
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        epochs=epochs,
        algorithm=alg
    )

    # Create a train and a test datasets adding names to
    # the new created columns
    train_ft_df = pd.DataFrame(data=train_ft_embeddings)
    train_ft_df.columns = [
        f"Feature {index+1}" for index in range(0, train_ft_df.shape[1])
    ]

    test_ft_df = pd.DataFrame(data=test_ft_embeddings)
    test_ft_df.columns = [
        f"Feature {index+1}" for index in range(0, test_ft_df.shape[1])
    ]

    return {
        "train": train_ft_df,
        "test": test_ft_df
    }


def trained_embeddings_pipeline(
        train_df: pd.DataFrame, train_text_col: str,
        test_df: pd.DataFrame, test_text_col: str,
        model: str, vector_size: int = 100):
    """
    Downloads the provided pre-trained model to then load
    its pre-trained embeddings ready to use. Following models 
    are available from `gensim` library:

        - 'fasttext-wiki-news-subwords-300'
        - 'conceptnet-numberbatch-17-06-300'
        - 'word2vec-ruscorpora-300'
        - 'word2vec-google-news-300'
        - 'glove-wiki-gigaword-50'
        - 'glove-wiki-gigaword-100'
        - 'glove-wiki-gigaword-200'
        - 'glove-wiki-gigaword-300'
        - 'glove-twitter-25'
        - 'glove-twitter-50'
        - 'glove-twitter-100'
        - 'glove-twitter-200'

    Parameters
    ----------
    train_df : Pandas dataframe
        A train dataset to encode.
    train_text_col : str
        A column name in which there are the set of train texts to encode.
    test_df : Pandas dataframe
        A test dataset to encode.
    test_text_col : str
        A column name in which there are the set of test texts to encode.
    model : str
        The name of the pre-trained model to download and load its embeddings.
    vector_size : int, optional (default 100)
        Size of the word embedding vectors.

    Returns
    -------
    A dictionary whose keys are:
        - 'train': contains a train dataset encoded through the chosen model.
        - 'test': contains a test dataset encoded through the chosen model.
    """
    # Download the pre-trained model if it's not been downloaded before
    pretrained_model = api.load(name=model)

    # Build a set of terms to then train the model
    pretrained_vocab = set(pretrained_model.index_to_key)

    # Split the train and test documents into tokens
    train_tokens = [
        doc.split(" ") for doc in list(train_df[train_text_col].values)
    ]
    test_tokens = [
        doc.split(" ") for doc in list(test_df[test_text_col].values)
    ]

    # Create aggregated sentence vectors based on the tokens and the pre-trained vocabulary
    train_agg_sentences = np.array([np.array(
        [
            pretrained_model[word] for word in doc if word in pretrained_vocab
        ])
        for doc in train_tokens
    ])
    test_agg_sentences = np.array([np.array(
        [
            pretrained_model[word] for word in doc if word in pretrained_vocab
        ])
        for doc in test_tokens
    ])

    # Normalize sentence vectors using the averaging of the word vectors
    # for each sentence in order to then be used in ML models
    train_avg_sentences = [
        sent.mean(axis=0) if sent.size else np.zeros(vector_size, dtype=float) 
        for sent in train_agg_sentences
    ]
    test_avg_sentences = [
        sent.mean(axis=0) if sent.size else np.zeros(vector_size, dtype=float) 
        for sent in test_agg_sentences
    ]

    # Create a train and a test datasets with column names
    train_embeddings_df = pd.DataFrame(data=train_avg_sentences)
    train_embeddings_df.columns = [
        f"Feature {index+1}" for index in range(0, train_embeddings_df.shape[1])
    ]

    test_embeddings_df = pd.DataFrame(data=test_avg_sentences)
    test_embeddings_df.columns = [
        f"Feature {index+1}" for index in range(0, test_embeddings_df.shape[1])
    ]

    return {
        "train": train_embeddings_df,
        "test": test_embeddings_df
    }
