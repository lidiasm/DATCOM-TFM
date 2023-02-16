import numpy as np
import pandas as pd 
import gensim.downloader as api
from gensim.models import Word2Vec, FastText
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def to_bag_of_words(train_docs: list, test_docs: list):
    '''
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
    '''
    # Create a CountVectorizer object
    bg_vectorizer = CountVectorizer()

    # Train the object with the train dataset in order to then
    # encode both datasets
    return {
        'train': bg_vectorizer.fit_transform(raw_documents=train_docs),
        'test': bg_vectorizer.transform(raw_documents=test_docs).toarray()
    }


def to_tf_idf(train_docs: list, test_docs: list):
    '''
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
    '''
    # Create a TF-IDF vectorizer object
    tfidf_vectorizer = TfidfVectorizer()

    # Train the object with the train dataset in order to then
    # encode both datasets
    return {
        'train': tfidf_vectorizer.fit_transform(raw_documents=train_docs),
        'test': tfidf_vectorizer.transform(raw_documents=test_docs).toarray()
    }


def to_word_2_vec(docs: list, vector_size: int = 100,
                  window: int = 5, min_count: int = 5, 
                  epochs: int = 5, algorithm: int = 0):
    '''
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
    '''
    # Split each document into tokens (words) deleting the last whitespace
    tokens = [doc.split(' ') for doc in docs]

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
    '''
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
    '''
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
        f'Feature {index+1}' for index in range(0, train_w2v_df.shape[1])
    ]

    test_w2v_df = pd.DataFrame(data=test_w2v_embeddings)
    test_w2v_df.columns = [
        f'Feature {index+1}' for index in range(0, test_w2v_df.shape[1])
    ]

    return {
        'train': train_w2v_df,
        'test': test_w2v_df
    }


def to_doc_2_vec(docs: list, vector_size: int = 100,
                window: int = 5, min_count: int = 5, 
                epochs: int = 5, algorithm: int = 0):
    '''
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
    '''
    # Split each document into tokens (words) deleting the last whitespace
    tokens = [doc.split(' ') for doc in docs]

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
    '''
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
    '''
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
        f'Feature {index+1}' for index in range(0, train_d2v_df.shape[1])
    ]

    test_d2v_df = pd.DataFrame(data=test_d2v_embeddings)
    test_d2v_df.columns = [
        f'Feature {index+1}' for index in range(0, test_d2v_df.shape[1])
    ]

    return {
        'train': train_d2v_df,
        'test': test_d2v_df
    }


def to_fast_text(docs: list, vector_size: int = 100,
                window: int = 5, min_count: int = 5, 
                epochs: int = 5, algorithm: int = 0):
    '''
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
    '''
    # Split each document into tokens (words) deleting the last whitespace
    tokens = [doc.split(' ') for doc in docs]

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
    '''
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
    '''
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
        f'Feature {index+1}' for index in range(0, train_ft_df.shape[1])
    ]

    test_ft_df = pd.DataFrame(data=test_ft_embeddings)
    test_ft_df.columns = [
        f'Feature {index+1}' for index in range(0, test_ft_df.shape[1])
    ]

    return {
        'train': train_ft_df,
        'test': test_ft_df
    }


def trained_embeddings_pipeline(
        train_df: pd.DataFrame, train_text_col: str,
        test_df: pd.DataFrame, test_text_col: str,
        model: str, vector_size: int = 100):
    '''
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
    '''
    # Download the pre-trained model if it's not been downloaded before
    pretrained_model = api.load(name=model)

    # Build a set of terms to then train the model
    pretrained_vocab = set(pretrained_model.index_to_key)

    # Split the train and test documents into tokens
    train_tokens = [
        doc.split(' ') for doc in list(train_df[train_text_col].values)
    ]
    test_tokens = [
        doc.split(' ') for doc in list(test_df[test_text_col].values)
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
        f'Feature {index+1}' for index in range(0, train_embeddings_df.shape[1])
    ]

    test_embeddings_df = pd.DataFrame(data=test_avg_sentences)
    test_embeddings_df.columns = [
        f'Feature {index+1}' for index in range(0, test_embeddings_df.shape[1])
    ]

    return {
        'train': train_embeddings_df,
        'test': test_embeddings_df
    }
