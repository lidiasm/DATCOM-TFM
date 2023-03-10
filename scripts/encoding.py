import numpy as np
import pandas as pd 
import gensim.downloader as api
from gensim.models import Word2Vec
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def to_bag_of_words(train_docs: list, test_docs: list):
    '''
    Creates a bag of word for a train and test texts by
    converting them into numeric vectors by computing the
    word frequencies per document. 

    Parameters
    ----------
    train_docs : list
        A list of train documents to encode.
    test_docs : list
        A list of test documents to encode.

    Returns
    -------
    A tuple with the train and test bags of words.
    '''
    bg_vectorizer = CountVectorizer()
    
    return bg_vectorizer.fit_transform(raw_documents=train_docs).toarray(), \
        bg_vectorizer.transform(raw_documents=test_docs).toarray()


def to_tf_idf(train_docs: list, test_docs: list):
    '''
    Creates a TF-IDF vector for a train and test texts by
    calculating the absolute and relative frequencies so
    the most relevant terms can represent the documents.

    Parameters
    ----------
    train_docs : list
        A list of train documents to encode.
    test_docs : list
        A list of test documents to encode.

    Returns
    -------
    A tuple with the train and test TF-IDF vectors.
    '''
    tfidf_vectorizer = TfidfVectorizer()

    return tfidf_vectorizer.fit_transform(raw_documents=train_docs), \
        tfidf_vectorizer.transform(raw_documents=test_docs).toarray()


def to_word_2_vec(
    docs: list, vector_size: int = 100, window: int = 5, 
    min_count: int = 5, epochs: int = 5, algorithm: int = 0):
    '''
    Creates a word-embedding vector for a train and test texts
    by training a Word2Vec model able to encode the splitted
    texts into words to numeric vectors. 
    
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
    # Split each document into tokens (words)
    tokens = [doc.split(' ') for doc in docs]

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
    Creates a train and a test datasets based on the Word2Vec
    technique to encode a set of texts as word embeddings.
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
    A tuple with the train and test word-embeddings. 
    '''
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

    train_w2v_df = pd.DataFrame(data=train_w2v_embeddings)
    train_w2v_df.columns = [
        f'Feature {index+1}' for index in range(0, train_w2v_df.shape[1])
    ]

    test_w2v_df = pd.DataFrame(data=test_w2v_embeddings)
    test_w2v_df.columns = [
        f'Feature {index+1}' for index in range(0, test_w2v_df.shape[1])
    ]

    return train_w2v_df, test_w2v_df


def to_doc_2_vec(
    docs: list, vector_size: int = 100, window: int = 5, 
    min_count: int = 5, epochs: int = 5, algorithm: int = 0):
    '''
    Creates a word-embedding vector for a train and test texts
    by training a Doc2Vec model able to encode the splitted
    documents into words and identifiers to numeric vectors. 

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
    # Split each document into tokens (words)
    tokens = [doc.split(' ') for doc in docs]

    # Create another vector for the documents
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(tokens)]

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
    Creates a train and a test datasets based on the Doc2Vec
    technique to encode a set of texts as word and doc embeddings.
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
    A tuple with the train and test doc-embeddings.
    '''
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

    train_d2v_df = pd.DataFrame(data=train_d2v_embeddings)
    train_d2v_df.columns = [
        f'Feature {index+1}' for index in range(0, train_d2v_df.shape[1])
    ]

    test_d2v_df = pd.DataFrame(data=test_d2v_embeddings)
    test_d2v_df.columns = [
        f'Feature {index+1}' for index in range(0, test_d2v_df.shape[1])
    ]

    return train_d2v_df, test_d2v_df


def to_pretrained_embeddings(
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
    A tuple with two Pandas dataframes that contain the encoded
    train and text documents based on the pretrained embeddings.
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

    return train_embeddings_df, test_embeddings_df