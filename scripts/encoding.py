import numpy as np
import pandas as pd 
import gensim.downloader as api
from gensim.models import Word2Vec, FastText
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
    
    return bg_vectorizer.fit_transform(raw_documents=train_docs), \
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


def to_fast_text(
    docs: list, vector_size: int = 100, window: int = 5, 
    min_count: int = 5, epochs: int = 5, algorithm: int = 0):
    '''
    Creates a word-embedding vector for a train and test texts
    by training a FastText model able to encode the splitted
    documents into words to numeric vectors. 

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
    Creates a train and a test datasets based on the FastText
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
    A tuple with the train and test FastText word-embeddings.
    '''
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

    train_ft_df = pd.DataFrame(data=train_ft_embeddings)
    train_ft_df.columns = [
        f'Feature {index+1}' for index in range(0, train_ft_df.shape[1])
    ]

    test_ft_df = pd.DataFrame(data=test_ft_embeddings)
    test_ft_df.columns = [
        f'Feature {index+1}' for index in range(0, test_ft_df.shape[1])
    ]

    return train_ft_df, test_ft_df