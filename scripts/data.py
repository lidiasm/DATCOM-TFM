import pandas as pd
import nltk
from transformers import AutoTokenizer, AutoModelWithLMHead


def read_train_dataset():
    '''
    Function that reads the EXIST train file to load the 
    train dataset as dataframe.

    Returns
    -------
    A Pandas dataframe
    '''
    return pd.read_table('../data/EXIST2021_train.tsv')


def read_test_dataset():
    '''
    Function that reads the EXIST test file to load the 
    test dataset as dataframe.

    Returns
    -------
    A Pandas dataframe
    '''
    return pd.read_table('../data/EXIST2021_test.tsv')


def count_words(
    dataset: pd.DataFrame, text_col: str, 
    tokenize: bool = False, unique_words: bool = False):
    '''
    Function that computes the total number of words 
    or calculates the amount of unique words from the texts
    stored in the provided dataset, tokenizing them if desired.

    Parameters
    ----------
    dataset : Pandas dataframe
        The data which contains a set of texts.
    text_col : str
        The column name in which the set of texts is stored.
    tokenize : bool (optional, default False)
        True to split the documents into words without whitespaces
        and punctuation marks.
    unique_words : bool (optional, default False)
        True to only count unique words, False to take into account all of them.

    Returns
    -------
    An integer with the total number of words.
    '''
    # Delete punctuation marks and split the docs into words
    if (tokenize):
        dataset[text_col] = [
            record.split(' ') for record in 
            list(dataset[text_col].str.replace('[^\w\s]', ''))
        ]

    word_count = 0    
    if (unique_words):
        for doc in dataset[text_col]:
            word_count += len(set(doc))
        
    else:
        for doc in dataset[text_col]:
            word_count += len(doc)

    return word_count


def get_top_ngrams(dataset: pd.DataFrame, column: str, n_words: int):
    '''
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
    A Pandas dataframe with the most frequent N grams and their
    frequency value sorted by it.
    '''
    # Split texts in sentences of N words
    dataset['ngrams'] = dataset[column].str.split() \
        .apply(lambda x: list(map(' '.join, nltk.ngrams(x, n=n_words))))

    # Compute the frequency per sentece
    return (dataset.assign(count=dataset['ngrams']\
        .str.len()).explode('ngrams')) \
        .sort_values('count', ascending=False)


def get_emotions_from_texts(dataset: pd.DataFrame, text_col: str):
    '''
    Function that assigns an emotion to each text stored in
    a provided dataset using a tokenizer and a pre-trained
    Transformer model from Hugging Face.
    

    Parameters
    ----------
    dataset : Pandas dataframe
        A dataset which contains the set of texts to analyze.
    text_col : str
        The column name in which the set of texts is stored.

    Returns
    -------
    A Pandas dataframe with the analyzed texts, the detected
    emotions and the dependent variables.
    '''
    # Download a tokenizer and a Transformer pre-trained model
    # for emotion detection from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained('mrm8488/t5-base-finetuned-emotion')
    model = AutoModelWithLMHead.from_pretrained('mrm8488/t5-base-finetuned-emotion')

    emotion_dataset = {
        text_col: list(dataset[text_col].values),
        'task1': list(dataset['task1'].values),
        'task2': list(dataset['task2'].values),
        'emotion': []
    }

    dataset_dict = dataset.to_dict('records')

    for record in dataset_dict:
        # Encode the input to pass it to the Transformer model
        model_output = model.generate(
            input_ids=tokenizer.encode(
                f'{record[text_col]}</s>', 
                return_tensors='pt'),
            max_length=2)

        try:
            # Decode the output returned by the model
            decoded_output = [tokenizer.decode(ids) for ids in model_output]

            # Add the detected emotion if any
            emotion_dataset['emotion'].append(decoded_output[0][6:])
        except:
            emotion_dataset['emotion'].append('UNKNOWN')
    
    return pd.DataFrame.from_dict(emotion_dataset)


def analyze_predicted_probs(dataset: pd.DataFrame, probs_col: str):
    '''
    Function that shows the amount of the number of samples within 
    each confidence interval from a provided list of predicted
    probabilities between 0,0 and 1,0. 

    Parameters
    ----------
    dataset : Pandas dataframe
        A dataset which contains the predicted probabilities
        for a set of samples.
    probs_col : str
        The column name in which there are the predicted probabilities.
    '''
    confidence_intervals = {
        'Very low': (0.0, 0.2),
        'Low': (0.2, 0.4),
        'Medium': (0.4, 0.6),
        'High': (0.6, 0.8),
        'Very high': (0.8, 1.0)
    }

    for conf_interv in confidence_intervals:
        min = confidence_intervals[conf_interv][0]
        max = confidence_intervals[conf_interv][1]
        count = dataset[(dataset[probs_col] >= min) & (dataset[probs_col] < max)].shape[0] if max != 1.0 \
            else dataset[(dataset[probs_col] >= min) & (dataset[probs_col] <= max)].shape[0]
        print(f'Confidence Interval {conf_interv} {confidence_intervals[conf_interv]}: {count} samples')


def map_texts_to_emotions(text_ids: list, is_test: bool = True):
    ''''
    Function that relates each provided text with its detected
    emotion previoulsy calculated and saved in a file, one per
    a train dataset and another for a test dataset. 

    Parameters
    ----------
    text_ids : list
        A list of integers that contains the identifiers of the
        train or test texts.
    is_test : bool (optional, default True)
        True if the provided list of text ids are related to a
        test dataset, False if they're related to a train dataset.
    '''
    emotions_df = pd.read_csv('../data/emotions_EXIST2021_test.csv') if is_test \
        else pd.read_csv('../data/emotions_EXIST2021_train.csv') 
    
    # Filter the dataset by the provided text identifiers
    emotions_df = emotions_df[emotions_df['id'].isin(text_ids)]

    # Count the number of texts per each emotion
    unique_emotions = list(set(list(emotions_df['emotion'].values)))
    for emotion in unique_emotions:
        count = emotions_df[emotions_df['emotion'] == emotion].shape[0] 
        print(f'Emotion: {emotion} - No. of texts: {count}')
    

