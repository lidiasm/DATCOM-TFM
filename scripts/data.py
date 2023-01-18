import pandas as pd
import nltk
from transformers import AutoTokenizer, AutoModelWithLMHead

EXIST_TRAIN_DATASET_PATH = "../data/EXIST2021_train.tsv"
EXIST_TEST_DATASET_PATH = "../data/EXIST2021_test.tsv"

def read_train_dataset():
    """
    Function that reads the EXIST train file to load the 
    train dataset as dataframe.

    Returns
    -------
    A Pandas dataframe
    """
    return pd.read_table(EXIST_TRAIN_DATASET_PATH)

def read_test_dataset():
    """
    Function that reads the EXIST test file to load the 
    test dataset as dataframe.

    Returns
    -------
    A Pandas dataframe
    """
    return pd.read_table(EXIST_TEST_DATASET_PATH)

def count_words(dataset: pd.DataFrame, text_column: str, 
                tokenize: bool = False, unique_words: bool = False):
    # Delete punctuation marks and split the docs into words
    if (tokenize):
        dataset[text_column] = [
            record.split(" ") for record in 
            list(dataset[text_column].str.replace("[^\w\s]", ""))
        ]
    
    # Variable to count (different) words
    word_count = 0
        
    # Count different words in each doc
    if (unique_words):
        for doc in dataset[text_column]:
            word_count += len(set(doc))
        
    # Count the number of words in each doc
    else:
        for doc in dataset[text_column]:
            word_count += len(doc)

    return word_count

def get_top_ngrams(dataset: pd.DataFrame, column: str, n_words: int):
    """
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
    A Pandas dataframe
    """
    # Split texts in sentences of N words
    dataset["ngrams"] = dataset[column].str.split() \
        .apply(lambda x: list(map(" ".join, nltk.ngrams(x, n=n_words))))

    # Compute the frequency per sentece
    return (dataset.assign(count=dataset["ngrams"]\
        .str.len()).explode("ngrams")) \
        .sort_values("count", ascending=False)

def get_emotions(dataset: pd.DataFrame, text_column: str, class_column: str):
    """
    Function that downloads the tokenizer and a pre-trained model
    to detect the emotions from a set of documents within a dataset.
    The goal is to compute global metrics and specific calculations
    per each class to summarize the number of texts that belong
    to each available emotion.

    Parameters
    ----------
    dataset : Pandas dataframe
        The data which contains a set of texts to analyze.
    text_column : str
        The column name in which the set of texts is stored.
    class_column : str
        The column name in which the class labels are stored.

    Returns
    -------
    A dictionary with the global metrics as well as the calculations
    per each class.
    """
    # Download the tokenizer and the model for emotion detection from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-emotion")
    model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-emotion")

    # Variables to save the global results and the emotions per class
    global_emotions = {
        "sadness": 0,
        "joy": 0,
        "love": 0,
        "anger": 0,
        "fear": 0,
        "surprise": 0,
    }
    class_emotions = {
        "ideological-inequality": {
            "sadness": 0,
            "joy": 0,
            "love": 0,
            "anger": 0,
            "fear": 0,
            "surprise": 0,
        },
        "misogyny-non-sexual-violence": {
            "sadness": 0,
            "joy": 0,
            "love": 0,
            "anger": 0,
            "fear": 0,
            "surprise": 0,
        },
        "non-sexist": {
            "sadness": 0,
            "joy": 0,
            "love": 0,
            "anger": 0,
            "fear": 0,
            "surprise": 0,
        },
        "objectification": {
            "sadness": 0,
            "joy": 0,
            "love": 0,
            "anger": 0,
            "fear": 0,
            "surprise": 0,
        },
        "sexual-violence": {
            "sadness": 0,
            "joy": 0,
            "love": 0,
            "anger": 0,
            "fear": 0,
            "surprise": 0,
        },
        "stereotyping-dominance": {
            "sadness": 0,
            "joy": 0,
            "love": 0,
            "anger": 0,
            "fear": 0,
            "surprise": 0,
        }
    }

    # Convert the dataset into a list of records
    dataset_dict = dataset.to_dict("records")

    # Iterate over the documents within the dataset
    for record in dataset_dict:
        # Encode the input to pass it to the model to detect the emotion
        model_output = model.generate(
            input_ids=tokenizer.encode(
                f"{record[text_column]}</s>", 
                return_tensors="pt"),
            max_length=2)

        try:
            # Decode the output returned by the model
            decoded_output = [tokenizer.decode(ids) for ids in model_output]
            detected_emotion = decoded_output[0][6:]

            # Compute the results globally 
            global_emotions[detected_emotion] += 1

            # Compute the results per class
            class_emotions[record[class_column]][detected_emotion] += 1
        except:
            pass
    
    return {
        "global_emotions": global_emotions,
        "class_emotions": class_emotions
    }