import pandas as pd
from textaugment import EDA
from deep_translator import GoogleTranslator

import nltk  
nltk.download('omw-1.4')

def translate_english_spanish_texts(dataset: pd.DataFrame, text_col: str, lang_col: str):
    '''
    Translates english texts to spanish and spanish texts to
    english to increase the population of documents stored
    in a provided dataset. It could be used as a data augmentation
    technique to create more train samples.

    Parameters
    ----------
    dataset : Pandas dataframe
        The data which contains a set of texts to translate.
    text_col : str
        The column name in which the set of texts is stored.
    lang_col : str
        The column name in which the languages of the texts are stored.

    Returns
    -------
    A list of strings with the translated documents.
    '''
    # Convert the dataset to a list of dictionaries
    dataset_to_dict = dataset.to_dict('records')

    # Initialize two translators
    en_to_es_translator = GoogleTranslator(source='en', target='es')
    es_to_en_translator = GoogleTranslator(source='es', target='en')

    # Variable to save the new texts
    new_texts = []

    # Iterate over the texts to translate them depending on their language
    for record in dataset_to_dict:
        if (record[lang_col] == 'en'):
            new_texts.append(en_to_es_translator.translate(record[text_col]))

        elif (record[lang_col] == 'es'):
            new_texts.append(es_to_en_translator.translate(record[text_col]))

    return new_texts


def apply_easy_data_augmentation(dataset: pd.DataFrame, text_col: str, 
                                n_replacements: int, n_times: int):
    '''
    Searchs for N synonyms to replace the N original words 
    for N times in order to augment the number of texts 
    stored within a provided dataset. It is a data augmentation
    technique for NLP problems known as EDA (Easy Data Augmentation).

    Parameters
    ----------
    dataset : Pandas dataframe
        The data which contains a set of texts to augment.
    text_col : str
        The column name in which the set of texts is stored.
    n_replacements : int
        The number of words to replace with their synonyms
    n_times : int
        The number of times to apply this technique. 
        E.g.: 1 time produces the double of samples.
        E.g.: 2 times produces the triple of samples.
        ....

    Returns
    -------
    A list of strings with the original texts along with
    the new synthetic documents.
    '''
    # Initialize an EDA object
    text_aug_obj = EDA()

    # Get texts only
    train_texts = list(dataset[text_col].values)

    # Add the original texts to the final variable
    # in which the new samples will be stored as well
    aug_texts = list(train_texts)

    # Iterate over the texts to create new samples
    for time in range(0, n_times):
        for text in train_texts:
            aug_texts.append(text_aug_obj.synonym_replacement(
                sentence=text, 
                n=n_replacements))
    
    return aug_texts
