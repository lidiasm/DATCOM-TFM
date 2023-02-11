import pandas as pd
from deep_translator import GoogleTranslator


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
