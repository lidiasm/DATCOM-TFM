import sys
sys.path.append("../scripts")

# Import data read and compute functions
from data import read_train_dataset, read_test_dataset

# Import text preprocess functions
from processing import *

# numpy: to work with numeric codifications and embeddings
import numpy as np

# keras: to define and build LSTM models
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.layers import LSTM, Activation, Dense, Input, Embedding
from keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

# time: to measure execution time
import time

# Read EXIST datasets
train_df = read_train_dataset()
test_df = read_test_dataset()

# Variables to save the different accuracy and AUC values
# to then calculate the average of all iterations
train_accuracy_values = []
test_accuracy_values = []
train_auc_values = []
test_auc_values = []

# Open a file to save the metrics of each iteration
opened_file = open("../outputs/1st_experiment.txt", "w")

def get_train_test_matrix(
    max_n_words: int, sequence_len: int, 
    lemm: bool = False, stemm: bool = False):
    """
    Process the train and test documents to then convert them
    into numeric sequence matrixes so the datasets can be
    used to train a LSTM model.

    Parameters
    ----------
    max_n_words : int
        Maximum number of words to keep within the LSTM memory
        based on computing the word frequency.
    sequence_len : int
        Maximum lenght of all sequences.
    lemm : bool (optional)
        True to apply lemmatization to the train and test documents.
    stemm : bool (optional)
        True to apply stemming to the train and test documents.

    Returns
    -------
    A dictionary with the following keys:
        - 'tokenizer': a Keras Tokenizer object based on the train documents
        that contains the vocabulary to then be used to create the embeddings.
        - 'train_matrix', 'test_matrix': the numeric sequence matrixes
        after converting the train and test documents.
        - 'train_labels', 'test_labels': two numeric lists which contains
        the encoded class labels for train and test datasets.
    """
    # Process train and test text documents
    processed_df = process_encode_datasets(
        train_df=train_df, 
        test_df=test_df,
        lemm=lemm, 
        stemm=stemm
    )

    # Processed train texts and encoded train labels 
    train_texts = list(processed_df["train_df"]["cleaned_text"].values)
    train_labels = processed_df["encoded_train_labels"]

    # Processed test texts and encoded test labels
    test_texts = list(processed_df["test_df"]["cleaned_text"].values)
    test_labels = processed_df["encoded_test_labels"]

    # Createa a tokenizer based on train texts
    tokenizer = Tokenizer(num_words=max_n_words)
    tokenizer.fit_on_texts(train_texts)

    # Transform each text into a numeric sequence
    train_sequences = tokenizer.texts_to_sequences(train_texts)

    # Transform each numeric sequence into a 2D vector
    train_matrix = pad_sequences(
        sequences=train_sequences, 
        maxlen=sequence_len)

    # Tokenize the test documents using the prior trained tokenizer
    test_sequences = tokenizer.texts_to_sequences(test_texts)

    # Transform each numeric sequence into a 2D vector
    test_matrix = pad_sequences(
        sequences=test_sequences,
        maxlen=sequence_len)

    return {
        "tokenizer": tokenizer,
        "train_matrix": train_matrix,
        "train_labels": train_labels,
        "test_matrix": test_matrix,
        "test_labels": test_labels
    }

def get_embedding_matrix(embedding_file: str, tokenizer: Tokenizer, sequence_len: int):
    """
    Load the embeddings stored in the provided file to then
    create a matrix with the numeric encoding of each
    available word within the tokenizer vocabulary.

    Parameters
    ----------
    embedding_file : str
        The path to the file which contains a set of embeddings
    tokenizer : Tokenizer (Keras)
        A trained Keras tokenizer which contains the vocabulary
        of the documents to use during the training of models
    sequence_len : int
        Maximum lenght of all embeddings.

    Returns
    -------
    A Numpy ndarray which represents an embedding matrix.
    """
    # Load the embeddings stored in a TXT file
    embedding_file = open(embedding_file)

    # Store each word with its embeddings
    embeddings_index = {
        line.split()[0]:np.asarray(line.split()[1:], dtype="float32") 
        for line in embedding_file
    }

    # Initialize the embedding matrix with zeros
    embedding_matrix = np.zeros(shape=(len(tokenizer.word_index)+1, sequence_len))

    # Complete the matrix with the prior loaded embeddings
    for word, i in tokenizer.word_index.items():
        # Search for the embeddings of each word
        embedding_vector = embeddings_index.get(word)

        # Words not found will be zeros
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix    

def validate_lstm_model(
    model: Model, 
    train_matrix: np.ndarray, train_labels: list, 
    test_matrix: np.ndarray, test_labels: list):
    """
    Evaluates the provided trained LSTM model over the 
    train and test datasets to get the accuracy, AUC and
    a confusion matrix. To create the predictions for a
    binary classification a threshold has been set:
        - <= 0.5 represents the negative class (non-sexist).
        - > 0.5 represents the positive class (sexist).

    Parameters
    ----------
    model : Keras model
        A trained Keras model to be evaluated.
    train_matrix : Numpy ndarray
        A numeric sequence matrix with the trained documents.
    train_labels : list
        A numeric list with the class labels of the train dataset.
    test_matrix : Numpy ndarray
        A numeric sequence matrix with the test documents.
    test_labels : list
        A numeric list with the class labels of the test dataset.
    metrics_filename : str (optional)
        A path and filename to store the metrics over the 
        train and test datasets in a TXT file.
    conf_matrix_filename : str (optional)
        A path and filename to store the confusion matrix in
        a PNG image.

    Returns
    -------
    None.
    """
    # Compute and print the accuracy and AUC over train
    train_acc = model.evaluate(
        x=train_matrix, 
        y=np.array(train_labels))

    # Compute and print the accuracy and AUC over test
    test_acc = model.evaluate(
        x=test_matrix, 
        y=np.array(test_labels))

    # Save train metrics 
    train_accuracy_values.append(train_acc[1])
    train_auc_values.append(train_acc[2])

    # Save test metrics
    test_accuracy_values.append(test_acc[1])
    test_auc_values.append(test_acc[2])
    
    print(f"Accuracy over train dataset: {train_acc[1]}", file=opened_file) 
    print(f"AUC over train dataset: {train_acc[2]}", file=opened_file) 
    print(f"Accuracy over test dataset: {test_acc[1]}", file=opened_file) 
    print(f"AUC over test dataset: {test_acc[2]}", file=opened_file) 

#############################################################################################
###################################### RUN EXPERIMENTS ######################################
N_ITERATIONS = 30
APPLY_LEMMATIZATION = False
APPLY_STEMMING = False
MAX_N_WORDS = 1000
SEQUENCE_MAX_LEN = 100
EMBEDDING_FILE_PATH = "../embeddings/glove.6B.100d.txt"
BATCH_SIZE = 128
N_EPOCHS = 100
VALID_RATE = 0.2
MODEL_CALLBACKS = [EarlyStopping(
    monitor="val_loss",
    min_delta=0.001,
    patience=15,
    restore_best_weights=True)]
LOSS_FUNCTION = "binary_crossentropy"
OPTIMIZER = "adam"
VALID_METRICS = ["accuracy", "AUC"]

# Write the settings in the prior opened file
print(f"\nNo. of iterations for the experiment: {N_ITERATIONS}", file=opened_file) 
print(f"Apply lemmatization?: {APPLY_LEMMATIZATION}", file=opened_file)
print(f"Apply stemming?: {APPLY_STEMMING}", file=opened_file)
print(f"Max no. of words: {MAX_N_WORDS}", file=opened_file)
print(f"Max sequence size: {SEQUENCE_MAX_LEN}", file=opened_file)
print(f"Pre-trained embedding file: {EMBEDDING_FILE_PATH}", file=opened_file)

print(f"Batch size: {BATCH_SIZE}", file=opened_file)
print(f"No. of epochs: {N_EPOCHS}", file=opened_file)
print(f"Validation rate: {VALID_RATE}", file=opened_file)
print(f"Training callbacks: EarlyStopping(val_loss, 0.001, 15, restore_best_weights)", file=opened_file)
print(f"Loss function: {LOSS_FUNCTION}", file=opened_file)
print(f"Optimizer: {OPTIMIZER}", file=opened_file)
print(f"Validation metrics: {VALID_METRICS}", file=opened_file)

## START MEASURING TIME
start = time.time()

for i in range(0, N_ITERATIONS):
    print(f"\nIteration: {i}", file=opened_file) 

    # Create a tokenizer based on the train texts
    # Process train and test texts
    lstm_data = get_train_test_matrix(
        max_n_words=MAX_N_WORDS,
        sequence_len=SEQUENCE_MAX_LEN,
        lemm=APPLY_LEMMATIZATION,
        stemm=APPLY_STEMMING
    )

    # Load the embeddings stored in the defined file path
    # Encode the train matrix with these embeddings
    lstm_embedding_matrix = get_embedding_matrix(
        embedding_file=EMBEDDING_FILE_PATH,
        tokenizer=lstm_data["tokenizer"],
        sequence_len=SEQUENCE_MAX_LEN
    )

    # LSTM ARCHITECTURE
    ## Input layer
    input_layer = Input(
        name="inputs",
        shape=[SEQUENCE_MAX_LEN])

    ## Embedding layer: pre-trained embeddings
    layer = Embedding(
        input_dim=len(lstm_data["tokenizer"].word_index)+1,
        output_dim=SEQUENCE_MAX_LEN,
        weights=[lstm_embedding_matrix],
        input_length=MAX_N_WORDS,
        trainable=False)(input_layer)

    ## LSTM layer
    layer = LSTM(units=64)(layer)

    ## Output layer
    layer = Dense(
        name="output",
        units=1)(layer)

    ## Activation layer
    output_layer = Activation(activation="sigmoid")(layer)

    # CREATE A LSTM MODEL WITH THE PRIOR ARCHITECTURE
    ## Model object
    lstm_model1 = Model(
        inputs=input_layer,
        outputs=output_layer)

    ## Compile the model 
    lstm_model1.compile(
        loss=LOSS_FUNCTION,
        optimizer=OPTIMIZER,
        metrics=VALID_METRICS)

    # LSTM TRAINING
    ## Train the prior built model
    lstm_model1.fit(
        x=lstm_data["train_matrix"], 
        y=np.array(lstm_data["train_labels"]),
        batch_size=BATCH_SIZE,
        epochs=N_EPOCHS,
        validation_split=VALID_RATE,
        callbacks=MODEL_CALLBACKS)

    ## LSTM VALIDATION
    ## Evaluate the trained LSTM model over train and test datasets
    validate_lstm_model(
        model=lstm_model1,
        train_matrix=lstm_data["train_matrix"],
        train_labels=lstm_data["train_labels"],
        test_matrix=lstm_data["test_matrix"],
        test_labels=lstm_data["test_labels"]
    )

## FINISH MEASURING TIME
end = time.time()

# Calculate the average of each metric
print("\nTOTAL RESULTS", file=opened_file) 
print(f"Train accuracy avg: {sum(train_accuracy_values)/len(train_accuracy_values)}", file=opened_file) 
print(f"Train AUC avg: {sum(train_auc_values)/len(train_auc_values)}", file=opened_file) 
print(f"Test accuracy avg: {sum(test_accuracy_values)/len(test_accuracy_values)}", file=opened_file) 
print(f"Test AUC avg: {sum(test_auc_values)/len(test_auc_values)}", file=opened_file) 

# Add the execution time
print(f"\nEXECUTION TIME: {end-start}", file=opened_file) 