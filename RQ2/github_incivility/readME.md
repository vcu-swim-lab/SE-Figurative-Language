# Incivility Classification using Transformers

This repository contains code for training a transformer-based model for incivility classification. The model is implemented using the PyTorch library and utilizes the Hugging Face Transformers library for pre-trained transformer models.

## Prerequisites

Before running the code, ensure you have the following installed:

1. Python 3.6 or later
2. PyTorch
3. Hugging Face Transformers library
4. NLTK library
5. Pandas library
6. Scikit-learn library

You can install the required dependencies using `pip`:

`pip install torch transformers nltk pandas scikit-learn`


## Usage

2. Make sure you have training and test data in CSV format. The CSV files should contain two columns: "text" (containing the text to classify) and "label" (containing the class labels). The label value should be "civil" and "uncivil". This is a binary classification.

3. Run the script `incivility_classification.py` with the following command:


`python incivility_classification.py --epoch EPOCH --delta DELTA --batch_size BATCH_SIZE --col LABEL --model_name MODEL_NAME --contrastive_flag CONTRASTIVE_FLAG --path_to_contrastive_weights PATH_TO_CONTRASTIVE_WEIGHTS --output OUTPUT_FILE --train_file TRAIN_CSV --test_file TEST_CSV`

### Arguments:


Replace the arguments in angle brackets with appropriate values:
- `<EPOCH>`: Number of training epochs (default is 30).
- `<DELTA>`: Delta value for early stopping based on training loss (default is 0.01).
- `<BATCH_SIZE>`: Batch size for training (default is 128).
- `<LABEL>`: Column name for the class labels in the CSV file (choose from "civil" or "uncivil").
- `<MODEL_NAME>`: Name of the pre-trained transformer model to use (choose from "bert", "roberta", "albert", "codebert").
- `<CONTRASTIVE_FLAG>`: Flag indicating if a contrastive model is used (1 for True, 0 for False).
- `<PATH_TO_CONTRASTIVE_WEIGHTS>`: Path to the contrastive model weights (required if contrastive_flag is set to 1).
- `<OUTPUT_FILE>`: Name of the output CSV file to save predictions and true labels.
- `<TRAIN_CSV>`: Path to the training CSV file.
- `<TEST_CSV>`: Path to the test CSV file.

## About the Code

The provided code demonstrates the following main functionalities:

1. Loading and processing the training and test data from CSV files.
2. Loading the pre-trained transformer model and tokenizer from Hugging Face.
3. Optionally, loading a contrastive model if the contrastive_flag is set to 1.
4. Preparing the data using tokenization and data cleaning techniques.
5. Training the classification model with early stopping based on training loss.
6. Evaluating the trained model on the validation data and saving the results to an output CSV file.

## Output

The script will print the training progress, average training loss, average validation loss, and validation F1-score for each epoch. Additionally, it will save the predictions in the specified output file in CSV format, containing columns "Pred" (predicted labels) and "True" (true labels).


## Note

- Ensure that your system has access to a CUDA-enabled GPU if you want to use GPU acceleration for training. The script automatically checks for GPU availability and uses it if available.

- The code also includes a data cleaning function `text_cleaning()` that removes block quotes, URLs, user mentions, HTML markup, non-ASCII characters, and digits. It also performs lemmatization and removes stopwords. Modify this function according to your specific needs.
