# Sentence Similarity Analysis Script

This Python script analyzes the similarity between sentences using pretrained transformer models and computes similarity metrics. It takes as input a CSV file containing sentences and performs the following steps:

1. Preprocess the sentences to remove unwanted characters and clean the text.
2. Load and preprocess the data from the CSV file based on the data type ('SE', 'General', or 'Other').
3. Compute sentence embeddings using a pretrained transformer model (BERT, RoBERTa, etc.) and tokenizer.
4. Calculate cosine similarity between pairs of sentences and apply soft decay to the embeddings if required.
5. Compute effect size using Cliff's delta and perform a one-tailed Wilcoxon signed-rank test.

## Requirements

Before running the script, you need to ensure you have the following libraries installed:

- string
- pandas
- numpy
- torch
- transformers
- scikit-learn
- scipy
- nltk

You can install the required libraries using the following command:

```
pip install pandas numpy torch transformers scikit-learn scipy nltk
```


Additionally, you need to download the necessary NLTK resources.



## Usage

```
python script.py <datapath> <modelpath> <data_type>

```


- `<datapath>`: Path to the CSV file containing the data.
- `<modelpath>`: Path to the directory or name of the pretrained transformer model (e.g., "bert-base-uncased", "roberta-base", "albert-base-v2", "microsoft/codebert-base").
- `<data_type>`: Type of data to process ('SE', 'General', or 'Other').



## Example

```
python script.py data.csv bert-base-uncased SE

```
