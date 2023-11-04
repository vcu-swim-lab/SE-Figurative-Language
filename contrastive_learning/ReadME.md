# Contrastive Learning Training for Figurative Language

This repository contains Python code to train a model using the InfoNCE loss function for learning text embeddings. The code uses the Hugging Face Transformers library for handling pretrained language models and PyTorch for training.

## Overview

The purpose of this code is to train a model using InfoNCE (InfoMax) loss, which is a contrastive learning objective that helps learn better text representations. It is particularly useful when dealing with limited labeled data.

## Prerequisites

Before running the code, make sure you have the following installed:

1. Python (3.6 or higher)
2. PyTorch (1.8.0 or higher)
3. Transformers library from Hugging Face
4. pandas library
5. nltk library

You can install the required libraries using pip:

```bash
pip install torch transformers pandas nltk
```

## Usage

To run the training script, use the following command:

```bash
python contrastive_learning.py <model_name> <data_file> <output_save_path>
```


- `<model_name>`: Pretrained model name. Choose from "bert", "roberta", "albert", or "codebert".
- `<data_file>`: Path to the CSV file containing the training data.
- `<output_save_path>`: Folder path to save the trained model and optimizer.


Example
-------

```
python train_model.py bert Fig_Lan_Annotation.csv output
```


This will train a BERT-based model using the InfoNCE loss with the provided CSV data file and save the trained model and optimizer in the "output" directory.

Notes
-----

1. The code supports different pretrained models from the Hugging Face Transformers library (BERT, RoBERTa, ALBERT, CodeBERT).
2. The InfoNCE loss is used for training, which helps improve text embeddings in a self-supervised manner.
3. The training process will stop if the loss falls below a threshold (0.1) or after 10 epochs, whichever comes first.

Please ensure that you have enough GPU memory to run the training process if using a GPU. The code will automatically use the GPU if available; otherwise, it will use the CPU for training.

Note: Make sure to customize the CSV file with your data before running the training script.

Happy training!

