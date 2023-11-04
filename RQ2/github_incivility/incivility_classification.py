import torch.nn as nn
import pandas as pd
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification
import nltk
import numpy as np
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import re, sys, string, argparse, os
from transformers import get_linear_schedule_with_warmup
from nltk import pos_tag

# just run first time
# nltk.download('all')


# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(train_file, test_file):
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(test_file)
    return train_df, val_df

def load_model(model_name, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False,
    )
    return model, tokenizer

def get_model_path(model_name):
    model_paths = {
        "bert": "bert-base-uncased",
        "roberta": "roberta-base",
        "albert": "albert-base-v2",
        "codebert": "microsoft/codebert-base"
    }
    if model_name not in model_paths:
        raise ValueError("Invalid model_name. Choose from bert, roberta, albert, codebert.")
    return model_paths[model_name]

def load_contrastive_model(model, model_name, contrastive_flag, path_to_contrastive_weights):
    if contrastive_flag == 1:
        # Load the model and optimizer
        checkpoint = torch.load(path_to_contrastive_weights)
        # Load the state dict of the saved model
        state_dict = checkpoint['model_state_dict']

        if model_name in ["bert"]:
            # Filter out the 'bert' keys in the state dict
            state_dict = {k.replace('bert.', ''): v for k, v in state_dict.items()}
            # Load the state dict into the model
            model.bert.load_state_dict(state_dict)
            print(model_name, ": contrastive model loaded")
        elif model_name in ["albert"]:
            # Filter out the 'albert' keys in the state dict
            state_dict = {k.replace('albert.', ''): v for k, v in state_dict.items()}
            # Load the state dict into the model
            model.albert.load_state_dict(state_dict)
            print(model_name, ": contrastive model loaded")
        elif model_name in ["roberta", "codebert"]:
            # Filter out the 'roberta' keys in the state dict
            state_dict = {k.replace('roberta.', ''): v for k, v in state_dict.items()}
            state_dict.pop('pooler.dense.weight')
            state_dict.pop('pooler.dense.bias')
            # Load the state dict into the model
            model.roberta.load_state_dict(state_dict)
            print(model_name, ": contrastive model loaded")
    return model


def text_cleaning(text):
    printable = set(string.printable)
    text = ''.join(filter(lambda x: x in printable, text))
    text = text.replace('\x00', ' ')  # remove nulls
    text = text.replace('\r', ' ')
    text = text.replace('\n', ' ')
    text = text.lower()  # Lowercasing
    text = text.strip()
    return text

# Define a mapping dictionary for labels
label_mapping = {
    "civil": 1,
    "uncivil": 0
}


def prepare_data(df, tokenizer, max_len):
    class CSVDataset(Dataset):
        def __init__(self, df, tokenizer, max_len):
            self.df = df
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            text = self.df.iloc[idx]["text"]
            text = text_cleaning(text)
            # Map the label to an integer using the label_mapping dictionary
            label = label_mapping.get(self.df.iloc[idx]["label"], 0)

            # Tokenize the text and pad the sequences
            tokens = self.tokenizer.tokenize(text)
            tokens = tokens[:self.max_len - 2]
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_ids = torch.nn.functional.pad(torch.tensor(input_ids), pad=(0, self.max_len - len(input_ids)), value=0)

            # Convert the label to a tensor
            label = torch.tensor(label).long()

            return input_ids, label

    dataset = CSVDataset(df, tokenizer, max_len)
    return dataset

def train_model(model, train_dataloader, epochs, delta, optimizer, scheduler, val_dataloader, output_file):
    model.to(device)
    f1 = -0.1
    loss_val = 100
    for epoch in range(epochs):
        total_train_loss = 0
        model.train()

        for step, batch in enumerate(train_dataloader):
            b_input_ids, b_labels = batch
            b_input_ids = b_input_ids.to(device)
            b_labels = b_labels.to(device)

            model.zero_grad()        

            # Forward pass
            loss, logits = model(b_input_ids, labels=b_labels)[:2]

            # Accumulate the training loss
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient
            optimizer.step()

            # Update the learning rate
            scheduler.step()

        # Calculate the average loss over all of the batches
        avg_train_loss = total_train_loss / len(train_dataloader)            

        if avg_train_loss < loss_val:
            loss_val = avg_train_loss

        # Validation
        model.eval()

        total_eval_loss = 0
        predictions, true_labels = [], []
        validation_loss = 0

        for batch in val_dataloader:
            b_input_ids, b_labels = batch
            b_input_ids = b_input_ids.to(device)
            b_labels = b_labels.to(device)

            with torch.no_grad():        
                loss, logits = model(b_input_ids, labels=b_labels)[:2]

            total_eval_loss += loss.item()
            validation_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Store predictions and true labels
            predictions.append(logits)
            true_labels.append(label_ids)

        # Calculate the average loss over all of the batches
        avg_test_loss = validation_loss / len(val_dataloader)

        # Flatten the predictions and true values for aggregate evaluation on all classes.
        predictions = np.concatenate(predictions, axis=0)
        true_labels = np.concatenate(true_labels, axis=0)

        # For each sample, pick the label (0 or 1) with the higher score.
        pred_flat = np.argmax(predictions, axis=1).flatten()

        # Calculate the validation accuracy of the model
        val_f1_score = f1_score(true_labels, pred_flat, average='micro')

        print(f"Epoch {epoch + 1}: Average training loss: {avg_train_loss:.4f}, Average validation loss: {avg_test_loss:.4f}, Validation f1-score: {val_f1_score:.4f}")

        if avg_train_loss < delta:
            break

        if epoch == 0 or (f1 <= 0.0 and val_f1_score > 0.0) or (avg_test_loss < val_loss):
            val_loss = avg_test_loss 
            f1 = val_f1_score
            print(confusion_matrix(true_labels, pred_flat))
            print(classification_report(true_labels, pred_flat))
            my_array_pred = np.array(pred_flat)
            my_array_true = np.array(true_labels)
            df = pd.DataFrame({'Pred': my_array_pred, 'True': my_array_true})
            df.to_csv(output_file, index=False)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Incivility Classification.")
    parser.add_argument("--epoch", type=int, default=30, help="Number of epochs.")
    parser.add_argument("--delta", type=float, default=0.01, help="Delta value.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch Size.")
    parser.add_argument("--model_name", type=str, default='bert', required=True, choices=["bert", "roberta", "albert", "codebert"], help="Flag. Choose from bert, roberta, albert, codebert.")
    parser.add_argument("--contrastive_flag", type=int, default=0, help="Figurative language fine-tuned or not indicator.")
    parser.add_argument("--path_to_contrastive_weights", type=str, default=None, help="Path to contrastive learned model weights.")
    parser.add_argument("--output", type=str, default="output.csv", help="Output file name.")  # New argument for output file
    parser.add_argument("--train_file", type=str, default="incivility-train.csv", required=True, help="Path to the training CSV file.")
    parser.add_argument("--test_file", type=str, default="incivility-test.csv", required=True, help="Path to the test CSV file.")
    return parser.parse_args()



def main():
    args = parse_arguments()
    epochs = args.epoch
    delta = args.delta
    batch_size = args.batch_size
    model_name = args.model_name
    contrastive_flag = args.contrastive_flag
    path_to_contrastive_weights = args.path_to_contrastive_weights
    output_file = args.output
    train_file = args.train_file
    test_file = args.test_file

    # Check if contrastive_flag is 1 and path_to_contrastive_weights is not provided
    if contrastive_flag == 1 and path_to_contrastive_weights is None:
        raise ValueError("If contrastive_flag is 1, path_to_contrastive_weights must be provided.")

    if contrastive_flag == 1:
        output_file = model_name + '_contrastive_' + output_file
    else:
        output_file = model_name + '_' + output_file

    print("Epoch:", epochs)
    print("Delta:", delta)
    print("Model:", model_name)
    print("Contrastive Flag:", contrastive_flag)
    print("Path to Contrastive Weights:", path_to_contrastive_weights)
    print("Output File:", output_file)

    # Set the maximum sequence length
    MAX_LEN = 128

    # Load the training and validation data
    train_df, val_df = load_data(train_file, test_file)

    # Get model path from HuggingFace
    model_path = get_model_path(model_name)

    # Load the model and tokenizer
    model, tokenizer = load_model(model_name, model_path)

    # Load contrastive model if applicable
    if contrastive_flag == 1:
        model = load_contrastive_model(model, model_name, contrastive_flag, path_to_contrastive_weights)

    # Prepare datasets
    train_dataset = prepare_data(train_df, tokenizer, MAX_LEN)
    val_dataset = prepare_data(val_df, tokenizer, MAX_LEN)

    print(len(train_dataset))
    print(len(val_dataset))

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define the optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    # Total number of training steps is [number of batches] x [number of epochs]
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    train_model(model, train_dataloader, epochs, delta, optimizer, scheduler, val_dataloader, output_file)

if __name__ == "__main__":
    main()
