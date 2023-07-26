import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import sys
import string
import re
import argparse


parser = argparse.ArgumentParser(description="Script description.")
parser.add_argument("--epoch", type=int, default=100, help="Number of epochs.")
parser.add_argument("--delta", type=float, default=0.01, help="Delta value.")
parser.add_argument("--batch_size", type=int, default=128, help="Batch Size.")
parser.add_argument("--col", type=str, default="Anger", required=True, choices=["Anger", "Fear", "Love", "Joy", "Surprise", "Sadness"], help="Choose from Anger, Fear, Love, Joy, Sadness and Surprise.")
parser.add_argument("--model_name", type=str, choices=["bert", "roberta", "albert", "codebert"], required=True, help="Flag. Choose from bert, roberta, albert, codebert.")
parser.add_argument("--contrastive_flag", type=int, default=0, help="Figurative language finutuned or not indicator.")
parser.add_argument("--path_to_contrastive_weights", type=str, default="No path is provided", help="Path to contrastive learned model weights.")
parser.add_argument("--output", type=str, default="output.csv", help="Output file name.")  # New argument for output file

args = parser.parse_args()

epoch = args.epoch
delta = args.delta
batch_size = args.batch_size
col = args.col
model_name = args.flag
contrastive_flag = args.contrastive_flag
path_to_contrastive_weights = args.path_to_contrastive_weights
output_file = args.output

if contrastive_flag == 1:
    output_file = model_name + '_' + col + '_contrastive_' + output_file
else:
    output_file = model_name + '_' + output_file


print("Epoch:", epoch)
print("Delta:", delta)
print("Column:", col)
print("Flag:", flag)
print("Contrastive Flag:", contrastive_flag)
print("Path to Contrastive Weights:", path_to_contrastive_weights)
print("Output File:", output_file)


if model_name == 'bert':
    model_path = 'bert-base-uncased'
elif model_name == 'roberta':
    model_path = 'roberta-base'
elif model_name == 'albert':
    model_path = 'albert-base-v2'
elif model_name == 'codebert':
    model_path = 'microsoft/codebert-base'


tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,  # Start from base model
    num_labels=2,  # Number of classification labels
    output_attentions=False,  # Whether the model returns attentions weights
    output_hidden_states=False,  # Whether the model returns all hidden-states
)


if model_name in ["bert", "albert"]:
    if contrastive_flag == 1:
        # Load the model and optimizer
        checkpoint = torch.load(path_to_contrastive_weights)

        # Load the state dict of the saved model
        state_dict = checkpoint['model_state_dict']

        # Filter out the 'bert' keys in the state dict
        state_dict = {k.replace('bert.', ''): v for k, v in state_dict.items()}
        # print(state_dict.keys())

        # Load the state dict into the BERT model
        model.bert.load_state_dict(state_dict)
        print(model_name, ": contrastive model loaded")

elif model_name in ["roberta", "codebert"]:
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained(
        'roberta-base',  # Start from base BERT model
        num_labels=2,  # Number of classification labels
        output_attentions=False,  # Whether the model returns attentions weights
        output_hidden_states=False,  # Whether the model returns all hidden-states
    )

    if contrastive_flag == 1:
        # Load the model and optimizer
        checkpoint = torch.load(path_to_contrastive_weights)

        # Load the state dict of the saved model
        state_dict = checkpoint['model_state_dict']
        # print(state_dict.keys())

        # Filter out the 'roberta' keys in the state dict
        state_dict = {k.replace('roberta.', ''): v for k, v in state_dict.items()}

        # Load the state dict into the BERT model
        model.roberta.load_state_dict(state_dict, strict=False)
        print(model_name, ": contrastive model loaded")

model.to(device)

# Load the training and validation data from CSV files
train_df = pd.read_csv("github-train.csv")
val_df = pd.read_csv("github-test.csv")

print(train_df.keys())


def text_cleaning(text):
    printable = set(string.printable)
    text = ''.join(filter(lambda x: x in printable, text))
    text = text.replace('\x00', ' ')  # remove nulls
    text = text.replace('\r', ' ')
    text = text.replace('\n', ' ')
    text = text.lower()  # Lowercasing
    text = text.strip()
    return text



# Define the dataset class
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
        label = self.df.iloc[idx][col]

        # Tokenize the text and pad the sequences
        tokens = self.tokenizer.tokenize(text)
        tokens = tokens[:self.max_len - 2]
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.nn.functional.pad(torch.tensor(input_ids), pad=(0, self.max_len - len(input_ids)), value=0)

        # Convert the label to a tensor
        label = torch.tensor(label).long()

        return input_ids, label


# Set the maximum sequence length
MAX_LEN = 128

# Create the datasets
train_dataset = CSVDataset(train_df, tokenizer, MAX_LEN)
val_dataset = CSVDataset(val_df, tokenizer, MAX_LEN)

print(len(train_dataset))
print(len(val_dataset))

# Create the dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define the optimizer and learning rate scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

f1 = -0.1
loss_val = 100

# Train the model
for epoch in range(epochs):
    total_train_loss = 0
    model.train()  # Put the model into training mode

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

    # ========================================
    #               Validation
    # ========================================
    model.eval()  # Put the model in evaluation mode

    total_eval_accuracy = 0
    total_eval_loss = 0
    predictions , true_labels = [], []
    validation_loss = 0

    for batch in val_dataloader:
        b_input_ids, b_labels = batch
        b_input_ids = b_input_ids.to(device)
        b_labels = b_labels.to(device)

        with torch.no_grad():        
            loss, logits = model(b_input_ids, labels=b_labels)[:2]
                
        total_eval_loss += loss.item()

        # Accumulate the training loss
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
    val_f1_score = f1_score(true_labels, pred_flat, average='binary')

    print(f"Epoch {epoch + 1}: Average training loss: {avg_train_loss:.4f}, Average validation loss: {avg_test_loss:.4f}, Validation f1-score: {val_f1_score:.4f}")

    if avg_train_loss < delta:
        break
    
    if epoch == 0 or (val_loss < val_f1_score):
        val_loss = val_f1_score 
        print(confusion_matrix(true_labels, pred_flat), val_loss)
        print(classification_report(true_labels, pred_flat))
        my_array_pred = np.array(pred_flat)
        my_array_true = np.array(true_labels)
        df = pd.DataFrame({'Pred': my_array_pred, 'True': my_array_true})
        df.to_csv(output_file, index=False)

