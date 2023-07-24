import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import sys
import string
import re

epoch = 300
delta = 0.01

col = sys.argv[1]
flag = int(sys.argv[2])
contrastive_flag = int(sys.argv[3])


print(col)

if flag == 0:
    model_name = "bert"
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        'bert-base-uncased',  # Start from base BERT model
        num_labels=7,  # Number of classification labels
        output_attentions=False,  # Whether the model returns attentions weights
        output_hidden_states=False,  # Whether the model returns all hidden-states
    )
    if contrastive_flag == 1:
        # Load the model and optimizer
        checkpoint = torch.load('../bert_model_and_optimizer.pt')

        # Load the state dict of the saved model
        state_dict = checkpoint['model_state_dict']

        # Filter out the 'bert' keys in the state dict
        state_dict = {k.replace('bert.', ''): v for k, v in state_dict.items()}
        # print(state_dict.keys())

        # Load the state dict into the BERT model
        model.bert.load_state_dict(state_dict)
        print("bert contrastive model loaded")



elif flag == 1:
    model_name = "roberta"


    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained(
        'roberta-base',  # Start from base BERT model
        num_labels=7,  # Number of classification labels
        output_attentions=False,  # Whether the model returns attentions weights
        output_hidden_states=False,  # Whether the model returns all hidden-states
    )

    if contrastive_flag == 1:
        # Load the model and optimizer
        checkpoint = torch.load('../roberta_model_and_optimizer.pt')

        # Load the state dict of the saved model
        state_dict = checkpoint['model_state_dict']
        # print(state_dict.keys())

        # Filter out the 'roberta' keys in the state dict
        state_dict = {k.replace('roberta.', ''): v for k, v in state_dict.items()}

        # Load the state dict into the BERT model
        model.roberta.load_state_dict(state_dict, strict=False)
        print("roberta contrastive model loaded")



elif flag == 2:
    model_name = "albert"

    tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
    model = AutoModelForSequenceClassification.from_pretrained(
        'albert-base-v2',  # Start from base BERT model
        num_labels=7,  # Number of classification labels
        output_attentions=False,  # Whether the model returns attentions weights
        output_hidden_states=False,  # Whether the model returns all hidden-states
    )

    if contrastive_flag == 1:
        # Load the model and optimizer
        checkpoint = torch.load('../albert_model_and_optimizer.pt')

        # Load the state dict of the saved model
        state_dict = checkpoint['model_state_dict']

        # Filter out the 'bert' keys in the state dict
        state_dict = {k.replace('bert.', ''): v for k, v in state_dict.items()}

        # Load the state dict into the BERT model
        model.albert.load_state_dict(state_dict)
        print("albert contrastive model loaded")




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
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# Define the optimizer and learning rate scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

f1 = -0.1
loss_val = 100

# Train the model
for epoch in range(epoch):
#    print("epoch", epoch, "is running")
    train_loss = 0
    train_acc = 0
    model.train()
    for input_ids, labels in train_dataloader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        outputs = model(input_ids, labels=labels)
        _, preds = torch.max(outputs[1], dim=1)
        train_acc += (preds == labels).sum().item()

        loss = outputs[0]

        # backpropagation
        loss.backward()

        # update the weights
        optimizer.step()

        # clear the gradients
        optimizer.zero_grad()

        train_loss += loss.item()
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)

    if train_loss < loss_val:
        loss_val = train_loss

    # Test the model
    # Initialize the confusion matrix
    y_true = []
    y_pred = []
    val_loss = 0
    val_acc = 0
    model.eval()
    with torch.no_grad():
        for input_ids, labels in val_dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            outputs = model(input_ids, labels=labels)
            _, preds = torch.max(outputs[1], dim=1)
            val_acc += (preds == labels).sum().item()
            y_true.extend(labels.cpu().detach().numpy())
            y_pred.extend(preds.cpu().detach().numpy())

    if f1 < f1_score(y_true, y_pred):
        print(f"Epoch {epoch + 1}: Train loss: {train_loss:.4f}")
        f1 = f1_score(y_true, y_pred)
        print(confusion_matrix(y_true, y_pred), f1)
        my_array_pred = np.array(y_pred)
        my_array_true = np.array(y_true)
        df = pd.DataFrame({'Pred': my_array_pred, 'True': my_array_true})
#        df.to_csv(model_name+'_pred_'+col+'_contrastive.csv', index=False)
        if contrastive_flag == 1:
            df.to_csv(model_name+'_pred_'+col+ '_contrastive' + str(f1) +'.csv', index=False)
        else:
            df.to_csv(model_name+'_pred_'+col+'_simple.csv', index=False)

    if train_loss < delta:
        break

print(f1)
