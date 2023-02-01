import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import sys
import string
import re

epoch = 30
delta = 0.0001

col = sys.argv[1]
modelpath = sys.argv[2]
constrastive_flag = int(sys.argv[3])
constrastive_path = sys.argv[4]
data_flag = sys.argv[4]

tokenizer = AutoTokenizer.from_pretrained(modelpath)
model = AutoModelForSequenceClassification.from_pretrained(modelpath)

if constrastive_flag == 1:
    state_dict = torch.load(constrastive_path)
    model.load_state_dict(state_dict, strict=False)
    print("contrastive model loaded")

model.to(device)

# Load the training and validation data from CSV files
train_df = pd.read_csv("dataset/sentiment-train.csv")
val_df = pd.read_csv("dataset/sentiment-test.csv")

print(train_df.keys())

if data_flag == "emotion":
    train_df = pd.read_csv("dataset/github-train.csv")
    val_df = pd.read_csv("dataset/github-test.csv")
elif data_flag == "civility":
    train_df = pd.read_csv("dataset/civility-train.csv")
    val_df = pd.read_csv("dataset/civility-test.csv")
else:
    train_df = pd.read_csv("dataset/sentiment-train-2.csv")
    val_df = pd.read_csv("dataset/sentiment-test-2.csv")


def text_cleaning(text):
    printable = set(string.printable)
    text = ''.join(filter(lambda x: x in printable, text))
    text = text.replace('\x00', ' ')  # remove nulls
    text = text.replace('\r', ' ')
    text = text.replace('\n', ' ')
    #    text = re.sub('[^0-9a-zA-Z]+', ' ', text)
    #    text = re.sub("(<.*?>)", " ", text)  # remove html markup
    #    text = re.sub("(\\W|\\d)", " ", text)  # remove non-ascii and digits

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

# Define the loss function
from sklearn.metrics import confusion_matrix, f1_score

f1 = 0
loss_val = 100

# Train the model
for epoch in range(epoch):
    print("epoch", epoch, "is running")
    train_loss = 0
    train_acc = 0
    model.train()
    for input_ids, labels in train_dataloader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, labels=labels)
        loss = outputs[0]
        train_loss += loss.item()
        _, preds = torch.max(outputs[1], dim=1)
        train_acc += (preds == labels).sum().item()
        #        print(preds)
        loss.backward()
        optimizer.step()
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)
    print(f"Epoch {epoch + 1}: Train loss: {train_loss:.4f}")

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
            loss = outputs[0]
            val_loss += loss.item()
            _, preds = torch.max(outputs[1], dim=1)
            val_acc += (preds == labels).sum().item()
            y_true.extend(labels.cpu().detach().numpy())
            y_pred.extend(preds.cpu().detach().numpy())
    val_loss /= len(val_dataloader)
    val_acc /= len(val_dataloader)
    print(f"Epoch {epoch + 1}: Validation loss: {val_loss:.4f}")
    # Print the confusion matrix
    print(confusion_matrix(y_true, y_pred), f1_score(y_true, y_pred), f1)
    if f1 < f1_score(y_true, y_pred):
        f1 = f1_score(y_true, y_pred)

    if train_loss < delta:
        break

print(f1)
