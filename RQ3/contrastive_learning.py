import string
import sys

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

datapath = sys.argv[1]
modelpath = sys.argv[2]
model_savepath = sys.argv[3]

tokenizer = AutoTokenizer.from_pretrained("modelpath")
model = AutoModelForSequenceClassification.from_pretrained("modelpath")


def get_data():
    dataframe = pd.read_csv(datapath)
    print(dataframe.keys())
    dataframe = dataframe.drop(["Id", "Fig_Exp", "SE", "General", "Selected option (a/b)"], axis=1)

    # # We will remove any empty values
    dataframe = dataframe.dropna()

    print(len(dataframe['Sentence'].values.tolist()))
    print(dataframe.head())
    print(dataframe.columns)
    return dataframe


dataframe = get_data()


def remove_block_quotes(text):
    modified_text = ''
    prev_line_block = False
    for line in text.split('\n'):
        if not line.strip().startswith('>'):
            modified_text += line + '\n'
            prev_line_block = False
        else:
            if prev_line_block is True:
                continue
            else:
                modified_text += '[BLOCK QUOTE].' + '\n'
                prev_line_block = True
    return modified_text


def remove_newlines(text):
    # replace new lines with space
    modified_text = text.replace("\n", " ")
    return modified_text


def remove_extra_whitespaces(text):
    return ' '.join(text.split())


def remove_triple_quotes(text):
    occurrences = [m.start() for m in re.finditer('```', text)]
    idx = len(occurrences)
    if idx % 2 == 1:
        text = text[:occurrences[idx - 1]]
        idx = idx - 1
    for i in range(0, idx, 2):
        if idx > 0:
            text = text[:occurrences[idx - 2]] + '[TRIPLE QUOTE].' + text[(occurrences[idx - 1] + 3):]
            idx = idx - 2
    return text


def remove_stacktrace(text):
    st_regex = re.compile('at [a-zA-Z0-9\.<>$]+\(.+\)')
    lines = list()
    for line in text.split('\n'):
        matches = st_regex.findall(line.strip())
        if len(matches) == 0:
            lines.append(line)
        else:
            for match in matches:
                line = line.replace(match, ' ')
            lines.append(line.strip(' \t'))

    lines = '\n'.join(lines)
    # hack to get rid of multiple spaces in the text
    # lines = ' '.join(lines.split())
    return lines


def remove_url(text):
    text = re.sub(r"http\S+", "[URL]", text)
    return text


def remove_usermention(text):
    text = re.sub(' @[^\s]+', ' [USER]', text)
    if text.startswith('@'):
        text = re.sub('@[^\s]+', '[USER]', text)
    return text


def filter_nontext(text):
    text = remove_url(text)
    text = remove_usermention(text)
    text = remove_block_quotes(text)
    text = remove_stacktrace(text)
    # text = remove_newlines(text)
    text = remove_triple_quotes(text)
    # text = remove_extra_whitespaces(text)
    return text.strip()


def text_cleaning(text):
    printable = set(string.printable)
    text = ''.join(filter(lambda x: x in printable, text))
    text = text.replace('\x00', ' ')  # remove nulls
    text = text.replace('\r', ' ')
    text = text.replace('\n', ' ')
    text = text.lower()  # Lowercasing
    text = text.strip()
    text = filter_nontext(text)
    return text


texts = []

for i in range(len(dataframe['Sentence'].values.tolist())):
    sentences = dataframe.iloc[i].values.tolist()
    texts.append([text_cleaning(sentences[0]), text_cleaning(sentences[1]), text_cleaning(sentences[2])])

dataframe = pd.DataFrame(texts, columns=['anchor', 'positive', 'negative'])

print(dataframe)

import csv
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertModel, BertTokenizer

# Read the CSV file and extract the positive and negative samples
positives = []
negatives = []

for row in texts:
    positives.append(row[0])
    negatives.append(row[2])
    positives.append(row[1])
    negatives.append(row[2])

print(len(positives))
print(len(negatives))

# Tokenize the positive and negative samples and convert them into input sequences
positive_ids = []
negative_ids = []
max_len = 0
for text in positives:
    positive_ids.append(tokenizer.encode(text))
    max_len = max(max_len, len(tokenizer.encode(text)))
for text in negatives:
    negative_ids.append(tokenizer.encode(text))
    max_len = max(max_len, len(tokenizer.encode(text)))

# Pad the input sequences to the same length
# max_len = max(len(positive_ids[0]), len(negative_ids[0]))
positive_ids = [torch.tensor(ids + [0] * (max_len - len(ids))) for ids in positive_ids]
negative_ids = [torch.tensor(ids + [0] * (max_len - len(ids))) for ids in negative_ids]
positive_ids = torch.nn.utils.rnn.pad_sequence(positive_ids, batch_first=True)
negative_ids = torch.nn.utils.rnn.pad_sequence(negative_ids, batch_first=True)

# Create a TensorDataset from the positive and negative input sequences
dataset = TensorDataset(positive_ids, negative_ids)

# Create a DataLoader from the dataset
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# Define a contrastive loss function
def contrastive_loss(positive_embeddings, negative_embeddings, margin):
    similarity = nn.CosineSimilarity(dim=1)
    pos_sim = similarity(positive_embeddings, negative_embeddings)
    loss = torch.mean(torch.max(torch.zeros_like(pos_sim), margin - pos_sim))
    return loss


# Define an Adam optimizer and a learning rate scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

# Iterate over the training data using a DataLoader and compute the contrastive loss
for epoch in range(5):
    running_loss = 0.0
    for positive_ids, negative_ids in dataloader:
        # Compute the contrastive loss
        positive_embeddings = model(positive_ids)[0]
        negative_embeddings = model(negative_ids)[0]
        loss = contrastive_loss(positive_embeddings, negative_embeddings, margin=0.5)

        # Backpropagate the loss and update the model parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Compute the running loss
        running_loss += loss.item()

    # Print the epoch loss
    print(f'Epoch {epoch + 1} loss: {running_loss}')

torch.save(model.state_dict(), model_savepath + '.pt')
