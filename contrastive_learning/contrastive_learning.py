import string
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import re
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn as nn
import pandas as pd
from transformers import AutoTokenizer
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from sklearn.metrics import classification_report
from nltk.corpus import wordnet
from nltk import pos_tag
nltk.download('maxent_treebank_pos_tagger')
nltk.download('averaged_perceptron_tagger')

import re
from transformers import get_linear_schedule_with_warmup


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
import sys
import string
import re

flag = int(sys.argv[1])
print(flag)

if flag == 0:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    saved_file = 'bert_model_and_optimizer.pt'
elif flag == 1:
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoModel.from_pretrained("roberta-base")
    saved_file = 'roberta_model_fig_lan.pt'
elif flag == 2:
    tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
    model = AutoModel.from_pretrained("albert-base-v2")
    saved_file = 'albert_model_and_optimizer.pt'

model.to(device)
print(saved_file)

# =======================================================================================================
# READING DATASET

def get_data():
    dataframe = pd.read_csv("Fig_Lan_Annotation.csv")
    print(dataframe.keys())

    dataframe = dataframe.dropna()

    print(len(dataframe['Sentence'].values.tolist()))
    print(dataframe.head())
    print(dataframe.columns)
    return dataframe


dataframe = get_data()




def text_cleaning(text):


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

    def remove_stopwords(text):
        # Tokenize the text into individual words
        tokens = word_tokenize(text)
        
        # Get the list of English stopwords
        stop_words = set(stopwords.words('english'))
        
        # Filter out the stopwords
        filtered_tokens = [word for word in tokens if word.casefold() not in stop_words]
        
        # Join the filtered tokens back into a single string
        filtered_text = ' '.join(filtered_tokens)
        
        return filtered_text



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

    def lemmatize_text(text):
        # Tokenize the text into individual words
        tokens = word_tokenize(text)
        
        # Map POS tags to WordNet tags
        def get_wordnet_pos(tag):
            if tag.startswith('J'):
                return wordnet.ADJ
            elif tag.startswith('V'):
                return wordnet.VERB
            elif tag.startswith('N'):
                return wordnet.NOUN
            elif tag.startswith('R'):
                return wordnet.ADV
            else:
                return wordnet.NOUN  # Assume nouns by default

        # Initialize WordNet lemmatizer
        lemmatizer = WordNetLemmatizer()

        pos_tags = pos_tag(tokens)

        # Lemmatize tokens with POS tags
        lemmatized_tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(pos_tag)) for token, pos_tag in pos_tags]
        
        # Join the lemmatized tokens back into a single string
        lemmatized_text = ' '.join(lemmatized_tokens)
        
        return lemmatized_text


    def filter_nontext(text):
        text = remove_url(text)
        text = remove_usermention(text)
        text = remove_block_quotes(text)
        text = remove_stacktrace(text)
        # text = remove_newlines(text)
        text = remove_triple_quotes(text)
        # text = remove_extra_whitespaces(text)

        return text.strip()

    printable = set(string.printable)
    text = ''.join(filter(lambda x: x in printable, text))
    text = text.replace('\x00', ' ')  # remove nulls
    text = text.replace('\r', ' ')
    text = text.replace('\n', ' ')
    text = re.sub('[^0-9a-zA-Z]+', ' ', text)
    text = re.sub("(<.*?>)", " ", text)  # remove html markup
    text = re.sub("(\\W|\\d)", " ", text)  # remove non-ascii and digits

    text = text.lower()  # Lowercasing
    text = text.strip()

    # text = filter_nontext(text)
    # text = remove_stopwords(text)
    # text = lemmatize_text(text)
    # text = text.strip()
    
    return text




texts = []

for i in range(len(dataframe['Sentence'].values.tolist())):
    sentences = dataframe.iloc[i].values.tolist()
    texts.append([text_cleaning(sentences[1]), text_cleaning(sentences[3]), text_cleaning(sentences[4])])

dataframe = pd.DataFrame(texts, columns=['anchor', 'positive', 'negative'])

print(dataframe)

# Read the CSV file and extract the positive and negative samples
ancors = []
positives = []
negatives = []

for row in texts:
    ancors.append (row[0])
    positives.append (row[1])
    negatives.append (row[2])

    ancors.append (row[1])
    positives.append (row[0])
    negatives.append (row[2])


print(len(positives))
print(len(negatives))

# =============================================================================================================

import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, anchors, positives, negatives, tokenizer, max_len):
        self.anchors = anchors
        self.positives = positives
        self.negatives = negatives
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, idx):
        anchor = self.anchors[idx]
        positive = self.positives[idx]
        negative = self.negatives[idx]

        anchor = self.tokenizer.encode_plus(
            anchor,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        positive = self.tokenizer.encode_plus(
            positive,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        negative = self.tokenizer.encode_plus(
            negative,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'anchor': anchor['input_ids'].flatten(),
            'positive': positive['input_ids'].flatten(),
            'negative': negative['input_ids'].flatten(),
        }

# Create a dataset and a dataloader
dataset = TextDataset(ancors, positives, negatives, tokenizer, max_len=128)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Define a contrastive loss function

# def contrastive_loss(anchor, positive, negative, margin=0.3):
#     pos_sim = F.cosine_similarity(anchor, positive)
#     neg_sim = F.cosine_similarity(anchor, negative)
#     return torch.mean(F.relu(pos_sim - neg_sim + margin))


def info_nce_loss(anchor_embeddings, positive_embeddings, negative_embeddings):
    # Compute the similarity between the anchor and the positive sample
    positive_sim = torch.nn.functional.cosine_similarity(anchor_embeddings, positive_embeddings)

    # Compute the similarity between the anchor and the negative samples
    negative_sim = torch.nn.functional.cosine_similarity(anchor_embeddings, negative_embeddings)

    # Compute the InfoNCE loss
    loss = -torch.log(torch.exp(positive_sim) / (torch.exp(positive_sim) + torch.exp(negative_sim)))

    return loss.mean()



# Define an Adam optimizer and a learning rate scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
previous_loss = 100

# Train the model
model.train()
for epoch in range(30):
    running_loss = 0.0
    for batch in dataloader:
        anchor = model(batch['anchor'].to(device))[1]
        positive = model(batch['positive'].to(device))[1]
        negative = model(batch['negative'].to(device))[1]

        # loss = contrastive_loss(anchor, positive, negative)

        # Compute the InfoNCE loss
        loss = info_nce_loss(anchor, positive, negative)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch {epoch + 1} loss: {running_loss / len(dataloader)}')

    if running_loss < previous_loss:
        previous_loss = running_loss
        # Save the model and optimizer
        torch.save({
            'model_state_dict': model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
        }, saved_file)

    if running_loss < 0.1:
        break
