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

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

import sys
import string
import re

# don't increase epochs too high as dataset is smaller, models get overfitted
epochs = 30
delta = 0.01
batch_size = 128

model_flag = int(sys.argv[1])
contrastive_flag = int(sys.argv[2])
try:
    batch_size = int(sys.argv[3])
except:
    batch_size = 128

if model_flag == 0:
    model_name = "bert"
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        'bert-base-uncased',  # Start from base BERT model
        num_labels=2,  # Number of classification labels
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



elif model_flag == 1:
    model_name = "roberta"


    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained(
        'roberta-base',  # Start from base BERT model
        num_labels=2,  # Number of classification labels
        output_attentions=False,  # Whether the model returns attentions weights
        output_hidden_states=False,  # Whether the model returns all hidden-states
    )
    if contrastive_flag  == 1:
        # Load the model and optimizer
        checkpoint = torch.load('../roberta_model_fig_lan.pt')

        # Load the state dict of the saved model
        state_dict = checkpoint['model_state_dict']
        # print(state_dict.keys())

        # Filter out the 'roberta' keys in the state dict
        state_dict = {k.replace('roberta.', ''): v for k, v in state_dict.items()}
        state_dict.pop('pooler.dense.weight')
        state_dict.pop('pooler.dense.bias')

        # Load the state dict into the BERT model
        model.roberta.load_state_dict(state_dict)
        print("roberta contrastive model loaded")



elif model_flag == 2:
    model_name = "albert"

    tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
    model = AutoModelForSequenceClassification.from_pretrained(
        'albert-base-v2',  # Start from base BERT model
        num_labels=2,  # Number of classification labels
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

train_df = pd.read_csv("incivility_train.csv")
val_df = pd.read_csv("incivility_test.csv")


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

    text = filter_nontext(text)

    text = remove_stopwords(text)
    text = lemmatize_text(text)
        
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
        text = self.df.iloc[idx][2]
        text = str(text)
        text = text_cleaning(text)
        label = self.df.iloc[idx][3]
        if label == 'civil':
            label = 0
        elif label == 'uncivil':
            label = 1

        # Tokenize the text and pad the sequences
        tokens = self.tokenizer.tokenize(text)
        tokens = tokens[:self.max_len - 2]
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.nn.functional.pad(torch.tensor(input_ids), pad=(0, self.max_len - len(input_ids)), value=0)

        # Convert the label to a tensor
        label = torch.tensor(label).long()

        return input_ids, label



# train_df = train_df[train_df['comment_code'] != 'technical']
# val_df = val_df[val_df['comment_code'] != 'technical']

# train_df.to_csv('incivility_train_sampled.csv')
# val_df.to_csv('incivility_test_sampled.csv')

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
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5, weight_decay=1e-5)

# Define the loss function
from sklearn.metrics import confusion_matrix, f1_score

f1 = -0.1
loss_val = 100

# Total number of training steps is [number of batches] x [number of epochs]
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=5, num_training_steps=total_steps)

f1 = -0.1
loss_val = 100


validation_loss_counter = 0
val_loss = 100

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
    # val_f1_score = f1_score(true_labels, pred_flat, average='macro')
    val_f1_score = f1_score(true_labels, pred_flat, average='micro')

    print(f"Epoch {epoch + 1}: Average training loss: {avg_train_loss:.4f}, Average validation loss: {avg_test_loss:.4f}, Validation f1-score: {val_f1_score:.4f}")

    if avg_train_loss < delta:
        break

    if val_loss > avg_test_loss:
        val_loss = avg_test_loss 
        print(confusion_matrix(true_labels, pred_flat), val_f1_score)
        print(classification_report(true_labels, pred_flat))
        my_array_pred = np.array(pred_flat)
        my_array_true = np.array(true_labels)
        df = pd.DataFrame({'Pred': my_array_pred, 'True': my_array_true})
        if contrastive_flag == 1:
            df.to_csv('results/incivility/' + model_name+'_incivil_pred_contrastive_' + str(epoch) + '_' + str(val_f1_score) + '.csv', index=False)
        else:
            df.to_csv('results/incivility/' + model_name+'_incivil_pred_simple_' + str(epoch) + '_' + str(val_f1_score) + '.csv', index=False)

print(f1)
