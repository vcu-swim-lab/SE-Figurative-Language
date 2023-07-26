import string
import pandas as pd
import torch
import re
import sys
import argparse
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

def select_pretrained_model(model_name, output_save_path):
    models = {
        "bert": ("bert-base-uncased", f"{output_save_path}/bert_model_and_optimizer.pt"),
        "roberta": ("roberta-base", f"{output_save_path}/roberta_model_and_optimizer.pt"),
        "albert": ("albert-base-v2", f"{output_save_path}/albert_model_and_optimizer.pt"),
        "codebert": ("microsoft/codebert-base", f"{output_save_path}/codebert_model_and_optimizer.pt"),
    }
    if model_name not in models:
        raise ValueError("Invalid model value. Use bert, roberta, albert, codebert.")
    return models[model_name]

def get_data(data_file):
    dataframe = pd.read_csv(data_file)
    print(dataframe.keys())

    dataframe = dataframe.dropna()

    print(len(dataframe['Sentence'].values.tolist()))
    print(dataframe.head())
    print(dataframe.columns)
    return dataframe


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
        text = remove_triple_quotes(text)
        return text.strip()

    printable = set(string.printable)
    text = ''.join(filter(lambda x: x in printable, text))
    text = text.replace('\x00', ' ')  # remove nulls
    text = text.replace('\r', ' ')
    text = text.replace('\n', ' ')
    text = text.lower()  # Lowercasing
    text = text.strip()
    text = filter_nontext(text)
    text = text.strip()

    return text



def prepare_dataset(dataframe, tokenizer, max_len=128):
    texts = []
    for i in range(len(dataframe['Sentence'].values.tolist())):
        sentences = dataframe.iloc[i].values.tolist()
        texts.append([text_cleaning(sentences[1]), text_cleaning(sentences[3]), text_cleaning(sentences[4])])

    dataframe = pd.DataFrame(texts, columns=['anchor', 'positive', 'negative'])

    ancors, positives, negatives = [], [], []
    for row in texts:
        ancors.append(row[0])
        positives.append(row[1])
        negatives.append(row[2])
        ancors.append(row[1])
        positives.append(row[0])
        negatives.append(row[2])

    dataset = TextDataset(ancors, positives, negatives, tokenizer, max_len=max_len)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    return dataloader

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

def info_nce_loss(anchor_embeddings, positive_embeddings, negative_embeddings):
    # Compute the similarity between the anchor and the positive sample
    positive_sim = torch.nn.functional.cosine_similarity(anchor_embeddings, positive_embeddings)

    # Compute the similarity between the anchor and the negative samples
    negative_sim = torch.nn.functional.cosine_similarity(anchor_embeddings, negative_embeddings)

    # Compute the InfoNCE loss
    loss = -torch.log(torch.exp(positive_sim) / (torch.exp(positive_sim) + torch.exp(negative_sim)))

    return loss.mean()

def train_model(model, dataloader, num_epochs=30, saved_file='model_and_optimizer.pt'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    previous_loss = 100

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in dataloader:
            anchor = model(batch['anchor'].to(device))[1]
            positive = model(batch['positive'].to(device))[1]
            negative = model(batch['negative'].to(device))[1]

            loss = info_nce_loss(anchor, positive, negative)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch + 1} loss: {running_loss / len(dataloader)}')

        if running_loss < previous_loss:
            previous_loss = running_loss
            # Save the model and optimizer
            torch.save({'model_state_dict': model.state_dict()}, saved_file)

        if running_loss < 0.1:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="Pretrained model name (bert, roberta, albert, codebert)")
    parser.add_argument("data_file", help="Path to the data file")
    parser.add_argument("output_save_path", help="Path to save the trained model and optimizer")
    args = parser.parse_args()

    selected_model, saved_file = select_pretrained_model(args.model_name, args.output_save_path)
    tokenizer = AutoTokenizer.from_pretrained(selected_model)
    model = AutoModel.from_pretrained(selected_model)

    dataframe = get_data(args.data_file)
    dataloader = prepare_dataset(dataframe, tokenizer, max_len=128)

    train_model(model, dataloader, num_epochs=30, saved_file=saved_file)
