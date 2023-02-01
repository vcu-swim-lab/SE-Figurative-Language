import string

import pandas as pd
from compute_similarity import compute_similarity
import numpy as np
from transformers import AutoTokenizer, AutoModel

import sys
import string
import re

datapath = sys.argv[1]
modelpath = sys.argv[2]
type = sys.argv[3]

tokenizer = AutoTokenizer.from_pretrained(modelpath)
model = AutoModel.from_pretrained(modelpath)


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


def get_data():
    dataframe = pd.read_csv(datapath)
    print(dataframe.keys())
    # important keys are Sentence, Replacement, and Negative
    # We will drop other keys

    dataframe = dataframe.drop(['Id', 'Fig_Exp'], axis=1)

    print(len(dataframe['Sentence'].values.tolist()))

    na_free = dataframe.dropna()
    print(len(na_free['Sentence'].values.tolist()))
    dataframe = dataframe[np.invert(dataframe.index.isin(na_free.index))]

    print(len(dataframe['Sentence'].values.tolist()))

    if type == "SE":
        dataframe = dataframe.drop(['SE'], axis=1)
        dataframe = dataframe.dropna()
        dataframe = dataframe.drop(['General'], axis=1)
    elif type == "General":
        dataframe = dataframe.drop(['General'], axis=1)
        dataframe = dataframe.dropna()
        dataframe = dataframe.drop(['SE'], axis=1)
    else:
        dataframe = dataframe.drop(['SE'], axis=1)
        dataframe = dataframe.drop(['General'], axis=1)

    return dataframe


dataframe = get_data()
count_a = 0
count_b = 0
for i in range(len(dataframe['Sentence'].values.tolist())):
    sentences = dataframe.iloc[i].values.tolist()

    texts = []
    for sentence in sentences:
        texts.append(text_cleaning(str(sentence)))

    cos_similarities = compute_similarity(texts, "soft_decay", model, tokenizer)
    format_float0 = "{:.3f}".format(cos_similarities[0])
    format_float1 = "{:.3f}".format(cos_similarities[1])
    print(f'{format_float0},{format_float1}')

    if cos_similarities[0] > cos_similarities[1]:
        count_a = count_a + 1
    else:
        count_b = count_b + 1

print(count_a / (count_a + count_b), count_b / (count_a + count_b))
