import string
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import wilcoxon
import sys, re, nltk, argparse
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize

#nltk.download('all')


def soft_decay(embeddings):
    """
    Apply soft decay to input embeddings.

    Parameters:
        embeddings (torch.Tensor): Input embeddings.

    Returns:
        torch.Tensor: Embeddings after soft decay.
    """
    u, s, v = torch.svd(embeddings)
    max_s = torch.max(s, dim=0).values.unsqueeze(-1)
    eps = 1e-7
    alpha = -0.6
    new_s = -torch.log(1 - alpha * (s + alpha) + eps) / alpha
    max_new_s = torch.max(new_s, dim=0).values.unsqueeze(-1)
    rescale_number = max_new_s / max_s
    new_s = new_s / rescale_number
    rescale_s_dia = torch.diag_embed(new_s, dim1=-2, dim2=-1)
    new_input = torch.matmul(torch.matmul(u, rescale_s_dia), v.transpose(1, 0))
    return new_input


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


def load_data(datapath, data_type):
    dataframe = pd.read_csv(datapath)
    dataframe = dataframe.drop(['idx', 'Fig_Exp'], axis=1)

    if data_type == "SE":
        na_free = dataframe.dropna()
        dataframe = dataframe[np.invert(dataframe.index.isin(na_free.index))]
        dataframe = dataframe.drop(['General'], axis=1)
        dataframe = dataframe.dropna()
        dataframe = dataframe.drop(['SE'], axis=1)
    elif data_type == "General":
        na_free = dataframe.dropna()
        dataframe = dataframe[np.invert(dataframe.index.isin(na_free.index))]
        dataframe = dataframe.drop(['SE'], axis=1)
        dataframe = dataframe.dropna()
        dataframe = dataframe.drop(['General'], axis=1)
    else:
        dataframe = dataframe.drop(['SE'], axis=1)
        dataframe = dataframe.drop(['General'], axis=1)

    return dataframe


def compute_similarity(sentences, embedding_type, model, tokenizer):
    """
    Compute cosine similarity between sentence embeddings.

    Parameters:
        sentences (list): List of sentences to compare.
        embedding_type (str): Type of embedding to use ('soft_decay' or 'default').
        model: Pretrained transformer model.
        tokenizer: Pretrained tokenizer.

    Returns:
        list: List of cosine similarity scores.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    sentence_embeddings = []

    for sentence in sentences:
        encoded_input = tokenizer(sentence, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
        with torch.no_grad():
            model_output = model(**encoded_input)
        sentence_embedding = mean_pooling(model_output, encoded_input['attention_mask']).detach().cpu().clone()

        if embedding_type == "soft_decay":
            sentence_embedding = soft_decay(sentence_embedding)

        sentence_embeddings.append(sentence_embedding)

    similarity = cosine_similarity(sentence_embeddings[0], sentence_embeddings[1])[0][0]
    similarity_2 = cosine_similarity(sentence_embeddings[0], sentence_embeddings[2])[0][0]
    
    return [similarity, similarity_2]


def cliffs_delta(group_a, group_b):
    """
    Calculate Cliff's delta effect size between two groups.

    Parameters:
        group_a (list): List containing the values of Group A.
        group_b (list): List containing the values of Group B.

    Returns:
        float: Cliff's delta effect size.
    """
    def count_concordant_discordant_pairs(a, b):
        concordant = discordant = 0
        for i in range(len(a)):
            for j in range(len(b)):
                if a[i] < b[j]:
                    concordant += 1
                elif a[i] > b[j]:
                    discordant += 1
        return concordant, discordant

    concordant_pairs, discordant_pairs = count_concordant_discordant_pairs(group_a, group_b)
    total_pairs = len(group_a) * len(group_b)

    cliffs_delta = (concordant_pairs - discordant_pairs) / total_pairs
    return cliffs_delta


def compute_effect_size(group_a, group_b):
    """
    Compute the effect size using Cliff's delta.

    Parameters:
        group_a (list): List containing the values of Group A.
        group_b (list): List containing the values of Group B.

    Returns:
        float: Effect size.
    """
    return cliffs_delta(group_a, group_b)


def perform_one_tailed_wilcoxon_test(group_a, group_b, alternative='greater'):
    """
    Perform a one-tailed Wilcoxon signed-rank test.

    Parameters:
        group_a (list): List containing the values of Group A.
        group_b (list): List containing the values of Group B.
        alternative (str): Alternative hypothesis for the test ('greater' or 'less').

    Returns:
        tuple: Test statistic and p-value.
    """
    return wilcoxon(group_a, group_b, alternative=alternative)


# Main function
def main(datapath, modelpath, data_type):
    """
    Main function to process data and compute similarity metrics.

    Parameters:
        datapath (str): Path to the CSV file containing the data.
        modelpath (str): Path to the pretrained model.
        data_type (str): Type of data to process ('SE', 'General', or 'Other').
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(modelpath)
    model = AutoModel.from_pretrained(modelpath)
    model.to(device)

    dataframe = load_data(datapath, data_type)

    group_a = []
    group_b = []

    count_a = 0
    count_b = 0

    for i in range(len(dataframe['Sentence'].values.tolist())):
        sentences = dataframe.iloc[i].values.tolist()
        texts = [text_cleaning(str(sentence)) for sentence in sentences]
        cos_similarities = compute_similarity(texts, "soft_decay", model, tokenizer)
        group_a.append(cos_similarities[0])
        group_b.append(cos_similarities[1])

        if cos_similarities[0] > cos_similarities[1]:
            count_a = count_a + 1
        else:
            count_b = count_b + 1
    tc = count_b+count_a
    print(tc)
    print("Similarity percentage:", count_a / (count_a + count_b))

    effect_size = compute_effect_size(group_b, group_a)
    print("Cliff's delta:", effect_size)

    # Perform the one-tailed Wilcoxon signed-rank test with Group B expected to have higher values
    statistic, p_value = perform_one_tailed_wilcoxon_test(group_a, group_b, alternative='greater')
    print("Test statistic:", statistic)
    print("p-value:", p_value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process data and compute similarity metrics.')
    parser.add_argument('datapath', type=str, help='Path to the CSV file containing the data')
    parser.add_argument('modelpath', type=str, help='Path to the pretrained model')
    parser.add_argument('data_type', type=str, choices=['SE', 'General', 'Other'], help='Type of data to process (SE, General, or Other)')
    args = parser.parse_args()
    main(args.datapath, args.modelpath, args.data_type)
