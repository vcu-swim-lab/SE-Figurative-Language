import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from embedding import soft_decay

import numpy as np


def compute_similarity(sentences, embedding_type, model, tokenizer):
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    # Sentences we want sentence embeddings for
    sentence_1 = sentences[0]
    sentence_2 = sentences[1]
    sentence_3 = sentences[2]

    # Tokenize and encode the sentences
    encoded_input_1 = tokenizer(sentence_1, padding=True, truncation=True, max_length=128, return_tensors='pt')
    encoded_input_2 = tokenizer(sentence_2, padding=True, truncation=True, max_length=128, return_tensors='pt')
    encoded_input_3 = tokenizer(sentence_3, padding=True, truncation=True, max_length=128, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output_1 = model(**encoded_input_1)
        model_output_2 = model(**encoded_input_2)
        model_output_3 = model(**encoded_input_3)

    # Perform pooling. In this case, mean pooling
    sentence_embedding_1 = mean_pooling(model_output_1, encoded_input_1['attention_mask'])
    sentence_embedding_2 = mean_pooling(model_output_2, encoded_input_2['attention_mask'])
    sentence_embedding_3 = mean_pooling(model_output_3, encoded_input_3['attention_mask'])

    if embedding_type == "soft_decay":
        sentence_embedding_1 = soft_decay(sentence_embedding_1)
        sentence_embedding_2 = soft_decay(sentence_embedding_2)
        sentence_embedding_3 = soft_decay(sentence_embedding_3)

    # Calculate cosine similarity
    similarity = cosine_similarity(sentence_embedding_1, sentence_embedding_2)
    similarity_2 = cosine_similarity(sentence_embedding_1, sentence_embedding_3)

    # similarity = np.linalg.norm(np.array(sentence_embedding_1) - np.array(sentence_embedding_2))
    # similarity_2 = np.linalg.norm(np.array(sentence_embedding_1) - np.array(sentence_embedding_3))
    # print(similarity, similarity_2)
    return [similarity[0][0], similarity_2[0][0]]
    # return [similarity, similarity_2]
