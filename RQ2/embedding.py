import torch
import numpy as np
from torch.nn import functional as F


def soft_decay(embeddings):
    u, s, v = torch.svd(embeddings)
    maxS = torch.max(s, dim=0).values.unsqueeze(-1)
    eps = 1e-7
    alpha = -0.6
    newS = - torch.log(1 - alpha * (s + alpha) + eps) / alpha
    maxNewS = torch.max(newS, dim=0).values.unsqueeze(-1)
    rescale_number = maxNewS / maxS
    newS = newS / rescale_number
    rescale_s_dia = torch.diag_embed(newS, dim1=-2, dim2=-1)
    new_input = torch.matmul(torch.matmul(u, rescale_s_dia), v.transpose(1, 0))
    return new_input
