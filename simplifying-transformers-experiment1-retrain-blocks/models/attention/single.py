import torch.nn.functional as F
import torch

import math

from models.base_model import BaseModule


class Attention(BaseModule):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None, _print: bool = False):
        if _print:
            print("After ", query[0], key[0], value[0], query[0].size())
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if _print:
            print("Scores", scores[0], scores[0].size())
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        #
        # if _print:
        #     print(scores[0])
        p_attn = F.softmax(scores, dim=-1)

        if _print:
            print("P_attn", p_attn[0], p_attn[0].size())
        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn, scores
