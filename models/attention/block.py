from torch import nn

from models.attention.single import Attention
from models.base_model import BaseModule


class BlockMultiHeadedAttention(BaseModule):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, d_k = 64, dropout=0.1, out_linear_overwrite: int = None):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_k
        self.h = h
        self.d_reducedmodel = int(h * self.d_k)
        if out_linear_overwrite:
            print(self.d_k, self.d_reducedmodel, self.h)
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, self.d_reducedmodel) for _ in range(3)])
        self.output_linear = nn.Linear(
            self.d_reducedmodel, d_model
        )
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)
