import argparse

from torch import nn

from models.base_model import BaseModel
from models.embedding.bert import BERTEmbedding
from models.modules.transformer_block import TransformerBlock


class BERT(BaseModel):
    def __init__(self, config, vocab_size: int):
        super(BERT, self).__init__()
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = config.hidden_features
        self.n_layers = config.layers
        self.attn_heads = config.heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = self.hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=self.hidden)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.hidden, self.attn_heads, self.hidden * 4, config.dropout) for _ in range(config.layers)])

    def forward(self, x, segment_info):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, segment_info)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x

    @staticmethod
    def extend_parser(parser) -> argparse.ArgumentParser:
        parser.add_argument('--hidden_features', type=int, default=768, help='# of hidden features')
        parser.add_argument('--layers', type=int, default=12, help='# of layers in model')
        parser.add_argument('--heads', type=int, default=12, help='# of attention heads')
        parser.add_argument('--dropout', type=float, default=0.1, help='dropout probability')

        return parser

