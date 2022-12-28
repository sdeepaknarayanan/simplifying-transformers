import argparse

import numpy as np
from torch import nn
from torch.optim import Adam

from models.base_model import BaseModel
from models.embedding.bert import BERTEmbedding
from models.modules.transformer_block import TransformerBlock


class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))


class BERT(BaseModel):
    def __init__(self, config, vocab_size: int):
        super(BERT, self).__init__(config)
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """
        self.hidden = config.hidden_features
        self.n_layers = config.layers
        self.attn_heads = config.heads
        self.device = config.device

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = self.hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=self.hidden).to(self.conf.device)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.hidden, self.attn_heads, self.hidden * 4, config.dropout).to(self.conf.device)
            for _ in range(config.layers)
        ])

        self.optimizer = Adam(
            self.parameters(),
            lr=self.conf.lr,
            betas=(self.conf.adam_beta1, self.conf.adam_beta2),
            weight_decay=self.conf.adam_weight_decay
        )
        self.optim_schedule = ScheduledOptim(
            self.optimizer,
            self.hidden,
            n_warmup_steps=self.conf.warmup_steps
        )

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


class BERTLM(BaseModel):
    def __init__(self, config, vocab_size: int):
        super(BERTLM, self).__init__(config)
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """
        self.hidden = config.hidden_features
        self.n_layers = config.layers
        self.attn_heads = config.heads
        self.device = config.device

        self.mask_lm = MaskedLanguageModel(self.hidden, vocab_size).to(self.conf.device)

        self.bert = BERT(config, vocab_size)
        # paper noted they used 4*hidden_size for ff_network_hidden_size
        # self.feed_forward_hidden = self.hidden * 4
        #
        # # embedding for BERT, sum of positional, segment, token embeddings
        # self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=self.hidden).to(self.conf.device)
        #
        # # multi-layers transformer blocks, deep network
        # self.transformer_blocks = nn.ModuleList([
        #     TransformerBlock(self.hidden, self.attn_heads, self.hidden * 4, config.dropout).to(self.conf.device)
        #     for _ in range(config.layers)
        # ])

        self.optimizer = Adam(
            self.parameters(),
            lr=self.conf.lr,
            betas=(self.conf.adam_beta1, self.conf.adam_beta2),
            weight_decay=self.conf.adam_weight_decay
        )
        self.optim_schedule = ScheduledOptim(
            self.optimizer,
            self.hidden,
            n_warmup_steps=self.conf.warmup_steps
        )

    def forward(self, x, segment_info):
        x = self.bert(x, segment_info)
        return self.mask_lm(x)

    def load_state(self):
        print("Lading State for BERTML")
        self.bert.load_state()
        # TODO: change to distinguish between bert and bertml checkpoints
        # try:
        #     self.load_state()
        # except:
        #     self.bert.load_state()
    @staticmethod
    def extend_parser(parser) -> argparse.ArgumentParser:
        parser.add_argument('--hidden_features', type=int, default=768, help='# of hidden features')
        parser.add_argument('--layers', type=int, default=12, help='# of layers in model')
        parser.add_argument('--heads', type=int, default=12, help='# of attention heads')
        parser.add_argument('--dropout', type=float, default=0.1, help='dropout probability')

        return parser


class ScheduledOptim():
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
