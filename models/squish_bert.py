import argparse
import logging
import os

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from models.attention.block import BlockMultiHeadedAttention
from models.base_model import BaseModel, BaseModule
from models.embedding.bert import BERTEmbedding
from models.modules.feed_forward import PositionwiseFeedForward
from models.modules.sublayer_connection import LayerNorm, SublayerConnection


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
        self.linear = nn.Linear(hidden, hidden)
        self.act = nn.GELU()
        self.layer_norm = LayerNorm(hidden)
        self.decoder = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.linear(x)
        x = self.act(x)
        x = self.layer_norm(x)
        return self.decoder(x)


class RetrainedBlock(BaseModule):
    def __init__(self,
                 config,
                 depth: int,
                 hidden: int,
                 heads: int,
                 dk: int,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden = hidden
        self.heads = heads
        self.device = config.device
        self.d_k = dk
        self.dropout = dropout
        self.depth = depth

        self.conf = config

        self.attentionblock = BlockMultiHeadedAttention(
            self.heads, self.hidden, d_k=self.d_k, dropout=self.dropout
        )
        # self.output_linear = nn.Linear(self.hidden, self.hidden)

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

        self.to(config.device)

    def forward(self, x):
        return self.attentionblock(x,x,x)

    def load_state(self, load_optimizer: bool = True):
        if True:
            path = (
                    self.conf.storage_directory + '/models/_checkpoints/' + self.conf.dataset + '/block_'
                    + str(self.depth) + '_' + str(self.d_k) + '_' + str(self.heads) + '/'
                    + 'BLOCK-latest.pth'
            )
            tmp = path

            print(f"Block Path: {path}")
            if os.path.exists(path):
                try:
                    from collections import OrderedDict

                    checkpoint = torch.load(path)
                    print("Loaded block checkpoint")
                    try:
                        state_dict = checkpoint['model_state_dict']
                        if self.conf.train and load_optimizer:
                            opt_state_dict = checkpoint['optimizer_state_dict']

                            new_opt_state_dict = OrderedDict()
                            for k, v in state_dict.items():

                                if k[0:7] != 'module.':
                                    new_opt_state_dict = opt_state_dict
                                    break
                                else:
                                    name = k[7:]  # remove `module.`
                                    new_opt_state_dict[name] = v

                            self.optimizer.load_state_dict(new_opt_state_dict)

                    except KeyError:
                        state_dict = checkpoint
                        logging.warning('Could not access ["model_state_dict"] for {t}, this is expected for foreign models'
                                        .format(t=tmp))
                        exit()

                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():

                        if k[0:7] != 'module.':
                            new_state_dict = state_dict
                            break
                        else:
                            name = k[7:]  # remove `module.`
                            new_state_dict[name] = v

                    try:
                        self.load_state_dict(new_state_dict)
                        print('Successfully loaded state dict for block model')
                    except Exception as e:
                        logging.warning('Failed to load state dict into model\n{e}'
                                        .format(e=e))
                        exit()

                    try:
                        self.epoch = checkpoint['epoch']
                    except KeyError as e:
                        logging.warning('Failed to load epoch from state dict, epoch is set to 0:\n{e}'
                                        .format(e=e))
                        self.epoch = 0
                        exit()

                except RuntimeError as e:
                    logging.warning('Failed to load state dict into model. No State was loaded and model is initialized'
                                    'randomly. Epoch is set to 0:\n{e}'
                                    .format(e=e))
                    self.epoch = 0
                    exit()

            else:
                if self.conf.block_model_checkpoint != '':
                    logging.warning('Could not find a state dict for block model at the location specified.')
                self.epoch = 0
                exit()


class RetrainedTransformer(BaseModule):
    def __init__(self,
                 config,
                 depth: int,
                 hidden: int,
                 heads: int,
                 dk: int,
                 dropout: float = 0.1):
        super().__init__()
        self.depth = depth
        self.hidden = hidden
        self.attn_heads = heads
        self.device = config.device
        self.d_k = dk
        self.dropout = dropout

        self.conf = config

        self.attentionblock = BlockMultiHeadedAttention(
            self.attn_heads, self.hidden, d_k=self.d_k, dropout=self.dropout
        )

        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=self.hidden * 4, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

        self.to(config.device)

    def forward(self, x):
        x = self.input_sublayer(x, lambda _x: self.attentionblock.forward(_x, _x, _x))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class SquishBert(BaseModel):
    def __init__(self,
                 config,
                 vocab_size: int,
                 dks=[16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
                 heads=[12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]):
        super(SquishBert, self).__init__(config)
        assert len(dks) == len(heads)
        self.dks = dks
        self.heads = heads

        self.hidden = config.hidden_features
        self.n_layers = config.layers
        self.attn_heads = config.heads
        self.device = config.device

        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=self.hidden).to(self.conf.device)

        layers = []
        for index in range(len(dks)):
            transformer = RetrainedTransformer(config, index, self.hidden, heads[index], dks[index], config.dropout)
            layers.append(transformer)

        self.layers = nn.Sequential(*layers)

        self.mask_lm = MaskedLanguageModel(self.hidden, vocab_size).to(self.conf.device)

        self.optimizer = Adam(
            self.parameters(),
            lr=self.conf.lr,
            betas=(self.conf.adam_beta1, self.conf.adam_beta2),
            weight_decay=self.conf.adam_weight_decay
        ) if config.train else None

    def forward(self, x, segment_info):
        x = self.embedding(x, segment_info)
        x = self.layers(x)
        x = self.mask_lm(x)
        return x

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
